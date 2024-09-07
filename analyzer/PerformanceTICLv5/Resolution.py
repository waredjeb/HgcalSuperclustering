import sys
sys.path.append("../..")
from functools import partial
from typing import Literal

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import hist

from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties, CP2HitstoTrackster_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.stats import norm
import ROOT
import json 
from collections import defaultdict
import subprocess

# Ensure ROOT doesn't try to display canvases
ROOT.gROOT.SetBatch(True)

# Define the Crujiff function in ROOT
def crujiff(x, params):
    A = params[0]
    m = params[1]
    sigma_left = params[2]
    sigma_right = params[3]
    alpha_left = params[4]
    alpha_right = params[5]
    res = 0
    t = (x[0] - m) 
    if t < 0:
        t = (x[0] - m)/sigma_left 
        res = A*math.exp(-0.5 * t * t / (1 + 0.5*alpha_left * t * t))
    else:
        t = (x[0] - m)/sigma_right
        res = A*math.exp(-0.5 * t * t / (1 + 0.5*alpha_right * t * t))
    return res
        

# Function to compute effective sigma
def effective_sigma(fit_func, mean, sigma, *params):
    def integrand(x):
        return fit_func(x, *params)
    
    total_area, _ = quad(integrand, mean - 5 * sigma, mean + 5 * sigma)
    
    def target_func(x):
        return quad(integrand, mean - x, mean + x)[0] - 0.68 * total_area
    
    effective_sigma = sigma
    for i in range(10):
        sigma_guess = norm.ppf(0.84) * sigma
        effective_sigma = sigma_guess
        if target_func(sigma_guess) < 0.01 * total_area:
            break
    return effective_sigma


etaBinsText = ["2p2"]
inputPaths =  [
    "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/Resolution_0GeV_window0p072_15LCs_PostGeomFixLinearParametersSplittingTest1TighterCEE/",
    "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/Resolution_0GeV_window0p072_15LCs_PostGeomFixLinearParametersSplittingTest1TighterCEE/"
             ]
OutputDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/Resolution/CloseByPion200PU_0GeV_window0p072_15LCs_PostGeomFixLinearParametersSplittingTest1TighterCEE/"


jsonRes = "range_fits.json"
MergedListFits = [[],[]]
MergedListHistos = [[], []]
histoDirs = ['histo', 'histoV4']
labels = ['V5', 'V4']
dfs = []
dfs2Hits = []
annotations = ['CloseByPion 200PU']
for ih, (histoDir, label) in enumerate(zip(histoDirs, labels)):
    dfEta = []
    dfEta2Hits = []
    inputPath = inputPaths[ih]
    for etaText in etaBinsText:
        folderPrefix = "CloseByPionPU_"+ etaText 
        results = process_directories_with_prefix(inputPath, folderPrefix, histoDir, limitFile=None)
        readers = []
        for i,r in enumerate(results):
            readers.extend(r.inputReaders)

        binsEdges = []

        pattern = rf"{etaText}_(\d+)"
        for res in readers:
            match = re.search(pattern, res.__repr__())
            binEnergy = float(match.group(1))
            binsEdges.append(binEnergy)

        histedges_equalN = np.unique(np.array(binsEdges))
        res = runComputations([CPtoTracksterMerged_properties, CP2HitstoTrackster_properties], readers, max_workers=24 )
        df = res[0]
        dfEta.append(df)
        dfEta2Hits.append(res[1])
        bin_edgesEnergy = [(i-0.5,i+0.5) for i in histedges_equalN ]
        it = 0
        histos = []
        histosCLUE3D = []
        # for  minE, maxE in bin_edgesEnergy:
        #     histo = make_scOrTsOverCP_energy_histogram(name=f"tsOverCP_energy{label}",minEn = minE, maxEn = maxE, label=f"Trackster Merged - {label} / CaloParticle energy")
        #     fill_scOverCP_energy_histogram(histo, df, minE, maxE)
        #     histos.append(histo)


        # # histo_fit = fitMultiHistogram(histos)
        # MergedListFits[ih].append(histo_fit)
        # MergedListHistos[ih].append(histos)
    dfs.append(dfEta)
    dfs2Hits.append(dfEta2Hits)




# Define the bin intervals
bin_intervals = [(9, 11), (19, 21), (29, 31), (49, 51), (99, 101), (199, 201), (299, 301), (399, 401), (599, 601)]

def categorize_energy(value):
    for interval in bin_intervals:
        if interval[0] <= value < interval[1]:
            return interval
    return None


##Single Plots
Effs = []
Erreffs = []
Energies = []
Etas = []
Labels = []
resultsFit = []

dfsTot = [dfs, dfs2Hits]
# dfsTot = [dfs]
suffixes = ["", "2Hits"]

# Load the JSON file
with open(jsonRes, 'r') as f:
    fit_params_file = json.load(f)
fit_params = fit_params_file["fits"]
print(f"Fit Params {fit_params} {len(fit_params)}")

for d in range(len(dfsTot)):
    for i in range(len(labels)):
            
        oututDirResolutionRawEnergies = create_directory(OutputDir + f"/tracksterMerged{suffixes[d]}/RawEnergies/{labels[i]}")
        oututDirResolution = create_directory(OutputDir + f"/tracksterMerged{suffixes[d]}/Response/{labels[i]}")
        oututDirResolutionSharedE = create_directory(OutputDir + f"/tracksterMerged{suffixes[d]}/ResponseSharedE/{labels[i]}")
        oututDirFit = create_directory(OutputDir + f"/tracksterMerged{suffixes[d]}/Fits/{labels[i]}")
        DF = dfsTot[d]
        for ieta in range(len(DF[i])):
            print(f"d {d} i {i} ieta {ieta}")

            # Group the data by energy_bin
            df = DF[i][ieta]
        
            df['energy_bin'] = df['regressed_energy_CP'].apply(categorize_energy)

            # Group by energy_bin and count the number of entries in each bin
            grouped = df.groupby('energy_bin')
            bin_counts = grouped.size()

            # # Select the top 3 bins with the highest number of entries
            # top_bins = bin_counts.nlargest(3).index

            # # Calculate the scaling factor using the top 3 bins
            # scaling_factor = df[df['energy_bin'].isin(top_bins)]['raw_energy'].sum() / df[df['energy_bin'].isin(top_bins)]['raw_energy_CP'].sum()

            # # Apply the scaling factor to normalize raw_energy
            # df['scaled_energy'] = df['raw_energy'] / scaling_factor





            # Group by the new bin column
            grouped = df.groupby('energy_bin')
            effs = []
            erreffs = []
            energies = []
            etas = []
            for ig,(name, group) in enumerate(grouped):

                # print(f"Mean Initial {meanInitial} stdInitial {stdInitial}")
                histPre = ROOT.TH1F(f"preScalehist_{name[0]}_{name[1]}", "", 50, min(group['raw_energy']), max(group['raw_energy']))
                for value in group['raw_energy']:
                    histPre.Fill(value)
                # Define the quantiles you want to calculate
                quantiles = np.zeros(2)  # Array to store the quantiles
                probabilities = np.array([0.25, 0.75])  # 25th and 75th percentiles
                #now take all the bins between the quantiles, take the mean and find the scaling factor
                # Calculate the quantiles
                histPre.Scale(1. / histPre.Integral())
                histPre.GetQuantiles(2, quantiles, probabilities)
                # Output the quantiles
                # print(f"25th percentile (Q1): {quantiles[0]}")
                # print(f"75th percentile (Q3): {quantiles[1]}")
                #now take all the bins between the quantiles
                energies_between_quantiles = group['raw_energy'].values
                energies_between_quantiles = energies_between_quantiles[(energies_between_quantiles > quantiles[0]) & (energies_between_quantiles < quantiles[1])]
                meanPreScaling = np.mean(energies_between_quantiles)
                # Calculate the scaling factor such that mean is at regressed_energy
                en = (int(name[0]) + int(name[1])) / 2
                scaling_factor = meanPreScaling / en
                group['scaled_energy'] = group['raw_energy'] / scaling_factor

                # groupEffOnly = group[group['sharedE']/group['raw_energy_CP'] >= 0.5]
                groupEffOnly = group

                plt.figure(figsize = (10,10))
                plt.hist(group.raw_energy, range = (0,max(group.raw_energy_CP)),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]} {suffixes[d]}", histtype = "step", lw = 2, color = 'red')
                plt.xlabel("Energy [GeV]")
                plt.ylabel("Entries")
                plt.savefig(f"{oututDirResolutionSharedE}/EffOnlyEnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.savefig(f"{oututDirResolutionRawEnergies}/EnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()

                plt.figure(figsize = (10,10))
                plt.hist(group.raw_energy / group.raw_energy_CP, range = (0,2),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
                plt.xlabel("Response")
                plt.ylabel("Entries")
                plt.savefig(f"{oututDirResolutionSharedE}/EffOnlyEnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.savefig(f"{oututDirResolution}/Responses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()

                plt.figure(figsize = (10,10))
                plt.hist(groupEffOnly.raw_energy, range = (0,max(groupEffOnly.raw_energy_CP)),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
                plt.xlabel("Energy [GeV]")
                plt.ylabel("Entries")
                plt.savefig(f"{oututDirResolutionSharedE}/EffOnlyEnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.savefig(f"{oututDirResolutionRawEnergies}/EffOnlyEnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()

                plt.figure(figsize = (10,10))
                plt.hist(groupEffOnly.raw_energy / groupEffOnly.raw_energy_CP, range = (0,2),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
                plt.xlabel("Response")
                plt.ylabel("Entries")
                plt.savefig(f"{oututDirResolutionSharedE}/EffOnlyEnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.savefig(f"{oututDirResolution}/EffOnlyResponses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()

                plt.figure(figsize = (10,10))
                plt.hist(groupEffOnly.raw_energy, range = (0,max(groupEffOnly.raw_energy_CP)),  bins = 100, label = f"Raw - TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
                plt.hist(groupEffOnly.sharedE, range = (0,max(groupEffOnly.raw_energy_CP)),  bins = 100, label = f"Shared - TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'blue')
                plt.xlabel("Energy [GeV]")
                plt.ylabel("Entries")
                plt.legend()
                plt.savefig(f"{oututDirResolutionSharedE}/EffOnlyEnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()

                plt.figure(figsize = (10,10))
                plt.hist(groupEffOnly.raw_energy/ groupEffOnly.raw_energy_CP, range = (0,2),  bins = 100, label = f"Raw - TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
                plt.hist(groupEffOnly.sharedE/ groupEffOnly.raw_energy_CP, range = (0,2),  bins = 100, label = f"Shared - TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'blue')
                plt.xlabel("Response")
                plt.ylabel("Entries")
                plt.legend()
                plt.savefig(f"{oututDirResolutionSharedE}/EffOnlyResponses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()

                sharedOverRaw = group.sharedE / group.raw_energy_CP
                N = len(sharedOverRaw)
                print((sharedOverRaw >= 0.5).sum(), N, (sharedOverRaw >= 0.5).sum() / N)
                eff =  (sharedOverRaw >= 0.5).sum() / N            
                eff_error = np.sqrt(eff * (1 - eff) / N)
                effs.append(eff)
                erreffs.append(eff_error)
                en = (int(name[0]) + int(name[1])) / 2
                energies.append(en)
                
                scaled_energy = groupEffOnly['scaled_energy'].values
                
                # ROOT TF1 definition
                

                # # Initial parameters for the fit [alphaL, alphaR, nL, nR, sigma, mean]
                # meanInitial = np.mean(scaled_energy)
                # stdInitial = np.std(scaled_energy)

                # sigmaMultipl = 3
                # initial_params = [0.05, meanInitial, stdInitial, stdInitial, -0.2, 0.1]
                # crujiff_function = ROOT.TF1("crujiff", crujiff, meanInitial-sigmaMultipl*stdInitial, meanInitial+sigmaMultipl*stdInitial, len(initial_params))

                # # print(f"Mean Initial {meanInitial} stdInitial {stdInitial}")
                #use fit_params to extract minX and maxX
                minX = fit_params[ig]["minX"]
                maxX = fit_params[ig]["maxX"]
                hist = ROOT.TH1F(f"hist_{name[0]}_{name[1]}", "", 70, minX, maxX)
                for value in scaled_energy:
                    hist.Fill(value)
                # Define the quantiles you want to calculate
                quantiles = np.zeros(2)  # Array to store the quantiles
                probabilities = np.array([0.25, 0.75])  # 25th and 75th percentiles

                # Calculate the quantiles
                hist.Scale(1. / hist.Integral())
                hist.GetQuantiles(2, quantiles, probabilities)

                # Output the quantiles
                print(f"25th percentile (Q1): {quantiles[0]}")
                print(f"75th percentile (Q3): {quantiles[1]}")

                # Fit the histogram with the Crujiff function
                # try:
                #     fit_result = hist.Fit(crujiff_function, "SM")
                # except:
                #     print(f"Fit for {en} {etaBinsText[ieta]} failed")
                    # Optionally, convert the histogram to a numpy array for further plotting with matplotlib
                bin_centers = np.array([hist.GetBinCenter(i) for i in range(1, hist.GetNbinsX() + 1)])
                hist_values = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX() + 1)])
                hist_err = np.array([hist.GetBinError(i) for i in range(1, hist.GetNbinsX() + 1)])
                # # Extract fit parameters
                # A, m, nL, nR, alphaL, alphaR = initial_params

                # chi2 = 999999
                # ndf = 0
                    

                # eff_sigma = (quantiles[1] - quantiles[0]) / 2
                # nR = m - quantiles[0]
                # nL = quantiles[1] - m
                eff_sigma = getEffSigma(hist)
                sigmaOverE = eff_sigma / np.mean(np.mean(group.raw_energy_CP))
                err_sigmaOverE = sigmaOverE / np.sqrt(2*len(scaled_energy))
                argmaxBin = np.argmax(hist_values)
                maxEnergyBin = bin_centers[argmaxBin]

                plt.figure(figsize=(10, 10))
                plt.axvline(x=maxEnergyBin - eff_sigma)
                plt.axvline(x=maxEnergyBin + eff_sigma)
                plt.errorbar(bin_centers, hist_values, yerr=hist_err, xerr=None, fmt='o', label=f'TICL{labels[i]}-{en}GeV-{etaBinsText[ieta]}', color = "blue")
                plt.annotate(text=f'$\mu$={meanPreScaling:.2f}, $\sigma_{{eff}}$={eff_sigma:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
                # plt.plot(x_fit, [crujiff([x], initial_params) for x in x_fit], '-', label=f'Fit $\mu$={m:.2f},$\sigma_L$={nL:.2f}, $\sigma_R$={nR:.2f} $\sigma$={sigmaM:.3f} $\sigma_{{eff}}$={eff_sigma:.2f} $\chi^2$/ndf={chi2:.2f}/{ndf} $\\alpha_L$={alphaL:.3f}, $\\alpha_R$={alphaR:.3f}', color = 'red')
                plt.xlabel('Energy [GeV]')
                plt.ylabel('Entries')
                plt.legend(fontsize = 12)
                plt.savefig(f"{oututDirFit}/FailedFit_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
                plt.close()
                # Store the results
                result = {
                    "label": labels[i] + "_" + suffixes[d],
                    "ieta": etaBinsText[ieta],
                    "energy_bin": name,
                    "mu": meanPreScaling,
                    "sigma_eff": eff_sigma,
                    "sigmaOverE": sigmaOverE,
                    "err_sigmaOverE": err_sigmaOverE
                }
                resultsFit.append(result)
            print(effs)
            Effs.append(effs)
            Erreffs.append(erreffs)
            Energies.append(energies)
            Etas.append(etaBinsText[ieta])
            Labels.append(labels[i]+"_"+suffixes[d])



    # Organize data by label and ieta
    grouped_results = defaultdict(lambda: defaultdict(list))

    for result in resultsFit:
        label = result['label']
        ieta = result['ieta']
        energy = (int(result['energy_bin'][0]) + int(result['energy_bin'][1])) / 2
        sigmaOverE = result['sigmaOverE']
        grouped_results[label][ieta].append((energy, sigmaOverE))

    ieta_grouped_results = {}
    for label, ieta_dict in grouped_results.items():
        for ieta, energy_sigma_pairs in ieta_dict.items():
            if ieta not in ieta_grouped_results:
                ieta_grouped_results[ieta] = {}
            ieta_grouped_results[ieta][label] = energy_sigma_pairs

    # Plotting
    for ieta, label_dict in ieta_grouped_results.items():
        plt.figure(figsize=(10, 8))
        for label, energy_sigma_pairs in label_dict.items():
            energy_sigma_pairs.sort()  # Sort by energy
            energies, sigmaOverEs = zip(*energy_sigma_pairs)            
            plt.errorbar(energies, sigmaOverEs, yerr=err_sigmaOverE, fmt='o', label=f"TICL-{label} $\eta$ = {ieta.replace('p','.')}", capsize = 5)
        
        plt.xlabel('Energy [GeV]')
        plt.ylabel('σ/E')
        plt.legend(title=annotations[0])
        plt.grid(True)
        plt.ylim(bottom=0.)
        plt.savefig(f"{OutputDir}/tracksterMerged{suffixes[d]}/Response/{ieta}_Resolution.png")
        plt.close()

    oututDirComparison = create_directory(OutputDir + f"/tracksterMerged{suffixes[d]}/Comparison/")
    plt.figure(figsize = (10,10))
    for e, ee, en, eta, l in zip(Effs, Erreffs, Energies, Etas, Labels):
        plt.errorbar(en, e, yerr = ee, label = f"TICL{l}-{eta}")
        plt.xlabel("Energies [GeV]")
        plt.ylabel("Efficiency")
        plt.ylim(0., 1.)
        plt.legend()
    plt.savefig(f"{oututDirComparison}/EfficiencyComparison.png")

#execute in bash pb_copy_index.py -r /eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/Resolution/CloseByPion0PU_0GeV_window0p072_15LCs_PostGeomFix/ -c
output_dir = OutputDir
subprocess.run(["pb_copy_index.py", "-r", output_dir])

#same thing for the 2Hits


# ##Single Plots
# for i in range(len(labels)):
#     oututDirResolutionRawEnergies = create_directory(OutputDir + f"/tracksterMerged/Response/{labels[i]}")
#     oututDirResolution = create_directory(OutputDir + f"/tracksterMerged/Response/{labels[i]}")
#     for ieta in range(len(dfs[i])):

#         # Group by the bins
#         df = dfs[i][ieta]
#         df['energy_bin'] = df['regressed_energy_CP'].apply(categorize_energy)

#         # Group by the new bin column
#         grouped = df.groupby('energy_bin')
#         for name, group in grouped:
#             plt.figure(figsize = (10,10))
#             plt.hist(group.raw_energy / group.raw_energy_CP, range = (0,2),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
#             plt.savefig(f"{oututDirResolutionRawEnergies}/EnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
#             plt.close()

#             plt.figure(figsize = (10,10))
#             plt.hist(group.raw_energy, range = (0,2),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
#             plt.savefig(f"{oututDirResolution}/Responses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
#             plt.close()            

# ##Single Plots
# oututDirResolutionRawEnergies = create_directory(OutputDir + f"/tracksterMerged/RawEnergies/Comparison/")
# oututDirResolution = create_directory(OutputDir + f"/tracksterMerged/Responses/Comparison")
# for ieta in range(len(etaBinsText)):
#   for i in range(len(labels)):
  
#           # Group by the bins
#           df = dfs[i][ieta]
#           df['energy_bin'] = df['regressed_energy_CP'].apply(categorize_energy)
  
#           # Group by the new bin column
#           grouped = df.groupby('energy_bin')
#           for name, group in grouped:
#               plt.figure(figsize = (10,10))
#               plt.hist(group.raw_energy, range = (0,name[1]),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
#               plt.savefig(f"{oututDirResolutionRawEnergies}/EnergyDistributions_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
#               plt.close()
  
#               plt.figure(figsize = (10,10))
#               plt.hist(group.raw_energy, range = (0,name[1]),  bins = 100, label = f"TICL{labels[i]} {etaBinsText[ieta]}", histtype = "step", lw = 2, color = 'red')
#               plt.savefig(f"{oututDirResolution}/Responses_{name[0]}_{name[1]}_{etaBinsText[ieta]}.png")
#               plt.close()
