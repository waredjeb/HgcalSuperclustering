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
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path

fileV5 =  "./ROOTALE/"
fileV4 =  "./ROOTALE/"
fileV5 =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/ComparisonPR/histoBase/"
fileV4 =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/ComparisonPR/histoV4/"
#fileV5Old = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/ComparisonPR/histoBase/"
#fileV5Old =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/ComparisonPR/histoBase/"
#fileV5Old =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByElectronPU/histo/"
#fileV5 =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByElectronPU/histo/"
#fileV4 =  "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/CloseByElectronPU/histoV4/"
#fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/ComparisonPR/histoV4/"
#fileV5Old = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/EnergyRegressionTICLV5New/CMSSW_14_X/D99/CloseByElectron_v5/histo/"


OutputDir = "/eos/user/w/wredjeb/www/HGCAL/TICLv5Performance/CloseByPionPU200V5vsV4TEST/"
#OutputDir = "./TestAle"
create_directory(OutputDir)

# Define the DumperInput objects
dumperInputs = [
    DumperInputManager([fileV5], limitFileCount=1),
    DumperInputManager([fileV4], limitFileCount=1)
]

# Run computations for each DumperInput object
results = [runComputations([CPtoTrackster_properties, CPtoTracksterMerged_properties], dumperInput, max_workers=24) for dumperInput in dumperInputs]

mergedResults = [result[1] for result in results]
mergedResults.extend([result[0] for result in results])
#mergedV5 = resV5[1]
#mergedV4 = resV4[1]
fig = plt.figure(figsize = (15,10))
# Define plot parameters
energyBins = 100
etaBins = 50
etaBins = 50
colors = ['red', 'blue', 'green','orange', 'black']
labels = ['TICLv5', 'TICLv4', 'TICLv5C3D', 'TICLv4C3D', 'SimTrackster']
#labels = ['TICLv5', 'TICLv4', 'TICLv4', 'SimTracksters']

# Create output directory
outputDirTracksterMerged = create_directory(OutputDir + "/tracksterMerged/")
#### BestRECO Plots #####
#print(mergedV5)
outputDirTracksterMerged = create_directory(OutputDir + "/tracksterMerged/")
fig = plt.figure(figsize=(15, 10))
norm = True
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy, range=(0, 600), bins=energyBins, label=f"{label} ({len(merged)})", histtype="step", lw=2, color=color, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(np.abs(merged.barycenter_eta), range = (1.7, 2.7), bins = etaBins, label = f"{label} [{len(merged)}]", histtype = "step", lw = 2, color = color, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(np.abs(merged.barycenter_eta), range = (-np.pi, np.pi), bins = etaBins, label = f"{label} [{len(merged)}]", histtype = "step", lw = 2, color = color, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoeta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy / merged.regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = f"{label} [{len(merged)}]", histtype = "step", lw = 2, color = color, density = norm)
plt.hist(mergedResults[0].raw_energy_CP / mergedResults[0].regressed_energy_CP, range = (0, 1.5), bins = energyBins, label = f"SimTracksters {[len(mergedResults[0].raw_energy_CP)]}", histtype = "step", lw = 2, color = 'black', density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "BestRecoEnergyResponse.png")
plt.close()

### Efficient Reco Trackster Plots
# Filter the data based on the condition raw_energy/regressed_energy_CP >= 0.5\
filteredMergedResults = [m[m['raw_energy'] / m['regressed_energy_CP'] >= 0.5] for m in mergedResults]
#filtered_data_V5 = mergedV5[mergedV5['raw_energy'] / mergedV5['regressed_energy_CP'] >= 0.5]
#filtered_data_V4 = mergedV4[mergedV4['raw_energy'] / mergedV4['regressed_energy_CP'] >= 0.5]
#filteredMergedResults = [filtered_data_V5, filtered_data_V4]
fig = plt.figure(figsize = (15,10))
print(labels[:-1])
for merged, color, label in zip(filteredMergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy, range = (0,600),  bins = energyBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(filteredMergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy,range = (1.7, 2.7),  bins = etaBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(filteredMergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy, range = (-np.pi, np.pi),  bins = etaBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoeta.png")
plt.close()

for merged, color, label in zip(filteredMergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy / merged.regressed_energy_CP,range = (0, 1.5),  bins = energyBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirTracksterMerged + "EfficientRecoEnergyResponse.png")
plt.close()


##### SimTrackster Plots

outputDirSimTracksters = create_directory(OutputDir + "/simTracksters/")
fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy_CP ,range = (0,600),  bins = energyBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Raw Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRawEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.regressed_energy_CP ,range = (0,600),  bins = energyBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Regressed Energy [Gev]")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimRegressedEnergy.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.barycenter_eta_CP ,range = (1.7, 2.7), bins = etaBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.barycenter_eta_CP ,range = (-np.pi, np.pi), bins = etaBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("eta")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "Simeta.png")
plt.close()

fig = plt.figure(figsize = (15,10))
for merged, color, label in zip(mergedResults, colors[:-1], labels[:-1]):
    plt.hist(merged.raw_energy_CP / merged.regressed_energy_CP , range = (0, 1.5),  bins = energyBins, label = f"{label} {[len(merged)]}", histtype = "step", lw = 2, color = {color}, density = norm)
plt.legend()
plt.ylabel("Entries")
plt.xlabel("Response w.r.t Regressed")
hep.cms.text("Simulation", loc=0)
plt.savefig(outputDirSimTracksters + "SimEnergyResponse.png")
plt.close()

#### Responses ####
# Get the "viridis" colormap
import matplotlib as mpl

cmap = mpl.colormaps['Set1']

# Create a custom colormap with the desired number of colors
custom_cmap = ListedColormap(cmap(np.linspace(0, 1, 10)))

# Set the color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=custom_cmap.colors)

bins = [0, 20, 50, 100, 200 ,400]

outputDirTracksterMerged = create_directory(OutputDir + "/tracksterMerged/Responses/")
for d, color, lab in zip(mergedResults, colors[:-1], labels[:-1]):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        ratios = filtered_data_V5.raw_energy / filtered_data_V5.regressed_energy_CP
        counts, bin_edges = np.histogram(ratios, bins=energyBins, range=(0, 1.5))
        bin_width = bin_edges[1] - bin_edges[0]
        normalized_counts = counts / (sum(counts) * bin_width)        
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=normalized_counts, range = (0, 1.5), label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Regressed")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponse" + lab + ".png")
    plt.close()

for d, color, lab in zip(mergedResults, colors[:-1], labels[:-1]):
    fig = plt.figure(figsize = (15,10))
    for b in bins:
        filtered_data_V5 = d[d['regressed_energy_CP'] >= b]
        plt.hist(filtered_data_V5.raw_energy / filtered_data_V5.raw_energy_CP, range = (0, 1.5), bins = energyBins, label = "SimTrackster >= " + str(b) + " GeV", histtype = "step", lw = 2, density = True)
    plt.legend()
    plt.ylabel("Entries")
    plt.xlabel("Raw Response w.r.t Raw")
    hep.cms.text("Simulation", loc=0)
    plt.savefig(outputDirTracksterMerged + "BestRecoRawEnergyResponseWrtRaw" + lab + ".png")
    plt.close()


## Efficiencies ##
outputDirEffFakeMergeDup = create_directory(OutputDir + "/tracksterMerged/EffFakeMergeDup/")
filtered_data_merged = []
for m in mergedResults:
    filtered_data_merged.append(m[m['sharedE'] / m['raw_energy_CP'] >= 0.5])
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)

for data, f_data, lab in zip(mergedResults, filtered_data_merged, labels):
    plot_ratio_single(f_data.raw_energy_CP, data.raw_energy_CP, energyBins, rangeX = (0,600), label1=f"{lab}", xlabel="Raw Energy [GeV]", color1='blue', saveFileName=f"{outputDirEffFakeMergeDup}/efficiency{lab}_energy.png")


nums = [f.raw_energy_CP for f in filtered_data_merged]
dens = [m.raw_energy_CP for m in mergedResults]
plot_ratio_multiple(nums, dens, energyBins, rangeX = (0,600), labels =labels[:-1], colors =colors[:-1], xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyComparisonRatio_energy.png")

etaBins = 20
nums = [f.barycenter_eta_CP for f in filtered_data_merged]
dens = [m.barycenter_eta_CP for m in mergedResults]
for n, d, col, lab in zip(nums,dens, colors[:-1], labels[:-1]):
    plot_ratio_single(n, d, etaBins,  rangeX = (1.7,2.7), xlabel="Eta", label1=f"{lab}", color1=col, saveFileName=f"{outputDirEffFakeMergeDup}/efficiency{lab}_eta.png")
plot_ratio_multiple(nums, dens, etaBins,  rangeX = (1.7,2.7), labels = labels[:-1], colors = colors[:-1], xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyComparisonRatio_eta.png")


phiBins = 20
nums = [f.barycenter_phi_CP for f in filtered_data_merged]
dens = [m.barycenter_phi_CP for m in mergedResults]

for n, d, col, lab in zip(nums,dens, colors[:-1], labels[:-1]):
    plot_ratio_single(n, d, phiBins, rangeX = (-np.pi, np.pi), xlabel="eta", label1=f"{lab}", color1=col, saveFileName=f"{outputDirEffFakeMergeDup}/efficiency{lab}_phi.png")
plot_ratio_multiple(nums, dens, phiBins, rangeX = (-np.pi, np.pi), labels = labels[:-1], colors = colors[:-1], xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyComparisonRatio_phi.png")

Rbins = 20 
nums = [np.sqrt(f.barycenter_x_CP**2 + f.barycenter_y_CP**2) for f in filtered_data_merged]
dens = [np.sqrt(m.barycenter_x_CP**2 + m.barycenter_y_CP**2) for m in mergedResults]

for n, d, col, lab in zip(nums,dens, colors[:-1], labels[:-1]):
    plot_ratio_single(n, d, Rbins, rangeX = (50, 180), xlabel="R", label1=f"{lab}", color1=col, saveFileName=f"{outputDirEffFakeMergeDup}/efficiency{lab}_R.png")
plot_ratio_multiple(nums, dens, Rbins, rangeX = (50, 180), labels = labels[:-1], colors = colors[:-1], xlabel="R", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyComparisonRatio_R.png")


outputDirEffFakeMergeDup = create_directory(OutputDir + "/tracksterMerged/EffFakeMergeDupVarious/")
effTh = [0.4,0.5,0.6,0.7,0.8]
colors = ['blue', 'red', 'green', 'orange', 'purple']
energyBins = np.linspace(0,400, 22)
energyBins = np.append(energyBins, 600)
labelsTh = []
for m, l in zip(mergedResults, labels):
  mergedNum = []
  mergedDen = []
  for th in effTh:
      filtered_data_V5 = m[m['sharedE'] / m['raw_energy_CP'] >= th]
      mergedNum.append(filtered_data_V5.raw_energy_CP)
      mergedDen.append(m.raw_energy_CP)
      labelsTh.append(f"{l}- {th}")
  plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,600), labels = labelsTh, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyEnergy{l}.png", doRatio = False)

etaBins = 20
labelsTh = []
for m, l in zip(mergedResults, labels):
  mergedNum = []
  mergedDen = []
  for th in effTh:
      filtered_data_V5 = m[m['sharedE'] / m['raw_energy_CP'] >= th]
      mergedNum.append(filtered_data_V5.barycenter_eta_CP)
      mergedDen.append(m.barycenter_eta_CP)
      labelsTh.append(f"{l}- {th}")
  plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (1.7, 2.7), labels = labelsTh, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyEta{l}.png", doRatio = False)
phiBins = 20
labelsTh = []
for m, l in zip(mergedResults, labels):
  mergedNum = []
  mergedDen = []
  for th in effTh:
      filtered_data_V5 = m[m['sharedE'] / m['raw_energy_CP'] >= th]
      mergedNum.append(filtered_data_V5.barycenter_phi_CP)
      mergedDen.append(m.barycenter_phi_CP)
      labelsTh.append(f"{l}- {th}")
  plot_ratio_multiple(mergedNum, mergedDen, phiBins, rangeX = (-np.pi, np.pi), labels = labelsTh, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyPhi{l}.png", doRatio = False)

Rbins= 50
labelsTh = []
for m, l in zip(mergedResults, labels):
  mergedNum = []
  mergedDen = []
  for th in effTh:
      filtered_data_V5 = m[m['sharedE'] / m['raw_energy_CP'] >= th]
      mergedNum.append((filtered_data_V5.barycenter_x_CP**2 + filtered_data_V5.barycenter_y_CP**2)**0.5)
      mergedDen.append((m.barycenter_x_CP**2 + m.barycenter_y_CP**2)**0.5)
      labelsTh.append(f"{l}- {th}")
  plot_ratio_multiple(mergedNum, mergedDen, Rbins, rangeX = (40,180), labels = labelsTh, colors = colors, xlabel="Phi", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyR{l}.png", doRatio = False)
#
#
#mergedNum = []
#mergedDen = []
#etaBins = 20
#labels = []
#for th in effTh:
#    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
#    mergedNum.append(filtered_data_V5.barycenter_eta_CP)
#    mergedDen.append(mergedV5.barycenter_eta_CP)
#    labels.append(f"TICLv5 - {th}")
#plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_eta.png", doRatio = False)
#
#mergedNum = []
#mergedDen = []
#Rbins = 50
#labels = []
#for th in effTh:
#    filtered_data_V5 = mergedV5[mergedV5['sharedE'] / mergedV5['raw_energy_CP'] >= th]
#    mergedNum.append((filtered_data_V5.barycenter_x_CP**2 + filtered_data_V5.barycenter_y_CP**2)**0.5)
#    mergedDen.append((mergedV5.barycenter_x_CP**2 + mergedV5.barycenter_y_CP**2)**0.5)
#    labels.append(f"TICLv5 - {th}")
#plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (40,180), labels = labels, colors = colors, xlabel="R", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv5_R.png", doRatio = False)
#
##same thing for v4
#mergedNum = []
#mergedDen = []
#labels = []
#for th in effTh:
#    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
#    mergedNum.append(filtered_data_V4.raw_energy_CP)
#    mergedDen.append(mergedV4.raw_energy_CP)
#    labels.append(f"TICLv4 - {th}")
#print(len(mergedNum))
#plot_ratio_multiple(mergedNum, mergedDen, energyBins, rangeX = (0,600), labels = labels, colors = colors, xlabel="Raw Energy [GeV]", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLv4_energy.png", doRatio = False)
#
#
#mergedNum = []
#mergedDen = []
#etaBins = 20
#labels = []
#for th in effTh:
#    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
#    mergedNum.append(filtered_data_V4.barycenter_eta_CP)
#    mergedDen.append(mergedV4.barycenter_eta_CP)
#    labels.append(f"TICLV4 - {th}")
#plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (1.7,2.7), labels = labels, colors = colors, xlabel="Eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLV4_eta.png", doRatio = False)
#
#
#mergedNum = []
#mergedDen = []
#etaBins = 20
#labels = []
#for th in effTh:
#    filtered_data_V4 = mergedV4[mergedV4['sharedE'] / mergedV4['raw_energy_CP'] >= th]
#    mergedNum.append(filtered_data_V4.barycenter_eta_CP)
#    mergedDen.append(mergedV4.barycenter_eta_CP)
#    labels.append(f"TICLV4 - {th}")
#plot_ratio_multiple(mergedNum, mergedDen, etaBins, rangeX = (-np.pi,np.pi), labels = labels, colors = colors, xlabel="eta", saveFileName=f"{outputDirEffFakeMergeDup}/efficiencyTICLV4_eta.png", doRatio = False)
#
