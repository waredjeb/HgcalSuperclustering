from functools import cached_property
from typing import Literal
from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from .assocs import *
from .tracksters import *
from .recoToSim import *
from .simToReco import *

from typing import List

import awkward as ak
print(ak.__version__)

def superclustersToEnergy(supercls_ts_idxs:ak.Array, tracksters:ak.Array) -> ak.Array:
    """ Computes the total energy of a supercluster from an array of supercluster ids 
    Preserves the inner index (usually CP id)
    Parameters :
     - supercls_ts_idxs : type nevts * var * var * uint64
    Returns : type nevts * var * float (energy sum)
    """
    # FIrst flatten the inner dimension (CP id) before taking tracksters
    energies_flat = tracksters.raw_energy[ak.flatten(supercls_ts_idxs, axis=-1)]
    # Rebuild the inner index
    energies = ak.unflatten(energies_flat,  ak.flatten(ak.num(supercls_ts_idxs, axis=-1)), axis=-1)

    return ak.sum(energies, axis=-1)



    
class DumperReader:
    class MultiFileReader:
        def __init__(self, files:list[uproot.ReadOnlyDirectory]):
            self.files = files
        
        def __getitem__(self, key):
            excpts = []
            for file in self.files:
                try:
                    return file[key]
                except uproot.KeyInFileError as e:
                    print(f"File error {file.file_path}")
                    excpts.append(e)
            raise KeyError(*excpts)
            
    def __init__(self, file: Union[str, uproot.ReadOnlyDirectory, List[uproot.ReadOnlyDirectory]], directoryName: str = "ticlDumper") -> None:
        try:
            self.fileDir = file[directoryName]
        except TypeError:
            try:
                self.fileDir = self.MultiFileReader([f[directoryName] for f in file])
            except TypeError:
                self.fileDir = uproot.open(file + ":" + directoryName)
    
    @property
    def nEvents(self):
        if any("trackstersCLUE3DHigh" in entry for entry in self.fileDir.files[0].keys()):       
            return self.fileDir["trackstersCLUE3DHigh"].num_entries
        else:
            return self.fileDir["trackstersclue3d"].num_entries

    @cached_property
    def tracksters(self) -> ak.Array:
        # Determine which key to use based on the fileDir structure
        if any("trackstersCLUE3DHigh" in entry for entry in self.fileDir.files[0].keys()):       
            tracksters_data = self.fileDir["trackstersCLUE3DHigh"].arrays()
        else:
            tracksters_data = self.fileDir["trackstersclue3d"].arrays()

        # Filter out fields that start with "vertices_"
        
        filtered_fields = [field for field in tracksters_data.fields if not field.startswith("vertices_")]
        filtered_fields = [field for field in filtered_fields if not field.startswith("id_probabilities")]
        filtered_fields = [field for field in filtered_fields if not field.startswith("eVector0_")]
        filtered_tracksters_data = tracksters_data[filtered_fields]
        return filtered_tracksters_data
    
    @cached_property
    def tracksters_zipped(self) -> ak.Array:
        return ak.zip({"ts_id" : ak.local_index(self.tracksters.raw_energy, axis=1)} | 
                      {key : self.tracksters[key] for key in self.tracksters.fields 
                        if key not in ["vertices_multiplicity","event", "NClusters", "NTracksters"]},
            depth_limit=2, # don't try to zip vertices
            with_name="tracksters"
        )
    @cached_property
    def simTracksters_zipped(self) -> ak.Array:
        return ak.zip({"ts_id" : ak.local_index(self.simTrackstersCP.raw_energy, axis=1)} | 
                      {key : self.simTrackstersCP[key] for key in self.simTrackstersCP.fields 
                        if key not in ["vertices", "vertices_multiplicity","event", "NClusters", "NTracksters"]},
            depth_limit=2, # don't try to zip vertices
            with_name="simTrackstersCP"
    )
    @cached_property
    def trackstersMerged(self) -> ak.Array:
        if any("trackstersmerged" in entry for entry in self.fileDir.files[0].keys()):
            return self.fileDir["trackstersmerged"].arrays()
        else:
            return self.fileDir["trackstersTiclCandidate"].arrays()

    @cached_property
    def trackstersMerged_zipped(self) -> ak.Array:
        # Create the base dictionary
        base_dict = {"ts_id": ak.local_index(self.trackstersMerged.raw_energy, axis=1)}
        # Update the base dictionary with the other fields
        base_dict.update({key: self.trackstersMerged[key] for key in self.trackstersMerged.fields 
                          if key not in ["vertices_multiplicity", "event", "NClusters", "NTracksters"]})
        return ak.zip(
            base_dict,
            depth_limit=2,  # don't try to zip vertices
            with_name="trackstersTiclCandidate"
        ) 

    @cached_property
    def tracksters_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.tracksters, levelname=lambda x : {0:"eventInternal", 1:"ts_id"}[x])
    @cached_property
    def simTrackstersCP(self) -> ak.Array:
        return self.fileDir["simtrackstersCP"].arrays(filter_name=["event_", "raw_energy", "raw_energy_em", "regressed_energy", "barycenter_*"])
    @cached_property
    def simTrackstersCP_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.simTrackstersCP, levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x])
    @cached_property
    def simTrackstersCP2Hits(self) -> ak.Array:
        return self.fileDir["simtracksters2HitsCP"].arrays(filter_name=["event_", "raw_energy", "raw_energy_em", "regressed_energy", "barycenter_*"])
    @cached_property
    def simTrackstersCP2Hits_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.simTrackstersCP2Hits, levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id"}[x])
    
    @cached_property
    def superclusters(self) -> ak.Array:
        """ Gets the supercluster trackster ids (since ticlv5 this actually includes also tracksters in one-trackster superclusters)
        type: nevts * var (superclsCount) * var (trackstersInSupercls) * uint64 (trackster id)
        """
        return self.fileDir["superclustering/linkedResultTracksters"].array()
    
    @cached_property
    def superclusters_all(self) -> ak.Array:
        """ Supercluster tracksters ids. Tracksters not in a supercluster are included in a one-trackster supercluster each
        Since ticlv5 this is actually identical to superclusters
        """
        #return self.fileDir["superclustering/superclusteredTrackstersAll"].array()
        return self.fileDir["superclustering/linkedResultTracksters"].array()

    @cached_property
    def superclusteringDnnScore(self) -> ak.Array:
        ar =  self.fileDir["superclustering/superclusteringDNNScore"].array()
        return ak.zip({"ts_seed":ar["superclusteringDNNScore._0"], "ts_cand":ar["superclusteringDNNScore._1"], "dnnScore":ak.values_astype(ar["superclusteringDNNScore._2"], np.single)/65535.}, with_name="dnnInferencePair")

    @cached_property
    def associations(self) -> ak.Array:
        return self.fileDir["associations"].arrays(filter_name=["event_", "tsCLUE3D_*", 'Mergetracksters_*', 'Mergetstracksters_*', 'ticlCandidate_*'])
    
    
    @cached_property
    def supercluster_df(self) -> pd.DataFrame:
        return ak.to_dataframe(self.superclusters, anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])

    @cached_property
    def supercluster_all_df(self) -> pd.DataFrame:
        """ Same as superclusters_df but tracksters not in a supercluster are included in a one-trackster supercluster each
        """
        return self.supercluster_df
        # return ak.to_dataframe(self.superclusters_all, anonymous="ts_id",
        #     levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x])
    
    @cached_property
    def supercluster_merged_properties_all(self) -> pd.DataFrame:
        """ Dataframe holding supercluster properties (one row per supercluster) 
        
        Tracksters not in a supercluster are included as one-trackster superclusters
        """
        return (supercluster_joinTracksters(self.supercluster_all_df, self.tracksters_zipped[["raw_energy", "regressed_energy"]])
                .groupby(level=["eventInternal", "supercls_id"])
                .agg(
                    raw_energy_supercls=pd.NamedAgg("raw_energy", "sum"),
                    regressed_energy_supercls=pd.NamedAgg("regressed_energy", "sum"),
        ))
    
    @cached_property
    def assocs_bestScore_simToReco_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster 
        
        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToRecoMerged(self.associations))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )
    
    @cached_property
    def assoc_filterScoreLower_recoToSim_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster 
        
        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_recoToSim_scores = assocs_filterScoreLess(assocs_zip_recoToSim(self.associations))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_recoToSim_scores[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )
    @cached_property
    def assocs_allScores_simToReco_df(self) -> pd.DataFrame:
        """
        Make a DataFrame of all associations of each SimTrackster, sorted by score.

        Index: eventInternal, ts_id
        Columns: caloparticle_id, score, sharedE
        """
        # Get all association scores sorted
        assocs_simToReco_allScores = assocs_allScores(assocs_zip_simToReco(self.associations))
        
        # Modify levelname function to handle more levels or provide default names
        def levelname(x):
            level_map = {0: "eventInternal", 1: "caloparticle_id_wrong"}
            return level_map.get(x, f"level_{x}")
        
        # Make a DataFrame out of it: index eventInternal, ts_id, column: caloparticle_id
        return (ak.to_dataframe(assocs_simToReco_allScores[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=levelname)
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )
    @cached_property
    def assocs_allScores_recoToSim_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a DataFrame of all associations of each Trackster based on recoToSim score.
        
        Parameters : 
            - dropOnes : if True, do not include associations with score 1 (worst score).
            
        Index:
            - eventInternal
            - caloparticle_id
        Columns:
            - ts_id
            - score
            - sharedE
        """
        # Get all association scores in the recoToSim direction
        assocs = assocs_allScores(assocs_zip_recoToSim(self.associations))
        
        # If dropOnes is True, filter out associations with score == 1
        if dropOnes:
            assocs = assocs[assocs.score < 1]

        # Function to set proper level names
        def levelname(x):
            level_map = {0: "eventInternal", 1: "ts_id_wrong"}
            return level_map.get(x, f"level_{x}")

        # Convert the associations to a DataFrame
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]],
                                levelname=levelname)
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
                ) 
           
    
    @cached_property
    def assocs_bestScore_simToReco2Hits_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster 
        
        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToRecoMerged(self.associations, simVariant="CP2Hits"))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        df = (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True))
        
        return df
    def assocs_bestScore_recoToSim_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster 
        
        Parameters : 
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )
    def assocs_bestScore_recoToSim2Hits_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster 
        
        Parameters : 
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )
        
    @cached_property
    def assocs_bestScore_simToRecoMerged_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster 
        
        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToRecoMerged(self.associations))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )
    @cached_property
    def assocs_bestScore_simToRecoMerged2Hits_df(self) -> pd.DataFrame:
        """ Make a Df of largest score associations of each SimTrackster 
        
        Index eventInternal	ts_id, column : caloparticle_id
        """
        # Get largest association score
        assocs_simToReco_largestScore = assocs_bestScore(assocs_zip_simToRecoMerged(self.associations, simVariant="CP2Hits"))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return  (ak.to_dataframe(assocs_simToReco_largestScore[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"caloparticle_id_wrong"}[x])
                .reset_index("caloparticle_id_wrong", drop=True)
                .set_index("caloparticle_id", append=True)
        )

    @cached_property
    def assocs_bestScore_simToRecoShared_df(self) -> pd.DataFrame:
        """Make a DataFrame of associations of each SimTrackster with shared energy > 0.
        
        Index: eventInternal, ts_id
        Columns: caloparticle_id
        """
        # Get associations where sharedE > 0
        assocs_simToReco_sharedE = assocs_zip_simToReco(self.associations)
        
        assocs_simToReco_sharedE = assocs_simToReco_sharedE[assocs_simToReco_sharedE["sharedE"] > 0]

        # Make a DataFrame out of it: index eventInternal, ts_id; columns: caloparticle_id
        def levelname(x):
            names = ["eventInternal", "caloparticle_id_wrong"] + [f"level_{i}" for i in range(x - 2)]
            return names[x] if x < len(names) else f"level_{x}"
        return (
            ak.to_dataframe(assocs_simToReco_sharedE[["ts_id", "caloparticle_id", "score", "sharedE"]],
                            levelname=levelname)
            .reset_index("caloparticle_id_wrong", drop=True)
            .set_index("caloparticle_id", append=True)
        )
    @cached_property
    def assocs_bestScore_simToRecoShared2Hits_df(self) -> pd.DataFrame:
        """Make a DataFrame of associations of each SimTrackster with shared energy > 0.
        
        Index: eventInternal, ts_id
        Columns: caloparticle_id
        """
        # Get associations where sharedE > 0
        assocs_simToReco_sharedE = assocs_zip_simToReco(self.associations, simVariant="CP2Hits")
        
        assocs_simToReco_sharedE = assocs_simToReco_sharedE[assocs_simToReco_sharedE["sharedE"] > 0]

        # Make a DataFrame out of it: index eventInternal, ts_id; columns: caloparticle_id
    
        def levelname(x):
            names = ["eventInternal", "caloparticle_id_wrong"] + [f"level_{i}" for i in range(x - 2)]
            return names[x] if x < len(names) else f"level_{x}"
        return (
            ak.to_dataframe(assocs_simToReco_sharedE[["ts_id", "caloparticle_id", "score", "sharedE"]],
                            levelname=levelname)
            .reset_index("caloparticle_id_wrong", drop=True)
            .set_index("caloparticle_id", append=True)
        )

    
    def assocs_bestScore_recoMergedToSim_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster 
        
        Parameters : 
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoMergedToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )
    def assocs_bestScore_recoMergedToSim2Hits_df(self, dropOnes=True) -> pd.DataFrame:
        """ Make a Df of largest score associations of each Trackster 
        
        Parameters : 
         - dropOnes : if True, do not include associations of score 1 (worst score)
        Index eventInternal	caloparticle_id, column : ts_id, score, sharedE
        """
        # Get largest association score
        assocs = assocs_bestScore((assocs_dropOnes if dropOnes else lambda x:x)(assocs_zip_recoMergedToSim(self.associations)))
        # Make a df out of it : index eventInternal	ts_id, column : caloparticle_id
        return (ak.to_dataframe(assocs[["ts_id", "caloparticle_id", "score", "sharedE"]], 
                                levelname=lambda x : {0:"eventInternal", 1:"ts_id_wrong"}[x])
                .reset_index("ts_id_wrong", drop=True)
                .set_index("ts_id", append=True)
        )
class Step3Reader:
    """ Reads step3 split EDM files """
    def __init__(self, file: Union[str, uproot.ReadOnlyDirectory]) -> None:
        try:
            self.eventsTree = file["Events"]
            self.file = file
        except TypeError:
            self.file = uproot.open(file)
            self.eventsTree = self.file["Events"]


class FWLiteDataframesReader:
    """ Reads pandas dataframe made by the FWLite step3 -> pandas dumper """
    def __init__(self, folder:str, sampleNb:int, readerForEventMapping:DumperReader) -> None:
        self.folder = folder
        self.sampleNb = sampleNb
        self.eventMapping = ak.to_dataframe(readerForEventMapping.tracksters.event_, levelname=lambda x:"eventInternal", anonymous="event_").reset_index()
        """ Dataframe mapping eventInternal (the index into the TICLDumper file) and the actual CMS event number (event_) """
    
    def readDataframe(self, name:str) -> pd.DataFrame:
        return pd.read_pickle(Path(self.folder) / f"{name}_{self.sampleNb}.pkl.gz")
    
    @cached_property
    def supercls(self) -> pd.DataFrame:
        return pd.merge(self.readDataframe("supercls"), self.eventMapping, on="event_").set_index(["eventInternal", "supercls_id", "ts_in_supercls_id"])

#ticlTracksters_ticlTracksterLinksSuperclustering_CLUE3D_RECO./ticlTracksters_ticlTracksterLinksSuperclustering_CLUE3D_RECO.obj/ticlTracksters_ticlTracksterLinksSuperclustering_CLUE3D_RECO.obj.regressed_energy_

