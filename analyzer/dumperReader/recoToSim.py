import pandas as pd
import awkward as ak
from typing import Union
from .tracksters import supercluster_joinTracksters
from .tracksters import trackster_joinSupercluster, tracksters_groupBy, _convertTsToDataframe



def superclusterToSim_df(supercluster_df:pd.DataFrame, assocs_bestScore_recoToSim_df:pd.DataFrame, tracksters_df:pd.DataFrame) -> pd.DataFrame:
    """ Make Df of tracksters in superclusters joined with recoToSim associations and trackster information
    
    Index : eventInternal	supercls_id	ts_in_supercls_id	
    Columns : ts_id	simts_id	score	sharedE	raw_energy	regressed_energy"""
    df = supercluster_df.join(assocs_bestScore_recoToSim_df(), on=["eventInternal", "ts_id"])
    df.score = df.score.fillna(1)
    df.sharedE = df.sharedE.fillna(0)
    
    return supercluster_joinTracksters(df, tracksters_df)

# def TracksterToCPProperties_allScores(
#     assocs_allScores_recoToSim_df: pd.DataFrame,
#     simTrackstersCP: Union[ak.Array, pd.DataFrame],
#     tracksters: pd.DataFrame
# ) -> pd.DataFrame:
#     """ 
#     For each CaloParticle, get the properties of all associated tracksters.
    
#     Parameters:
#      - assocs_allScores_recoToSim_df : DataFrame with all associations from reco to SimTracksters.
#      - tracksters : DataFrame (or zipped awkward array) of tracksters with properties to keep.
#      - simTrackstersCP_df : DataFrame with properties of SimTracksters (CaloParticles).
     
#     Returns:
#      - DataFrame with CaloParticle properties joined with all associated trackster properties.
#     """
#     # Step 1: Convert tracksters to DataFrame if it's an awkward array
#     # if isinstance(tracksters, ak.Array):
#     #     tracksters_df = _convertTsToDataframe(tracksters)
#     # else:
#     #     tracksters_df = tracksters
#     # print(simTrackstersCP_df)
#     # # Step 2: Join trackster properties with the association dataframe
#     # joined_df = assocs_allScores_recoToSim_df.join(
#     #     tracksters_df, on=["eventInternal", "ts_id"]
#     # )

#     # Step 3: Join with SimTrackster (CaloParticle) properties
#     print(simTrackstersCP)

#     df = (assocs_allScores_recoToSim_df
#         .join(simTrackstersCP, on=["eventInternal", "caloparticle_id"])
#         .join(tracksters, rsuffix="_RECO"))
#     return df

def TracksterToCPProperties_allScores(assocs_allScores_recoToSim_df: pd.DataFrame, 
                            tracksters: Union[ak.Array, pd.DataFrame], 
                            simTrackstersCP_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each CaloParticle, get all associated trackster properties.
    
    Parameters:
     - assocs_allScores_recoToSim_df : DataFrame containing associations from recoToSim.
     - tracksters : DataFrame or awkward array of tracksters with properties to keep.
     - simTrackstersCP_df : DataFrame of simTracksters (CaloParticles) properties to join.
    
    Returns:
     - A DataFrame with the properties of associated tracksters for each CaloParticle.
    """
    # Convert tracksters to a DataFrame if it's an awkward array
    tracksters_df = _convertTsToDataframe(tracksters)
    
    # Perform the joins
    result_df = (assocs_allScores_recoToSim_df
                 .join(tracksters_df, on=["eventInternal", "ts_id"])
                 .join(simTrackstersCP_df, rsuffix="_CP"))
    
    return result_df


