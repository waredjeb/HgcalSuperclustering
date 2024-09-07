from analyzer.driver.computations import DataframeComputation
from analyzer.dumperReader.reader import DumperReader, tracksters_getSeeds, tracksters_toDf, trackster_basic_fields,simTrackster_basic_fields, CPToTracksterProperties, CPToTracksterProperties_allScores, TracksterToCPProperties, CP2HitsToTracksterProperties
from analyzer.driver.fileTools import SingleInputReader

# cannot use a lambda as multirprocessing does not work due to pickle issues
def _seedTracksterProperties_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return tracksters_toDf(tracksters_getSeeds(reader.tracksters_zipped[trackster_basic_fields]))
tracksters_seedProperties = DataframeComputation(_seedTracksterProperties_fct, "tracksters_seedProperties")

def _CPtoTrackster_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToReco_df, reader.tracksters_zipped[trackster_basic_fields],
            reader.simTrackstersCP_df)
CPtoTrackster_properties = DataframeComputation(_CPtoTrackster_fct, "CPtoTrackster_properties")
def _CP2HitstoTrackster_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CP2HitsToTracksterProperties(reader.assocs_bestScore_simToReco2Hits_df, reader.trackstersMerged_zipped[trackster_basic_fields],
            reader.simTrackstersCP2Hits_df)
CP2HitstoTrackster_properties = DataframeComputation(_CP2HitstoTrackster_fct, "CP2HitstoTrackster_properties")

def _CPtoTracksterAllShared_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToRecoShared_df, reader.tracksters_zipped[trackster_basic_fields],
            reader.simTrackstersCP_df)
CPtoTracksterAllShared_properties = DataframeComputation(_CPtoTracksterAllShared_fct, "CPtoTracksterAllShared_properties")

def _CPtoTracksterMerged_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToRecoMerged_df, reader.trackstersMerged_zipped[trackster_basic_fields],
            reader.simTrackstersCP_df)
CPtoTracksterMerged_properties = DataframeComputation(_CPtoTracksterMerged_fct, "CPtoTracksterMerged_properties")

def _CPtoTrackster_allScores_fct(reader: DumperReader):
    """
    Function to merge CaloParticle to Trackster properties using all associations.
    
    Parameters:
    - reader: DumperReader object with required data for associations, tracksters, and SimTracksters.
    
    Returns:
    - DataFrame with merged CaloParticle and associated trackster properties for all associations.
    """
    reader = reader.ticlDumperReader
    
    # Use CPToTracksterProperties_allScores to handle all associations
    return CPToTracksterProperties_allScores(
        reader.assocs_allScores_simToReco_df,  # DataFrame with all associations
        reader.tracksters_zipped[trackster_basic_fields],  # Tracksters data with selected fields
        reader.simTrackstersCP_df  # CaloParticle (SimTracksters) data
    )
CPtoTracksterAll_properties = DataframeComputation(_CPtoTrackster_allScores_fct, "CPtoTracksterAll_properties")

def _CP2HitstoTracksterMerged_fct(reader:DumperReader):
    reader = reader.ticlDumperReader
    return CPToTracksterProperties(reader.assocs_bestScore_simToRecoMerged2Hits_df, reader.trackstersMerged_zipped[trackster_basic_fields],
            reader.simTrackstersCP2Hits_df)
CP2HitstoTracksterMerged_properties = DataframeComputation(_CP2HitstoTracksterMerged_fct, "CP2HitstoTracksterMerged_properties")

def _TracksterToCP_allScores_fct(reader: DumperReader):
    """
    Function to merge Trackster to CaloParticle properties using all associations.
    
    Parameters:
    - reader: DumperReader object with required data for associations, tracksters, and SimTracksters.
    
    Returns:
    - DataFrame with merged Trackster and associated CaloParticle properties for all associations.
    """
    reader = reader.ticlDumperReader
    
    return TracksterToCPProperties_allScores(
        reader.assocs_allScores_recoToSim_df,  # DataFrame with all associations
        reader.tracksters_df,  # Tracksters data
        reader.simTrackstersCP_df[simTrackster_basic_fields]  # CaloParticle (SimTracksters) data with selected fields
    )

# Define DataframeComputation using the function
TracksterToCPAll_properties = DataframeComputation(_TracksterToCP_allScores_fct, "TracksterToCPAll_properties")


def _TracksterToCP_recoToSimForFake_fct(reader: DumperReader):
    """
    Function to merge CaloParticle to Trackster properties using all associations.
    
    Parameters:
    - reader: DumperReader object with required data for associations, tracksters, and SimTracksters.
    
    Returns:
    - DataFrame with merged CaloParticle and associated trackster properties for all associations.
    """
    reader = reader.ticlDumperReader
    
    # Use CPToTracksterProperties_allScores to handle all associations
    return TracksterToCPProperties(
        reader.assoc_filterScoreLower_recoToSim_df,  # DataFrame with all associations
        reader.tracksters_zipped[trackster_basic_fields],  # Tracksters data with selected fields
        reader.simTrackstersCP_df  # CaloParticle (SimTracksters) data
    )
TracksterToCP_recoToSimForFake = DataframeComputation(_TracksterToCP_recoToSimForFake_fct, "TracksterToCP_recoToSimForFake")

