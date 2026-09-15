#------------------------------------------------------------------------------------------
# Load the required libraries
#------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------------------
# Assign farmers to behavioural groups
#------------------------------------------------------------------------------------------
def assign_farmers_to_behavioural_groups(holdings, params_behav, params_land, landscape_iteration):
    if params_behav['behaviour_flag'] == 1:
        print('Assigning farmers to behavioural groups...')
        # Load the data
        farmers = holdings.copy()
        # Load landscape shapefile
        iteration = int(landscape_iteration)
        farm_count = len(farmers['easting'])
        line_range = [iteration*farm_count, (iteration+1)*farm_count-1]
        gdf = pd.read_csv(params_land['landscape_shp_path'], skiprows=line_range[0], nrows=farm_count, header=0)
        gdf = pd.DataFrame(gdf)
        # Set column names
        gdf.columns = ['nation', 'region', 'county', 'cattle', 'easting', 'northing', 'iteration', 'cluster_id', 'sample_id']
        # Specify order of columns
        # If there is a column named 'county_shp' then rename it to 'county'
        if 'county_shp' in gdf.columns:
            gdf.rename(columns={'county_shp': 'county'}, inplace=True)

        # Assign group based on county, easting and northing
        farmers = pd.merge(farmers, gdf[['easting', 'northing', 'cluster_id']], on=['easting', 'northing'], how='inner')
        farmers['behavioural_group'] = farmers['cluster_id']

        # Based on behavioural group and vaccination_by_group, assign the epidemic stage the farmer will vaccinate
        farmers['vaccination_distance'] = farmers.apply(choose_vaccination_stage, args=(params_behav,), axis=1)
        return farmers['behavioural_group'].to_numpy(int), farmers['vaccination_distance'].to_numpy()
    if params_behav['behaviour_flag'] == 2:
        # Assign all farmers to the same group
        print('Assigning farmers to the same group...')
        # Load the data
        farmers = holdings.copy()
        farmers['behavioural_group'] = 1
        # Assign the same vaccination distance to all farmers
        farmers['vaccination_distance'] = params_behav['epidemic_stage_distances'][0]
        return farmers['behavioural_group'].to_numpy(int), farmers['vaccination_distance'].to_numpy()
    else:
        print('Behaviour not set')
        return None

def choose_vaccination_stage(row, params_behav):
    group = int(row['behavioural_group'])
    stage = np.random.choice(params_behav['epidemic_stage_names'], p=[params_behav['vaccination_by_group'][group]['early'], params_behav['vaccination_by_group'][group]['mid'], params_behav['vaccination_by_group'][group]['late'], params_behav['vaccination_by_group'][group]['never']])
    # Get distance for each stage
    if stage == 'early':
        stage = params_behav['epidemic_stage_distances'][0]
    elif stage == 'mid':
        stage = params_behav['epidemic_stage_distances'][1]
    elif stage == 'late':
        stage = params_behav['epidemic_stage_distances'][2]
    elif stage == 'never':
        stage = params_behav['epidemic_stage_distances'][3]
    return stage