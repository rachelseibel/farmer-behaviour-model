#------------------------------------------------------------------------------------------
# Load the required libraries
#------------------------------------------------------------------------------------------
import numpy as np
import random as rnd
import pandas as pd
import geopandas as gpd
import os

#------------------------------------------------------------------------------------------
# MAIN FUNCTION
#------------------------------------------------------------------------------------------
# Summary functions
def summarise_temporal_by_generation(df, generation_time, output_options, output_types):
    df['t'] = (df['t'] // generation_time) * generation_time
    agg_cols = {short: 'mean' for short in output_options.values()}
    output = df.groupby(['landscape_method', 'county_seed', 'county', 'total_holdings', 'total_cattle', 't']).agg(agg_cols)
    for key, typ in output_types.items():
        col = output_options[key]
        output[col] = output[col]
    return output.reset_index()

def summarise_temporal_threshold_exceedance(df, output_types, threshold_dict, output_options, generation_time):
    group_cols = ['landscape_method', 'county_seed', 'county', 'total_holdings', 'total_cattle', 't']
    # Get output for 0:max:generation_time
    df = df[df['t'] % generation_time == 0]
    # Prepare output
    results = []
    # Group once
    grouped = df.groupby(group_cols)
    for group_key, group in grouped:
        # Get county totals for this group
        total_holdings = group['total_holdings'].iloc[0]
        total_cattle = group['total_cattle'].iloc[0]
        for output, typ in output_types.items():
            col = output_options[output]
            thresholds = threshold_dict[output]
            # Compute absolute thresholds
            if typ == 'holdings':
                abs_thresholds = total_holdings * thresholds / 100
            else:
                abs_thresholds = total_cattle * thresholds / 100
            # Get values for this output
            values = group[col].values
            # Determine the percentage of values exceeding each threshold
            percent_exceeding = [(values > abs_thr).mean() * 100 for abs_thr in abs_thresholds]
            # Round to 2 decimal places
            percent_exceeding = [round(percent, 2) for percent in percent_exceeding]
            for thr, abs_thr, percent in zip(thresholds, abs_thresholds, percent_exceeding):
                results.append({
                    'landscape_method': group_key[0],
                    'county_seed': group_key[1],
                    'county': group_key[2],
                    't': group_key[5],
                    'output': output,
                    'threshold_percent': thr,
                    'threshold_absolute': abs_thr,
                    'percent_exceeding': percent
                })
    return pd.DataFrame(results)

def summarise_threshold_exceedance_by_county(df, output_types, threshold_dict, output_options):
    group_cols = ['landscape_method', 'county_seed', 'county', 'total_holdings', 'total_cattle']
    # Get data from max t for each landscape iteration and seed iteration
    df = df[df['t'] == df.groupby(['landscape_method', 'county_seed', 'county', 'landscape_iteration', 'seed_iteration', 'total_holdings', 'total_cattle'])['t'].transform('max')]
    # Prepare output
    results = []
    # Group once
    grouped = df.groupby(group_cols)
    for group_key, group in grouped:
        # Get county totals for this group
        total_holdings = group['total_holdings'].iloc[0]
        total_cattle = group['total_cattle'].iloc[0]
        for output, typ in output_types.items():
            col = output_options[output]
            thresholds = threshold_dict[output]
            # Compute absolute thresholds
            if typ == 'holdings':
                abs_thresholds = total_holdings * thresholds / 100
            else:
                abs_thresholds = total_cattle * thresholds / 100
            # Get values for this output
            values = group[col].values
            # Determine the percentage of values exceeding each threshold
            percent_exceeding = [(values > abs_thr).mean() * 100 for abs_thr in abs_thresholds]
            # Round to 2 decimal places
            percent_exceeding = [round(percent, 2) for percent in percent_exceeding]
            for thr, abs_thr, percent in zip(thresholds, abs_thresholds, percent_exceeding):
                results.append({
                    'landscape_method': group_key[0],
                    'county_seed': group_key[1],
                    'county': group_key[2],
                    'output': output,
                    'threshold_percent': thr,
                    'threshold_absolute': abs_thr,
                    'percent_exceeding': percent
                })
    return pd.DataFrame(results)

def summarise_threshold_exceedance(df, output_types, threshold_dict, output_options):    
    # Get data from max t for each landscape iteration and seed iteration
    df = df[df['t'] == df.groupby(['landscape_method', 'county_seed', 'county', 'landscape_iteration', 'seed_iteration', 'total_holdings', 'total_cattle'])['t'].transform('max')]
    # For each landscape method, county seed, landscape iteration and seed iteration, sum by county
    # Drop county
    df = df.drop(columns=['county'])
    df = df.groupby(['landscape_method', 'county_seed', 'landscape_iteration', 'seed_iteration']).sum().reset_index()
    # Prepare output
    results = []
    # Group once
    grouped = df.groupby(['landscape_method', 'county_seed'])
    for group_key, group in grouped:
        # Get county totals for this group
        total_holdings = group['total_holdings'].iloc[0]
        total_cattle = group['total_cattle'].iloc[0]
        for output, typ in output_types.items():
            col = output_options[output]
            thresholds = threshold_dict[output]
            # Compute absolute thresholds
            if typ == 'holdings':
                abs_thresholds = total_holdings * thresholds / 100
            else:
                abs_thresholds = total_cattle * thresholds / 100
            # Get values for this output
            values = group[col].values
            # Determine the percentage of values exceeding each threshold
            percent_exceeding = [(values > abs_thr).mean() * 100 for abs_thr in abs_thresholds]
            # Round to 2 decimal places
            percent_exceeding = [round(percent, 2) for percent in percent_exceeding]
            for thr, abs_thr, percent in zip(thresholds, abs_thresholds, percent_exceeding):
                results.append({
                    'landscape_method': group_key[0],
                    'county_seed': group_key[1],
                    'output': output,
                    'threshold_percent': thr,
                    'threshold_absolute': abs_thr,
                    'percent_exceeding': percent
                })
    return pd.DataFrame(results)


def summarise_duration_exceedance(df, duration_thresholds):
    # Group by simulation and get max t (duration) for each
    durations = df.groupby([
        'landscape_method', 'county_seed', 'county', 'landscape_iteration', 'seed_iteration'
    ])['t'].max().reset_index()

    results = []
    # Group by landscape_method, county_seed, county for summary
    group_cols = ['landscape_method', 'county_seed', 'county']
    grouped = durations.groupby(group_cols)

    for group_key, group in grouped:
        for threshold in duration_thresholds:
            exceed = group['t'] > threshold
            percent = exceed.mean() * 100
            # Round to 2 decimal places
            percent = round(percent, 2)
            results.append({
                'landscape_method': group_key[0],
                'county_seed': group_key[1],
                'county': group_key[2],
                'duration_threshold': threshold,
                'percent_exceeding': percent,
            })
    return pd.DataFrame(results)

def individual_simulation_summary(df, output_options):
    """
    Summarize each individual simulation with cumulative metrics.
    
    For each unique simulation (county_seed, landscape_method, landscape_iteration, seed_iteration),
    calculate cumulative totals at the final time step across all counties.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with simulation data
    output_options : dict
        Dictionary mapping output names to column names
    
    Returns:
    --------
    DataFrame with columns:
        - landscape_method
        - county_seed
        - landscape_iteration
        - seed_iteration
        - holdings_culled
        - cattle_culled
        - cattle_vax_success
        - cattle_vax_failure
        - holdings_vax_success
        - holdings_vax_failure
        - outbreak_duration_days
    """
    # Get data from max t for each simulation (summing across counties)
    df_max = df[df['t'] == df.groupby([
        'landscape_method', 'county_seed', 'county', 'landscape_iteration', 'seed_iteration'
    ])['t'].transform('max')].copy()
    
    # Sum across counties for each simulation
    group_cols = ['landscape_method', 'county_seed', 'landscape_iteration', 'seed_iteration']
    
    # Define the columns we want to sum
    sum_cols = {
        output_options['Culled']: 'holdings_culled',
        output_options['Culled Cattle']: 'cattle_culled',
        output_options['Vaccine Success Cattle']: 'cattle_vax_success',
        output_options['Vaccine Failure Cattle']: 'cattle_vax_failure',
        output_options['Vaccine Success Holdings']: 'holdings_vax_success',
        output_options['Vaccine Failure Holdings']: 'holdings_vax_failure'
    }
    
    # Aggregate across counties
    df_sum = df_max.groupby(group_cols).agg({
        col: 'sum' for col in sum_cols.keys()
    }).reset_index()
    
    # Rename columns
    df_sum = df_sum.rename(columns=sum_cols)
    
    # Get outbreak duration (max t for each simulation)
    df_duration = df.groupby(group_cols)['t'].max().reset_index()
    df_duration = df_duration.rename(columns={'t': 'outbreak_duration_days'})
    
    # Merge duration with cumulative data
    result = pd.merge(df_sum, df_duration, on=group_cols)
    
    return result

def main():

    # Set up working directory
    # path = '/Users/rachelseibel/'
    path = '/home/rachelseibel/farmer_behaviour/'
    path_summary = '/home/rachelseibel/summary_cluster/'
    ## Organise data into folders
    path_no_vax = path + 'simulations_no_vax/'
    path_idw = path + 'simulations_vax_idw_all/'
    path_softmax = path + 'simulations_vax_softmax_all/'
    county_names_path = path + 'scripts/inputs/holdings_counties.csv'

    # Set parameters
    generation_time = 5
    duration_thresholds = np.arange(10, 510, 10)

    # Paths to simulation sets
    simulation_sets = [path_idw, path_softmax, path_no_vax]

    landscape_methods = ['GB', 'Nation', 'County', 'blocking_method', 'interpolation_method']
    county_data = pd.read_csv(county_names_path)
    counties = county_data['county'].unique()

    # Set the generation time
    generation_time = 5

    output_options = {
        # 'Susceptible': 'S',
        # 'Exposed': 'E',
        # 'Infected': 'I',
        # 'Recovered': 'R',
        'Culled': 'C',
        'Culled Cattle': 'C_cattle',
        'Vaccine Pending Holdings': 'V_pending_holdings',
        'Vaccine Pending Cattle': 'V_pending_cattle',
        'Vaccine Success Holdings': 'V_success_holdings',
        'Vaccine Success Cattle': 'V_success_cattle',
        'Vaccine Failure Holdings': 'V_failure_holdings',
        'Vaccine Failure Cattle': 'V_failure_cattle'
    }

    output_types = {
        # 'Susceptible': 'holdings',
        # 'Exposed': 'holdings',
        # 'Infected': 'holdings',
        # 'Recovered': 'holdings',
        'Culled': 'holdings',
        'Culled Cattle': 'cattle',
        'Vaccine Pending Holdings': 'holdings',
        'Vaccine Pending Cattle': 'cattle',
        'Vaccine Success Holdings': 'holdings',
        'Vaccine Success Cattle': 'cattle',
        'Vaccine Failure Holdings': 'holdings',
        'Vaccine Failure Cattle': 'cattle'
    }

    # Pre-set thresholds for each output and mode
    default_thresholds = np.arange(1, 100, 1)
    default_thresholds = np.concatenate((np.arange(0.1, 1, 0.1), default_thresholds))
    threshold_dict = {k: default_thresholds for k in output_options.keys()}

    holdings_by_county = county_data.groupby('county').size().reset_index(name='holdings')
    cattle_by_county = county_data.groupby('county')['cattle'].sum().reset_index()
    county_totals = pd.merge(holdings_by_county, cattle_by_county, on='county')
    county_totals = county_totals.rename(columns={'holdings': 'total_holdings', 'cattle': 'total_cattle'})

    for simulation_set in simulation_sets:
        print(f'Processing simulation set: {simulation_set}')
        for landscape_method in landscape_methods:
            if (simulation_set != path_no_vax and landscape_method != 'GB'):
                print(f'Processing method: {landscape_method}')
                for county in counties:
                    print(f'Processing county: {county}')
                    folder_path = os.path.join(simulation_set + landscape_method + '/', county)
                    if not os.path.exists(folder_path):
                        continue
                    files = os.listdir(folder_path)
                    filtered_files = [f for f in files if 'landscapeIteration' in f and f.endswith('.csv')]
                    if not filtered_files:
                        continue

                    # Check if each summary file already exists - if so, skip (summary files 1-6)
                    summary_files = [
                        os.path.join(path_summary, f'summary1/summary_temporal_{landscape_method}_{county}.csv'),
                        os.path.join(path_summary, f'summary2/summary_temporal_threshold_{landscape_method}_{county}.csv'),
                        os.path.join(path_summary, f'summary3/summary_duration_{landscape_method}_{county}.csv'),
                        os.path.join(path_summary, f'summary4/summary_cumulative_by_county_{landscape_method}_{county}.csv'),
                        os.path.join(path_summary, f'summary5/summary_cumulative_{landscape_method}_{county}.csv'),
                        os.path.join(path_summary, f'summary6/individual_simulations_{landscape_method}_{county}.csv')
                    ]
                    if all(os.path.exists(file) for file in summary_files):
                        print(f'Skipping {county} - summary files already exist')
                        continue

                    # Read and concatenate all files for this county
                    df_list = []
                    for file in filtered_files:
                        file_path = os.path.join(folder_path, file)
                        df = pd.read_csv(file_path)
                        if not df.empty:
                            df_list.append(df)
                    if not df_list:
                        continue
                    df_all = pd.concat(df_list, ignore_index=True)
                    # Merge with county totals
                    df_all = df_all.merge(county_totals, on='county', how='left')

                    if simulation_set == path_no_vax:
                        tmp = 'no_vax/'
                    elif simulation_set == path_idw:
                        tmp = 'idw/'
                    elif simulation_set == path_softmax:
                        tmp = 'softmax/'

                    # 1. Temporal summary by generation
                    if not os.path.exists(os.path.join(path_summary+tmp, f'summary1/summary_temporal_{landscape_method}_{county}.csv')):
                        print(f'Calculating temporal summary for {county}')
                        summary1 = summarise_temporal_by_generation(df_all, generation_time, output_options, output_types)
                        summary1.to_csv(os.path.join(path_summary+tmp, f'summary1/summary_temporal_{landscape_method}_{county}.csv'), index=False)

                    # 2. Threshold exceedance summary
                    if not os.path.exists(os.path.join(path_summary+tmp, f'summary2/summary_temporal_threshold_{landscape_method}_{county}.csv')):
                        print(f'Calculating temporal threshold exceedance summary for {county}')
                        summary2 = summarise_temporal_threshold_exceedance(df_all, output_types, threshold_dict, output_options, generation_time)
                        summary2.to_csv(os.path.join(path_summary+tmp, f'summary2/summary_temporal_threshold_{landscape_method}_{county}.csv'), index=False)

                    # 3. Outbreak duration exceedance summary
                    # Skip this step if the summary file already exists
                    if not os.path.exists(os.path.join(path_summary+tmp, f'summary3/summary_duration_{landscape_method}_{county}.csv')):
                        print(f'Calculating duration exceedance summary for {county}')
                        summary3 = summarise_duration_exceedance(df_all, duration_thresholds)
                        summary3.to_csv(os.path.join(path_summary+tmp, f'summary3/summary_duration_{landscape_method}_{county}.csv'), index=False)

                    # 4. Cumulative summary
                    # Skip this step if the summary file already exists
                    if not os.path.exists(os.path.join(path_summary+tmp, f'summary4/summary_cumulative_by_county_{landscape_method}_{county}.csv')):
                        print(f'Calculating cumulative summary by county for {county}')
                        summary4 = summarise_threshold_exceedance_by_county(df_all, output_types, threshold_dict, output_options)
                        summary4.to_csv(os.path.join(path_summary+tmp, f'summary4/summary_cumulative_by_county_{landscape_method}_{county}.csv'), index=False)

                    # 5. Cumulative summary across all counties
                    # Skip this step if the summary file already exists
                    if not os.path.exists(os.path.join(path_summary+tmp, f'summary5/summary_cumulative_{landscape_method}_{county}.csv')):
                        print(f'Calculating cumulative summary for {county}')
                        summary5 = summarise_threshold_exceedance(df_all, output_types, threshold_dict, output_options)
                        summary5.to_csv(os.path.join(path_summary+tmp, f'summary5/summary_cumulative_{landscape_method}_{county}.csv'), index=False)

                    # 6. Individual simulation summary
                    # Skip this step if the summary file already exists
                    if not os.path.exists(os.path.join(path_summary+tmp, f'summary6/individual_simulations_{landscape_method}_{county}.csv')):
                        print(f'Calculating individual simulation summary for {county}')
                        summary6 = individual_simulation_summary(df_all, output_options)
                        summary6.to_csv(os.path.join(path_summary+tmp, f'summary6/individual_simulations_{landscape_method}_{county}.csv'), index=False)
    return

if __name__ == "__main__":
    main()
    print("Post-processing completed successfully.")