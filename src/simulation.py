#------------------------------------------------------------------------------------------
# Load the required libraries
#------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import binom
import os
import csv
from scipy.spatial import cKDTree

import landscape
import vaccination
import seeding
import behaviour

import math

def one_minus_exp(x):
    """
    Return solution to "1 - exp(x)" using a numerically stable approach.
    If x is very small, directly computing 1 - exp(x) can be inaccurate.
    """
    if x == 0.0:
        return 0.0
    elif abs(x) < 1e-5:  # Use Taylor expansion for small x
        return -(x + 0.5 * x * x)
    else:
        return -(math.exp(x) - 1.0)

# Define the set of fixed probabilities
binomial_lookup_ps = [0,5.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# def generate_binomial_random_variate(n, p, lookup_ps):
#     """Generate a binomial random variate based on n and p, rounded to nearest p in lookup_ps."""
#     # p_rounded = round_to_nearest_p(p, lookup_ps)
#     n_over = np.random.binomial(n, p)
#     return n_over, p

#------------------------------------------------------------------------------------------
# Simulate an outbreak
#------------------------------------------------------------------------------------------
def simulate_outbreak(rng, event_file, cells, holdings, cell_to_cell_transmiss_prob, kernel_lookup, distances_lookup, params_epi, params_land, params_behav, params_sim, params_intervention, params_out, binomial_RNG, P_CS, seed_iteration, landscape_iteration):
    #------------------------------------------------------------------------------------------
    # Unpack parameters
    #------------------------------------------------------------------------------------------
    max_time = params_sim['max_time']

    #------------------------------------------------------------------------------------------
    # Initialise variables
    #------------------------------------------------------------------------------------------
    num_holdings = len(holdings)
    num_cells = len(cells)

    day_infectious = params_epi['day_infectious']
    day_reported = params_epi['day_reported']
    day_culled = params_epi['day_culled']
    vaccine_effectiveness_delay = params_epi['vaccine_effectiveness_delay']
    
    num_events = day_culled
    num_vax_events = vaccine_effectiveness_delay
    # Get row numbers from holdings
    holding_idxs = range(0, len(holdings['cell_id']))
    # Initialise dictionary for iterate inputs
    params_iterate = {
        'num_holdings': num_holdings,
        'num_cells': num_cells,
    }

    events = [np.full(num_holdings, np.nan) for _ in range(num_events+num_vax_events+3)]
    county_names = pd.read_csv(params_sim['county_names_file_path'])
    nation_region_county_names = county_names[['nation', 'region', 'county']]
    # Remove duplicates
    unique_county_names = nation_region_county_names.drop_duplicates()
    # Reset index
    unique_county_names = unique_county_names.reset_index(drop=True)

    # Create holding status arrays
    a = [np.array(holdings['cell_id'].values), np.array(holdings['holding_id'].values)]
    holding_cell = np.array(a)
    # array of -1s
    holding_status = np.full(num_holdings, -1)
    holding_vax_status = np.full(num_holdings, -1)
    holding_county = np.array(holdings['county'])
    holding_cattle = np.array(holdings['cattle'])
    holding_easting = np.array(holdings['easting'])
    holding_northing = np.array(holdings['northing'])
    holding_transmissibility = np.array(holdings['transmissibility'])
    holding_susceptibility = np.array(holdings['susceptibility'])
    cell_status = [np.zeros(num_cells) for _ in range(num_events)]
    cell_status[0] = np.array(cells['num_nodes'])
    cell_vax_status = [np.zeros(num_cells) for _ in range(num_vax_events+2)]
    cell_vax_status[0] = np.array(cells['num_nodes'])

    # Set up events_county - columns are 'landscape_method', 'county_seed', 'landscape_iteration', 'seed_iteration', 'nation', 'region', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'V_S', 'V_E'
    events_county = pd.DataFrame(columns=['nation', 'region', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'C_cattle', 'V_pending_holdings', 'V_pending_cattle', 'V_success_holdings', 'V_success_cattle', 'V_failure_holdings', 'V_failure_cattle'])
    # events_county = pd.DataFrame(columns=['nation', 'region', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'C_cattle' 'V_S', 'V_S_a', 'V_E', 'V_E_a', 'V_S_S', 'V_S_S_a', 'V_S_E', 'V_S_E_a'])
    events_county['nation'] = unique_county_names['nation']
    events_county['region'] = unique_county_names['region']
    events_county['county'] = unique_county_names['county']
    county_names = unique_county_names['county']
    events_county['t'] = np.nan
    events_county['S'] = np.nan
    events_county['E'] = np.nan
    events_county['I'] = np.nan
    events_county['R'] = np.nan
    events_county['C'] = np.nan
    events_county['C_cattle'] = np.nan
    events_county['V_pending_holdings'] = np.nan
    events_county['V_pending_cattle'] = np.nan
    events_county['V_success_holdings'] = np.nan
    events_county['V_success_cattle'] = np.nan
    events_county['V_failure_holdings'] = np.nan
    events_county['V_failure_cattle'] = np.nan
    # events_county['V_S'] = np.nan
    # events_county['V_S_a'] = np.nan
    # events_county['V_E'] = np.nan
    # events_county['V_E_a'] = np.nan
    # events_county['V_S_S'] = np.nan
    # events_county['V_S_S_a'] = np.nan
    # events_county['V_S_E'] = np.nan
    # events_county['V_S_E_a'] = np.nan

    iterate_flag = 1
    t = 0

    # Get county_id from county_names and county_seed
    county_id = unique_county_names.index[unique_county_names['county'] == params_sim['seed_county']].tolist()[0]

    # Seed infection
    events, holding_status, cell_status = seeding.seed_infection(events, t, holdings, holding_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, cell_status, holding_idxs, params_sim, county_id, landscape_iteration, seed_iteration)

    holding_behaviour_group, holding_vaccination_stage = behaviour.assign_farmers_to_behavioural_groups(holdings, params_behav, params_land, landscape_iteration)

    holding_behaviour = [holding_behaviour_group, holding_vaccination_stage]

    # Log behaviour information
    if params_out['output_flag'] == 10:
        # Create a DataFrame for the holding behaviour
        behaviour_df = pd.DataFrame({
            'holding_id': holdings['holding_id'],
            'easting': holding_easting,
            'northing': holding_northing,
            'behaviour_group': holding_behaviour[0],
            'vaccination_stage': holding_behaviour[1]
        })
        # Define the Parquet file path
        behaviour_file = f"{event_file}_behaviour.parquet"
        # Write or append to the Parquet file
        if os.path.isfile(behaviour_file):
            existing_behaviour_df = pd.read_parquet(behaviour_file)
            updated_behaviour_df = pd.concat([existing_behaviour_df, behaviour_df], ignore_index=True)
            updated_behaviour_df.to_parquet(behaviour_file, index=False)
        else:
            behaviour_df.to_parquet(behaviour_file, index=False)
        # exit simulation after logging behaviour
        print("Behaviour information logged. Exiting simulation.")
        return events, events_county, cell_status, holding_behaviour
    
    while ((t < max_time) & (iterate_flag == 1)):
        print('Day: ', t)

        events, events_county, holding_county, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, iterate_flag = iterate(rng, unique_county_names, events, events_county, holding_county, holding_cattle, t, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, cell_to_cell_transmiss_prob, kernel_lookup, distances_lookup, binomial_RNG, P_CS, params_epi, params_iterate, params_intervention, params_sim, params_behav, iterate_flag)

        # Log infection events
        infected_holdings = np.where(holding_status > -1)[0]
        # print(f"Day {t}: Infected Holdings: {len(infected_holdings)}")        
        if len(infected_holdings) != len(set(infected_holdings)):
            print("Error: Duplicate infections detected.")
        # Log counts of susceptible and infectious holdings
        num_susceptible = len(np.where(holding_status == -1)[0])
        num_infectious = len(np.where((holding_status >= params_epi['day_infectious']) & (holding_status < params_epi['day_culled']))[0])
        # print(f"Day {t}: Susceptible: {num_susceptible}, Infectious: {num_infectious}")

        if params_out['output_flag'] == 1:
            tmp = pd.DataFrame(columns=['nation', 'region', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'C_cattle', 'V_pending_holdings', 'V_pending_cattle', 'V_success_holdings', 'V_success_cattle', 'V_failure_holdings', 'V_failure_cattle'])
            tmp['nation'] = unique_county_names['nation']
            tmp['region'] = unique_county_names['region']
            tmp['county'] = unique_county_names['county']
            tmp['t'] = np.nan
            tmp['S'] = np.nan
            tmp['E'] = np.nan
            tmp['I'] = np.nan
            tmp['R'] = np.nan
            tmp['C'] = np.nan
            tmp['C_cattle'] = np.nan
            tmp['V_pending_holdings'] = np.nan
            tmp['V_pending_cattle'] = np.nan
            tmp['V_success_holdings'] = np.nan
            tmp['V_success_cattle'] = np.nan
            tmp['V_failure_holdings'] = np.nan
            tmp['V_failure_cattle'] = np.nan
            # tmp['V_S'] = np.nan
            # tmp['V_S_a'] = np.nan
            # tmp['V_E'] = np.nan
            # tmp['V_E_a'] = np.nan
            # tmp['V_S_S'] = np.nan
            # tmp['V_S_S_a'] = np.nan
            # tmp['V_S_E'] = np.nan
            # tmp['V_S_E_a'] = np.nan

            # For each county, update the events_county dataframe
            for i, county in enumerate(unique_county_names['county']):
                holdings_in_county = np.where(holding_county == county)[0]
                tmp.loc[i, 't'] = t
                tmp.loc[i, 'S'] = len(np.where(holding_status[holdings_in_county] == -1)[0])
                tmp.loc[i, 'E'] = len(np.where((holding_status[holdings_in_county] > -1) & (holding_status[holdings_in_county] < day_infectious))[0])
                tmp.loc[i, 'I'] = len(np.where((holding_status[holdings_in_county] >= day_infectious) & (holding_status[holdings_in_county] < day_reported))[0])
                tmp.loc[i, 'R'] = len(np.where((holding_status[holdings_in_county] >= day_reported) & (holding_status[holdings_in_county] < day_culled))[0])
                tmp.loc[i, 'C'] = len(np.where(holding_status[holdings_in_county] == day_culled)[0])
                tmp.loc[i, 'C_cattle'] = np.sum(holding_cattle[holdings_in_county][np.where(holding_status[holdings_in_county] == day_culled)[0]])
                tmp.loc[i, 'V_pending_holdings'] = len(np.where((holding_vax_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] < params_epi['vaccine_effectiveness_delay']) & (holding_status[holdings_in_county] == -1))[0])
                tmp.loc[i, 'V_pending_cattle'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_vax_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] < params_epi['vaccine_effectiveness_delay']) & (holding_status[holdings_in_county] == -1))[0]])
                tmp.loc[i, 'V_success_holdings'] = len(np.where((holding_vax_status[holdings_in_county] == params_epi['vaccine_effectiveness_delay']) & (holding_status[holdings_in_county] == -1))[0])
                tmp.loc[i, 'V_success_cattle'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_vax_status[holdings_in_county] == params_epi['vaccine_effectiveness_delay']) & (holding_status[holdings_in_county] == -1))[0]])
                tmp.loc[i, 'V_failure_holdings'] = len(np.where((holding_vax_status[holdings_in_county] == 6))[0])
                tmp.loc[i, 'V_failure_cattle'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_vax_status[holdings_in_county] == 6))[0]])
                # tmp.loc[i, 'V_S'] = len(np.where((holding_status[holdings_in_county] == -1) & (holding_vax_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] < vaccine_effectiveness_delay))[0])
                # tmp.loc[i, 'V_S_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] == -1) & (holding_vax_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] < vaccine_effectiveness_delay))[0]])
                # tmp.loc[i, 'V_E'] = len(np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] == 6))[0])
                # tmp.loc[i, 'V_E_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] < vaccine_effectiveness_delay))[0]])
                # tmp.loc[i, 'V_S_S'] = len(np.where((holding_status[holdings_in_county] == -1) & (holding_vax_status[holdings_in_county] == 5))[0])
                # tmp.loc[i, 'V_S_S_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] == 0) & (holding_vax_status[holdings_in_county] == 5))[0]])
                # tmp.loc[i, 'V_S_E'] = len(np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] == 6))[0])
                # tmp.loc[i, 'V_S_E_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] == 6))[0]])
            if ((t == 0)):
                events_county = tmp
            else:
                events_county = pd.concat([events_county, tmp], ignore_index=False)
        if params_out['output_flag'] == 2:
            # Create a DataFrame for the current timestep
            data = {
                't': [t] * num_holdings,
                'holding_id': holdings['holding_id'],
                'easting': holding_easting,
                'northing': holding_northing,
                'status': holding_status
            }
            # print('Number of holdings in each status: ', np.unique(holding_status, return_counts=True))
            df = pd.DataFrame(data)
            # Define the Parquet file path
            parquet_file = f"{event_file}.parquet"
            # Write or append to the Parquet file
            if t == 0:
                if os.path.isfile(parquet_file):
                    os.remove(parquet_file)
                df.to_parquet(parquet_file, index=False)
            else:
                # Append to the existing Parquet file
                existing_df = pd.read_parquet(parquet_file)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_parquet(parquet_file, index=False)

        if params_out['output_flag'] == 3:
            # Create a DataFrame for the current timestep
            data = {
                't': [t] * num_holdings,
                'holding_id': holdings['holding_id'],
                'easting': holding_easting,
                'northing': holding_northing,
                'status': holding_status,
                'vax_status': holding_vax_status,
            }
            df = pd.DataFrame(data)
            # Define the Parquet file path
            parquet_file = f"{event_file}.parquet"
            # Write or append to the Parquet file
            if t == 0:
                if os.path.isfile(parquet_file):
                    os.remove(parquet_file)
                df.to_parquet(parquet_file, index=False)
            else:
                # Append to the existing Parquet file
                existing_df = pd.read_parquet(parquet_file)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_parquet(parquet_file, index=False)

        # if params_out['output_flag'] == 2:
        #     # Append a file with the following columns: t, easting, northing, status
        #     # Check if the file exists
        #     if not os.path.isfile(event_file + str(landscape_iteration) + '_' + str(seed_iteration) + '.csv'):
        #         # Create a new file to write to
        #         with open(event_file + str(landscape_iteration) + '_' + str(seed_iteration) + '.csv', 'w', newline='') as f:
        #             writer = csv.writer(f)
        #             writer.writerow(['t', 'holding_id', 'easting', 'northing', 'status'])
        #             for i in range(num_holdings):
        #                 writer.writerow([t, holdings['holding_id'][i], holding_easting[i], holding_northing[i], holding_status[i]])
        #     else:
        #         # Append to the file
        #         with open(event_file + str(landscape_iteration) + '_' + str(seed_iteration) + '.csv', 'a', newline='') as f:
        #             writer = csv.writer(f)
        #             for i in range(num_holdings):
        #                 writer.writerow([t, holdings['holding_id'][i], holding_easting[i], holding_northing[i], holding_status[i]])

        t = t + 1

    print('Simulation complete at day: ', t)

    return events, events_county, cell_status, holding_behaviour

#------------------------------------------------------------------------------------------
# Iteration function
#------------------------------------------------------------------------------------------
def iterate(rng, county_names, events, events_county, holding_county, holding_cattle, t, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, cell_to_cell_transmiss_prob, kernel_lookup, distances_lookup, binomial_RNG, P_CS, params_epi, params_iterate, params_intervention, params_sim, params_behav, iterate_flag):
    # Log RNG state
    # print(f"RNG State at Day {t}: {rng.bit_generator.state}")

    # # Verify cell-level counts
    # for cell_id in range(len(cell_status[0])):
    #     cell_susceptible_count = cell_status[0][cell_id]
    #     cell_infectious_count = cell_status[6][cell_id]
        
    #     actual_susceptible_count = len(np.where((holding_status == -1) & (holding_cell[0] == cell_id))[0])
    #     actual_infectious_count = len(np.where((holding_status >= params_epi['day_infectious']) & (holding_status < params_epi['day_culled']) & (holding_cell[0] == cell_id))[0])
        
    #     if cell_susceptible_count != actual_susceptible_count:
    #         print(f"Warning: Mismatch in susceptible count for cell {cell_id} on day {t}. Expected: {cell_susceptible_count}, Actual: {actual_susceptible_count}")
    #     if cell_infectious_count != actual_infectious_count:
    #         print(f"Warning: Mismatch in infectious count for cell {cell_id} on day {t}. Expected: {cell_infectious_count}, Actual: {actual_infectious_count}")
    
    # Unpack parameters
    day_infectious = params_epi['day_infectious']
    day_reported = params_epi['day_reported']
    day_culled = params_epi['day_culled']
    vaccine_effectiveness_delay = params_epi['vaccine_effectiveness_delay']
    vaccination_event_ID = params_intervention['vaccination_event_ID']

    if params_intervention['vaccinate_flag'] == 1:
        vaccination_event_ID = params_intervention['vaccination_event_ID']
        vaccination.vaccinate_holdings(t, events, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, vaccination_event_ID, day_infectious, day_reported, params_behav, params_epi)

    update_events(events, events_county, holding_county, t, params_intervention, params_epi)

    # if t > 0:
    #     indices = np.where((holding_status > -1) & (holding_status < day_culled))[0]
    #     # print(f"Indices: {indices}")
    #     holding_status[indices] += 1
    #     # Check this for negative cell status
    #     cell_status = [cell_status[0]] + [np.zeros(len(cell_status[0]))] + cell_status[1:-1]

    #     vax_indices = np.where(((holding_vax_status > 0) & (holding_vax_status < vaccine_effectiveness_delay)))[0]
    #     if len(vax_indices) > 0:
    #         holding_vax_status[vax_indices] += 1

    # tmp = pd.DataFrame(columns=['nation', 'region', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'C_cattle', 'V_S', 'V_S_a', 'V_E', 'V_E_a', 'V_S_S', 'V_S_S_a', 'V_S_E', 'V_S_E_a'])
    # tmp['nation'] = county_names['nation']
    # tmp['region'] = county_names['region']
    # tmp['county'] = county_names['county']
    # tmp['t'] = np.nan
    # tmp['S'] = np.nan
    # tmp['E'] = np.nan
    # tmp['I'] = np.nan
    # tmp['R'] = np.nan
    # tmp['C'] = np.nan
    # tmp['C_cattle'] = np.nan
    # tmp['V_S'] = np.nan
    # tmp['V_S_a'] = np.nan
    # tmp['V_E'] = np.nan
    # tmp['V_E_a'] = np.nan
    # tmp['V_S_S'] = np.nan
    # tmp['V_S_S_a'] = np.nan
    # tmp['V_S_E'] = np.nan
    # tmp['V_S_E_a'] = np.nan

    # # For each county, update the events_county dataframe
    # for i, county in enumerate(county_names['county']):
    #     holdings_in_county = np.where(holding_county == county)[0]
    #     tmp.loc[i, 't'] = t
    #     tmp.loc[i, 'S'] = len(np.where(holding_status[holdings_in_county] == -1)[0])
    #     tmp.loc[i, 'E'] = len(np.where((holding_status[holdings_in_county] > -1) & (holding_status[holdings_in_county] < day_infectious))[0])
    #     tmp.loc[i, 'I'] = len(np.where((holding_status[holdings_in_county] >= day_infectious) & (holding_status[holdings_in_county] < day_reported))[0])
    #     tmp.loc[i, 'R'] = len(np.where((holding_status[holdings_in_county] >= day_reported) & (holding_status[holdings_in_county] < day_culled))[0])
    #     tmp.loc[i, 'C'] = len(np.where(holding_status[holdings_in_county] == day_culled)[0])
    #     tmp.loc[i, 'C_cattle'] = np.sum(holding_cattle[holdings_in_county][np.where(holding_status[holdings_in_county] == day_culled)[0]])
    #     tmp.loc[i, 'V_S'] = len(np.where((holding_status[holdings_in_county] == -1) & (holding_vax_status[holdings_in_county] > 0) & (holding_vax_status[holdings_in_county] < vaccine_effectiveness_delay))[0])
    #     tmp.loc[i, 'V_S_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] == -1) & (holding_vax_status[holdings_in_county] > 0) & (holding_vax_status[holdings_in_county] < vaccine_effectiveness_delay))[0]])
    #     tmp.loc[i, 'V_E'] = len(np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] > 0) & (holding_vax_status[holdings_in_county] == 6))[0])
    #     tmp.loc[i, 'V_E_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] > 0) & (holding_vax_status[holdings_in_county] < vaccine_effectiveness_delay))[0]])
    #     tmp.loc[i, 'V_S_S'] = len(np.where((holding_status[holdings_in_county] == -1) & (holding_vax_status[holdings_in_county] == 5))[0])
    #     tmp.loc[i, 'V_S_S_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] == 0) & (holding_vax_status[holdings_in_county] == 5))[0]])
    #     tmp.loc[i, 'V_S_E'] = len(np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] == 6))[0])
    #     tmp.loc[i, 'V_S_E_a'] = np.sum(holding_cattle[holdings_in_county][np.where((holding_status[holdings_in_county] > -1) & (holding_vax_status[holdings_in_county] == 6))[0]])
    # if ((t == 0)):
    #     events_county = tmp
    # else:
    #     events_county = pd.concat([events_county, tmp], ignore_index=False)

    #################
    # Exit condition
    #################
    # Check if there are any infected holdings, if not, set iterate flag to 0
    infected_holdings = np.where((holding_status > -1) & ((holding_status < day_culled)))[0]
    if len(infected_holdings) == 0:
        iterate_flag = 0

    # Find cells with infectious holdings
    infectious_holdings = np.where((holding_status >= day_infectious) & ((holding_status < day_culled)) & ((holding_vax_status < params_epi['vaccine_effectiveness_delay'])))[0]
    infectious_cells_all = holding_cell[0][infectious_holdings]
    # Get unique infectious cells
    infectious_cells = np.unique(infectious_cells_all)

    num_infectious_cells = len(infectious_cells)

    conditional_infected_count = 0
    local_infected_count = 0

    if num_infectious_cells == 0:
        pass
    else:
        susceptible_holdings = np.where((holding_status == -1) & (holding_vax_status < params_epi['vaccine_effectiveness_delay']))[0]
        susceptible_cells = holding_cell[0][susceptible_holdings]
        # Get unique susceptible cells
        susceptible_cells = np.unique(susceptible_cells)
        # susceptible_cells = np.where(cell_status[0] > 0)[0]
        if (len(susceptible_cells) == 0) | (len(infectious_cells) == 0):
            pass
        else:
            for infectious_cell in infectious_cells:
                # num_infectious_in_cell = len(np.where(holding_cell[0][infectious_holdings] == infectious_cell)[0])
                # print(f"Cell {infectious_cell} has {num_infectious_in_cell} infectious holdings.")
                # print(f"Infectious cell: {infectious_cell}")
                for susceptible_cell in susceptible_cells:
                    # print(f"Susceptible cell: {susceptible_cell}")
                    cell_to_cell_transmiss_prob_ij = cell_to_cell_transmiss_prob[infectious_cell, susceptible_cell]
                    # susceptible_in_cell = np.where((holding_status == -1) & (holding_vax_status < params_epi['vaccine_effectiveness_delay']) & (holding_cell[0] == susceptible_cell))[0]
                    # num_susceptible_in_cell = len(susceptible_in_cell)
                    # print(f"Number of susceptible in cell {susceptible_cell}: {num_susceptible_in_cell}")
                    # num_susceptible_in_cell = cell_status[0][susceptible_cell]
                    if (susceptible_cell == infectious_cell):
                        # print(f"LOCAL CHECK PAIRWISE WITHIN CELL: {susceptible_cell}")
                        local_infected_count = local_pairwise_within_cell(local_infected_count, rng, t, infectious_cell, susceptible_cell, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, events, events_county, holding_county, params_epi, params_intervention, params_iterate, kernel_lookup, distances_lookup, params_sim)
                    else:
                        # print(f"Cell {susceptible_cell} is different from cell {infectious_cell}")
                        # print("CONDITIONAL CHECK BETWEEN CELLS")
                        if cell_to_cell_transmiss_prob_ij > 0:
                            # print(f"Cell {infectious_cell} to cell {susceptible_cell} transmissibility: {cell_to_cell_transmiss_prob_ij}")
                            conditional_infected_count = conditional_subsample(conditional_infected_count, rng, t, infectious_cell, susceptible_cell, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, events, events_county, holding_county, params_epi, params_intervention, params_iterate, cell_to_cell_transmiss_prob_ij, kernel_lookup, distances_lookup, binomial_RNG, P_CS, params_sim)

    # if t > 0:
    indices = np.where((holding_status > -1) & (holding_status < day_culled))[0]
    # print(f"Indices: {indices}")
    holding_status[indices] += 1
    # Check this for negative cell status
    cell_status = [cell_status[0]] + [np.zeros(len(cell_status[0]))] + cell_status[1:-1]

    vax_indices = np.where(((holding_vax_status > -1) & (holding_vax_status < vaccine_effectiveness_delay)))[0]
    if len(vax_indices) > 0:
        holding_vax_status[vax_indices] += 1

    # # Check that holding_vax_status is 0
    # if np.any(holding_vax_status != 0):
    #     print("Error: holding_vax_status is non-zero.")

    # print(f"Conditional infected count: {conditional_infected_count}, Local infected count: {local_infected_count}")
        
    return events, events_county, holding_county, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, iterate_flag

#------------------------------------------------------------------------------------------
# Conditional subsampling algorithm
#------------------------------------------------------------------------------------------
def conditional_subsample(conditional_infected_count, rng, t, infectious_cell, susceptible_cell, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, events, events_county, holding_county, params_epi, params_intervention, params_iterate, cell_to_cell_transmiss_prob_ij, kernel_lookup, distances_lookup, binomial_RNG, P_CS, params_sim):
    # Check status of holdings in susceptible cell
    cell_j_susceptibles = np.where(((holding_status == -1) & (holding_vax_status < params_epi['vaccine_effectiveness_delay'])) & (holding_cell[0] == susceptible_cell))[0]
    cell_i_infectious = np.where((holding_status >= params_epi['day_infectious']) & ((holding_status < params_epi['day_culled'])) & (holding_cell[0] == infectious_cell))[0]
    # print(f"Cell infectious: {cell_i_infectious}")

    num_infectious = len(cell_i_infectious)  # Get number of infected in cell with infectious nodes

    cumulative_p_over = 1.0 - ((1.0 - cell_to_cell_transmiss_prob_ij) ** num_infectious)

    rounded_index = np.searchsorted(P_CS, cumulative_p_over, side='right')
    # print(f"Rounded index: {rounded_index}, Cumulative p over: {cumulative_p_over}, P_CS: {P_CS[rounded_index]}")
    # Check boundary conditions
    if rounded_index == 0:
        closest_index = 0
    elif rounded_index >= len(P_CS):
        closest_index = len(P_CS) - 1
    else:
        closest_index = rounded_index

    # if params_sim['binomial_flag'] == 'closest':
    #     rounded_index = np.searchsorted(P_CS, cumulative_p_over, side='right')
    #     # if rounded_index >= len(P_CS):
    #     #     rounded_index = len(P_CS) - 1

    #     # Check boundary conditions and find the closest index
    #     if rounded_index == 0:
    #         closest_index = 0
    #     elif rounded_index >= len(P_CS):
    #         closest_index = len(P_CS) - 1
    #     else:
    #         # Compare the two nearest neighbors
    #         left_diff = abs(P_CS[rounded_index - 1] - cumulative_p_over)
    #         right_diff = abs(P_CS[rounded_index] - cumulative_p_over)
            
    #         closest_index = rounded_index - 1 if left_diff <= right_diff else rounded_index
    # elif params_sim['binomial_flag'] == 'right':
    #     rounded_index = np.searchsorted(P_CS, cumulative_p_over, side='right')
    #     # print(f"Rounded index: {rounded_index}, Cumulative p over: {cumulative_p_over}, P_CS: {P_CS[rounded_index]}")
    #     # Check boundary conditions
    #     if rounded_index == 0:
    #         closest_index = 0
    #     elif rounded_index >= len(P_CS):
    #         closest_index = len(P_CS) - 1
    #     else:
    #         closest_index = rounded_index
    # elif params_sim['binomial_flag'] == 'left':
    #     rounded_index = np.searchsorted(P_CS, cumulative_p_over, side='left')
    #     # Check boundary conditions
    #     if rounded_index == 0:
    #         closest_index = 0
    #     elif rounded_index >= len(P_CS):
    #         closest_index = len(P_CS) - 1
    #     else:
    #         closest_index = rounded_index - 1
    # elif params_sim['binomial_flag'] == 'exact':
    #     rounded_index = np.searchsorted(P_CS, cumulative_p_over, side='right')
    #     # Check boundary conditions
    #     if rounded_index == 0:
    #         closest_index = 0
    #     elif rounded_index >= len(P_CS):
    #         closest_index = len(P_CS) - 1
    #     else:
    #         closest_index = rounded_index
    # else:
    #     raise ValueError("Invalid binomial_flag value. Must be 'closest', 'right', 'left', or 'exact'.")

    # Draw the number of nodes that would get infected using the cumulative p_over.
    num_susceptible_in_cell = len(cell_j_susceptibles)
    
    # if num_susceptible_in_cell > 0 and rounded_p_over_idx >= 0:
    if num_susceptible_in_cell > 0:
        n_over = binomial_RNG[num_susceptible_in_cell, closest_index]()
        # if params_sim['binomial_flag'] == 'exact':
        #     # Use the exact cumulative probability
        #     n_over = rng.binomial(num_susceptible_in_cell, cumulative_p_over)
        # else:
        #     n_over = binomial_RNG[num_susceptible_in_cell, closest_index]()
            # print(f"Binomial RNG: {binomial_RNG[num_susceptible_in_cell, closest_index]}, Cumulative p over: {cumulative_p_over}, Rounded index: {rounded_index}, Closest index: {closest_index}, P_CS: {P_CS[closest_index]}, Num susceptible in cell: {num_susceptible_in_cell}")

        # Generate binomial random variate
        # n_over, p_rounded = generate_binomial_random_variate(num_susceptible_in_cell, cumulative_p_over, binomial_lookup_ps)
        # print(f"Generated binomial random variate: {n_over}, Rounded p: {p_rounded}, Number susceptible: {num_susceptible_in_cell}")
        # If nodes would have been infected using p_over, check if infected using actual transmission rate
        if n_over > 0:

            # Sample n_over nodes from susceptible nodes
            sample_suscep_nodes_r = rng.choice(cell_j_susceptibles, n_over, replace=False)
            # print(f"Sampled susceptible nodes: {sample_suscep_nodes_r}, Number sampled: {len(cell_j_susceptibles)}, Number to sample: {n_over}")
            n_inf = 0

            # Test if infected by collection of infectious node in InfCell
            for holding_j_susceptible in sample_suscep_nodes_r:
                # Skip if holding_status[holding_j_susceptible] > -1
                if holding_status[holding_j_susceptible] > -1:
                    continue
                # Calculate cumulative_p (Cumulative probability of at least one node in inf_cell infecting this sus_node.)
                # cumulative_p = compute_pairwise_cumulative_infection_prob(holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, kernel_lookup, params_sim['tstep'], cell_i_infectious, holding_j_susceptible)
                cumulative_p = compute_pairwise_cumulative_infection_prob(events, events_county, holding_county, kernel_lookup, distances_lookup, holding_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, susceptible_cell, infectious_cell, cell_i_infectious, holding_j_susceptible, params_sim)

                r = rng.random()
                # print("Random number: ", r)
                # print(f"Random number: {r}, Cumulative probability: {cumulative_p}, Cumulative probability over: {cumulative_p_over}, Cumulative p over cumulative p over: {cumulative_p / cumulative_p_over}")

                # Event-driven stochastic event. If infection occurs, update node status
                new_cumulative_p_over = P_CS[closest_index]
                if r < (cumulative_p / new_cumulative_p_over):
                # if r < (cumulative_p / cumulative_p_over):
                    if holding_status[holding_j_susceptible] > -1:  # Already infected
                        print(f"Warning: Holding {holding_j_susceptible} is being infected again in conditional_subsample.")
                    n_inf += 1
                    conditional_infected_count += 1
                    # print(f"Node {holding_j_susceptible} infected by node {cell_i_infectious}")
                    holding_status[holding_j_susceptible] += 1
                    cell_status[0][susceptible_cell] -= 1
                    cell_status[1][susceptible_cell] += 1
                    events[1][holding_j_susceptible] = t
                    # Check if holding is vaccinated
                    if holding_vax_status[holding_j_susceptible] > -1:
                        holding_vax_status[holding_j_susceptible] = 6
                        cell_vax_status[0][susceptible_cell] -= 1
                        cell_vax_status[6][susceptible_cell] += 1
                        events[params_intervention['vaccination_event_ID'] + 6][holding_j_susceptible] = t
                        # Check columns vaccination event ID to vaccination event ID + 5 and set anything greater than or equal to t to NA
                        cols = range(params_intervention['vaccination_event_ID'], params_intervention['vaccination_event_ID'] + 6)
                        for col in cols:
                            if events[col][holding_j_susceptible] >= t:
                                events[col][holding_j_susceptible] = np.nan
            # if n_inf > 0:
            # print("Conditional infected count: ", n_inf, "out of: ", n_over)

    # print("Difference between expected and conditional infected count: ", conditional_infected_count - n_over*cumulative_p_over)
    # print("")
    # print(f"Conditional infected count: {conditional_infected_count}")
    return conditional_infected_count

def local_pairwise_within_cell(local_infected_count, rng, t, infectious_cell, susceptible_cell, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, events, events_county, holding_county, params_epi, params_intervention, params_iterate, kernel_lookup, distances_lookup, params_sim):
    # # Check status of holdings in susceptible cell
    # cell_i_holdings = np.where(holding_cell[0] == infectious_cell)[0]
    # Determine susceptible holding ids in cell
    cell_i_susceptibles = np.where(((holding_status == -1) & (holding_vax_status < params_epi['vaccine_effectiveness_delay'])) & (holding_cell[0] == infectious_cell))[0]
    # Determine infectious holdings in cell
    cell_i_infectious = np.where((holding_status >= params_epi['day_infectious']) & ((holding_status < params_epi['day_culled'])) & (holding_cell[0] == infectious_cell))[0]

    # print(f"Cell {infectious_cell} has {len(cell_i_holdings)} holdings, {len(cell_i_susceptibles)} susceptible holdings, and {len(cell_i_infectious)} infectious holdings.")

    for holding_i_infectious in cell_i_infectious:
        for holding_i_susceptible in cell_i_susceptibles:
            # Skip if holding_status[holding_i_susceptible] > -1
            if holding_status[holding_i_susceptible] > -1:
                continue
            # print(f"Susceptible holding: {holding_i_susceptible}")
            # Get probability susceptible unit infected by any infectious node in cell
            # cumul_prob_against_suscept_holding = compute_pairwise_cumulative_infection_prob(holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, kernel_lookup, params_sim['tstep'], cell_i_infectious, holding_i_susceptible)
            cumul_prob_against_suscept_holding = compute_local_infection_prob(events, events_county, holding_county, kernel_lookup, distances_lookup, holding_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, susceptible_cell, infectious_cell, holding_i_infectious, holding_i_susceptible, params_sim)

            # print(f"cumul_prob_against_suscept_holding: {cumul_prob_against_suscept_holding}")

            # Draw random number. If less than cumul_prob_against_suscept_holding, infection event was successful
            # r = np.random.uniform()
            r = rng.random()
            # r = rng.random_sample()
            # print("Random number: ", r)

            if r < cumul_prob_against_suscept_holding:
                if holding_status[holding_i_susceptible] > -1:  # Already infected
                    print(f"Warning: Holding {holding_i_susceptible} is being infected again in local_pairwise_within_cell.")
                local_infected_count += 1
                # print(f"Node {holding_i_susceptible} infected by node {cell_i_infectious}")
                # print(f"Random number: {r}, Cumulative probability: {cumul_prob_against_suscept_holding}")
                holding_status[holding_i_susceptible] += 1
                cell_status[0][susceptible_cell] -= 1
                cell_status[1][susceptible_cell] += 1
                events[1][holding_i_susceptible] = t
                # Check if holding is vaccinated
                if holding_vax_status[holding_i_susceptible] > -1:
                    holding_vax_status[holding_i_susceptible] = 6
                    cell_vax_status[0][susceptible_cell] -= 1
                    cell_vax_status[6][susceptible_cell] += 1
                    events[params_intervention['vaccination_event_ID']+6][holding_i_susceptible] = t
                    # Check columns vaccination event ID to vaccination event ID + 5 and set anything greater than or equal to t to NA
                    cols = range(params_intervention['vaccination_event_ID'], params_intervention['vaccination_event_ID'] + 6)
                    for col in cols:
                        if events[col][holding_i_susceptible] >= t:
                            events[col][holding_i_susceptible] = np.nan

    return local_infected_count

def compute_local_infection_prob(events, events_county, holding_county, kernel_lookup, distances_lookup, holding_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, susceptible_cell, infectious_cell, holding_i_infectious, holding_i_susceptible, params_sim):
    
    d = distances_lookup[holding_i_infectious, holding_i_susceptible]

    # Calculate rate of infection
    dist_idx = int(d)
    FOI_rate = (holding_transmissibility[holding_i_infectious] *
                holding_susceptibility[holding_i_susceptible] *
                kernel_lookup[dist_idx])

    # Compute probability of infection (using 1-exp function)
    prob = -np.expm1(-FOI_rate * params_sim['tstep'])

    return prob

# def compute_pairwise_cumulative_infection_prob(holding_easting, holding_northing, holding_transmissibility, holding_susceptibility,
#                                                kernel_lookup, 
#                                                delta_t, 
#                                                cell_infectious, 
#                                                holding_susceptible):
#     """
#     Compute cumulative pairwise infection probability against a single susceptible unit.

#     Parameters:
#     - holding_vectors_and_arrays_params: Object containing holding attributes (locations, transmissibility, susceptibility).
#     - coord_type: Integer (1 for Cartesian in metres, 2 for Cartesian in km, 3 for Lat/Long).
#     - kernel_lookup_vec: Array of infection risk values against distance.
#     - delta_t: Float, timestep increment.
#     - infectious_holding_in_transmission_cell_IDs: List of IDs of infectious units in the current cell.
#     - holding_susceptible: ID of the susceptible unit.

#     Returns:
#     - cumulative_p: Float, probability that the susceptible unit is infected.
#     """
#     # Get location of the susceptible node
#     suscep_holding_X_loc = holding_easting[holding_susceptible]
#     suscep_holding_Y_loc = holding_northing[holding_susceptible]

#     # Initialize no infection probability variable
#     no_infection_prob = 1.0

#     # Iterate over each infectious node
#     for selected_infectious_holding_ID in cell_infectious:
#         # Get location of the infectious unit
#         InfNode_xLoc = holding_easting[selected_infectious_holding_ID]
#         InfNode_yLoc = holding_northing[selected_infectious_holding_ID]
    
#         d = landscape.holding_euclidean_distance(InfNode_xLoc, InfNode_yLoc, suscep_holding_X_loc, suscep_holding_Y_loc)

#         # Calculate rate of infection
#         dist_idx = int(d)
#         FOI_rate = (holding_transmissibility[selected_infectious_holding_ID] *
#                     holding_susceptibility[holding_susceptible] *
#                     kernel_lookup[dist_idx])

#         # Compute probability of infection (using 1-exp function)
#         local_prob_val = -np.expm1(-FOI_rate * delta_t)

#         # Revise no infection probability
#         no_infection_prob *= (1.0 - local_prob_val)

#     # Update cumulative probability of at least one node infecting the susceptible holding
#     cumulative_p = 1.0 - no_infection_prob

#     return cumulative_p

def compute_pairwise_cumulative_infection_prob(events, events_county, holding_county, kernel_lookup, distances_lookup, holding_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, susceptible_cell, infectious_cell, cell_infectious, holding_susceptible, params_sim):
    # Initialise cumulative probability
    no_infection = 1.0
    tstep = params_sim['tstep']
    holding_susceptibility_value = holding_susceptibility[holding_susceptible]

    # Iterate over infectious holdings
    for holding_infectious in cell_infectious:
        # print(f"Checking holding {holding_infectious} for infection of holding {holding_susceptible}")
        # Get distance between infectious and susceptible holdings from lookup table
        if infectious_cell == susceptible_cell:
            dist = round(distances_lookup[holding_infectious, holding_susceptible])
        else:
            # Calculate distance between holdings
            dist = round(landscape.holding_euclidean_distance(holding_easting[holding_infectious], holding_northing[holding_infectious], holding_easting[holding_susceptible], holding_northing[holding_susceptible]))
            # print(f"Distance between holding {holding_infectious} and holding {holding_susceptible}: {dist}")

        if dist < 50000:
            # Get transmission probability
            # print(f"Distance: {dist}, Transmissibility: {holding_transmissibility[holding_infectious]}, Susceptibility: {holding_susceptibility_value}, Kernel: {kernel_lookup[dist]}")
            FOI = holding_transmissibility[holding_infectious] * holding_susceptibility_value * kernel_lookup[dist]
            # print(f"transmissibility: {holding_transmissibility[holding_infectious]}, susceptibility: {holding_susceptibility_value}, kernel: {kernel_lookup[dist]}")
            local_p = -np.expm1(-FOI * tstep)
            no_infection *= (1 - local_p)
        else:
            FOI = 0
            local_p = 0
            no_infection *= (1 - local_p)
            # no_infection == 1

    # Return cumulative probability
    cumulative_p = 1 - no_infection
    return cumulative_p

def update_events(events, events_county, holding_county, t, params_intervention, params_epi):
    for i in range(1, (params_epi['day_culled'])):
        previous_events = np.where(events[i] == (t - 1))[0]
        events[i+1][previous_events] = t

    if params_intervention['vaccinate_flag'] == 1:
        for i in range(params_intervention['vaccination_event_ID'], params_intervention['vaccination_event_ID']+params_epi['vaccine_effectiveness_delay']):
            previous_events = np.where((events[i] == (t - 1)))[0]
            # previous_events = np.where((events[i] == (t - 1)) & ((events[params_intervention['vaccination_event_ID']+params_epi['vaccine_effectiveness_delay'] + 1] < (t - 2)) | np.isnan(events[params_intervention['vaccination_event_ID']+params_epi['vaccine_effectiveness_delay'] + 1])))[0]
            events[i+1][previous_events] = t

        # # Check columns 14 to 18 and shift values if column 19 is NaN
        # for col in range(params_intervention['vaccination_event_ID'], params_intervention['vaccination_event_ID']+params_epi['vaccine_effectiveness_delay']-1):
        #     shift_indices = np.where((events[col] == (t - 1)) & (np.isnan(events[params_intervention['vaccination_event_ID']+params_epi['vaccine_effectiveness_delay']])))[0]
        #     events[col+1][shift_indices] = t

    return