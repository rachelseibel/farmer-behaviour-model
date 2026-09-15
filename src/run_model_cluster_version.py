#------------------------------------------------------------------------------------------
# Load the required libraries
#------------------------------------------------------------------------------------------
import numpy as np
import random as rnd
import pandas as pd
from scipy.integrate import quad
import geopandas as gpd
import os
from scipy.stats import binom
from scipy.sparse import csr_matrix
from functools import partial

#------------------------------------------------------------------------------------------
# Import supporting functions from other files
#------------------------------------------------------------------------------------------
import parameters
import simulation
import transmission
import landscape

# run_model.py contains the following functions:
# iterate()
# kernel()
# main()

#------------------------------------------------------------------------------------------
# Kernel function
#------------------------------------------------------------------------------------------
def dispersal_kernel(d):
    # List of required parameter values
    k2 = 2000.0
    k3 = 2.0

    # Set limit on transmission distance to 50km (50,000m)
    max_transmission_distance = 50000

    # Set kernel value based on distance d
    # If above max transmission distance, set to zero
    if d > max_transmission_distance:
        kernel_val = 0.0
    else:
        # # Get normalisation constant, based on max_transmission_distance, k2 & k3 values
        # non_normalised_integral, err = quad(lambda d: ((2*np.pi*d) / (1 + ((d/k2)**k3))), 0, max_transmission_distance)
        # print(non_normalised_integral)
        # # Set normalisation constant
        # k1 = 1/non_normalised_integral
        # print(k1)
        k1 = 1.2*10**(-8)
        # Express transmission kernel formulation
        kernel_val = k1 / (1 + ((d/k2)**k3))

    # Return kernel value for distance d from function
    return kernel_val

def construct_dispersal_kernel(max_dist):
    max_dist = int(max_dist + 1)
    # Initialise lookup array
    kernel_lookup_vec = [0] * max_dist
    # Iterate over desired distances. Assign kernel value to array
    for DistItr in range(max_dist):
        kernel_lookup_vec[DistItr] = dispersal_kernel(float(DistItr))
    return kernel_lookup_vec

def construct_binomial_RNG(rng, n_holdings_per_cell):
    """
    Construct a collection of Binomial RNGs (random number generators).

    Inputs:
    - rng: Random number generator object.
    - n_holdings_per_cell: List or 1D numpy array with the number of holdings per grid cell.

    Outputs:
    - binomial_RNG_array: 2D numpy array of Binomial RNGs. Row per susceptible number, column per probability threshold.
    """

    # Pre-set probabilities.
    P_CS = [5.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Get maximum possible number of susceptibles per grid
    max_nodes_per_cell = int(max(n_holdings_per_cell))

    # Get quantity of preset probabilities to initialize RNG with
    num_thresholds = len(P_CS)

    # Construct the Binomial RNG
    # Row per susceptible number, column per probability threshold
    binomial_RNG_array = np.empty((max_nodes_per_cell + 1, num_thresholds), dtype=object)  # Initialize array of Binomial RNGs
    for i in range(max_nodes_per_cell + 1):
        for j in range(num_thresholds):
            # Create a callable generator using functools.partial
            binomial_RNG_array[i, j] = partial(rng.binomial, n=i, p=P_CS[j])

    return P_CS, binomial_RNG_array

# def construct_binomial_RNG_small(rng, n_holdings_per_cell):
#     """
#     Construct a collection of Binomial RNGs (random number generators).

#     Inputs:
#     - rng: Random number generator object.
#     - n_holdings_per_cell: List or 1D numpy array with the number of holdings per grid cell.
#     - P_CS: List or 1D numpy array of preset probabilities to initialize RNG with.

#     Outputs:
#     - binomial_RNG_array: 2D numpy array of Binomial RNGs. Row per susceptible number, column per probability threshold.
#     """

#     # Pre-set probabilities.
#     P_CS = [5.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

#     # Get maximum possible number of susceptibles per grid
#     max_nodes_per_cell = int(max(n_holdings_per_cell))

#     # Get quantity of preset probabilities to initialize RNG with
#     num_thresholds = len(P_CS)

#     # Construct the Binomial RNG
#     # Row per susceptible number, column per probability threshold
#     binomial_RNG_array = np.empty((max_nodes_per_cell + 1, num_thresholds), dtype=object)  # Initialize array of Binomial RNGs
#     for i in range(max_nodes_per_cell + 1):
#         for j in range(num_thresholds):
#             binomial_RNG_array[i, j] = rng.binomial(i, P_CS[j])  # Skip the first entry of P_CS (index 0)

#     return P_CS, binomial_RNG_array

# def construct_binomial_RNG(rng, n_holdings_per_cell, num_per_order):
#     """
#     Construct a collection of Binomial RNGs (random number generators).

#     Inputs:
#     - n_holdings_per_cell: List or 1D numpy array with the number of holdings per grid cell.

#     Outputs:
#     - binomial_RNG_array: 2D numpy array of Binomial RNGs. Row per susceptible number, column per probability threshold.
#     """
#     # Generate values spanning several orders of magnitude
#     p_min = 1e-9
#     p_max = 1.0
#     # num_per_order = 10  # Number of points per order of magnitude (e.g., 1e-9, 5e-9)

#     # Calculate how many orders of magnitude we want to cover
#     orders_of_magnitude = int(np.log10(p_max / p_min))

#     # Generate the points using a logarithmic scale
#     P_CS = []
#     for i in range(orders_of_magnitude + 1):
#         base_value = 10 ** (-i)
#         for j in range(num_per_order):
#             value = base_value * (j + 1) / num_per_order
#             if value <= p_max:
#                 # Round to avoid floating-point issues
#                 value = round(value, 10)
#                 P_CS.append(value)
#             else:
#                 break

#     # Add the maximum value (1.0) if it's not already in the list
#     if 1.0 not in P_CS:
#         P_CS.append(1.0)

#     # Sort the values in descending order (for easier indexing later)
#     P_CS = sorted(P_CS, reverse=False)
#     # Remove duplicates from the list
#     P_CS = list(dict.fromkeys(P_CS))
#     # Convert to numpy array
#     P_CS = np.array(P_CS)

#     # Get maximum possible number of susceptibles per grid
#     max_nodes_per_cell = int(max(n_holdings_per_cell))

#     # Get quantity of preset probabilities to initialize RNG with
#     num_thresholds = len(P_CS)

#     # Construct the Binomial RNG
#     # Row per susceptible number, column per probability threshold
#     binomial_RNG_array = np.empty((max_nodes_per_cell + 1, num_thresholds), dtype=object)  # Initialize array of Binomial RNGs
#     for i in range(max_nodes_per_cell + 1):
#         for j in range(num_thresholds):
#             binomial_RNG_array[i, j] = rng.binomial(i, P_CS[j])  # Skip the first entry of P_CS (index 0)

#     return P_CS, binomial_RNG_array

# def construct_binomial_RNG(n_holdings_per_cell, P_CS):
#     """
#     Construct a collection of Binomial RNGs (random number generators).

#     Inputs:
#     - n_holdings_per_cell: List or 1D numpy array with the number of holdings per grid cell.
#     - P_CS: List or 1D numpy array of preset probabilities to initialize RNG with.

#     Outputs:
#     - binomial_RNG_array: 2D numpy array of Binomial RNGs. Row per susceptible number, column per probability threshold.
#     """
#     # Get maximum possible number of susceptibles per grid
#     MaxPremPerGrid = max(n_holdings_per_cell)

#     # Get quantity of preset probabilities to initialize RNG with
#     PresetProbThresholds = len(P_CS) - 1  # Ignore first zero entry

#     # Construct the Binomial RNG
#     # Row per susceptible number, column per probability threshold
#     binomial_RNG_array = np.empty((MaxPremPerGrid, PresetProbThresholds), dtype=object)  # Initialize array of Binomial RNGs
#     for ii in range(PresetProbThresholds):
#         for jj in range(1, MaxPremPerGrid + 1):
#             binomial_RNG_array[jj - 1, ii] = binom(jj, P_CS[ii + 1])  # Skip the first entry of P_CS (index 0)

#     return binomial_RNG_array

#------------------------------------------------------------------------------------------
# MAIN FUNCTION
#------------------------------------------------------------------------------------------
def main(batch, args):
    # Load parameters
    params_epi, params_land, params_behav, params_sim, params_intervention, params_out = parameters.parameter_loading(args)

    # rng = np.random.default_rng(params_sim['random_seed'])

    # Unpack parameters
    cells, holdings, bounding_box = landscape.create_landscape(params_land, params_epi)

    # # After assigning holdings to cells in create_landscape
    # print("Holding-to-Cell Mapping:")
    # for i, holding in enumerate(holdings.iterrows()):
    #     print(f"Holding {holdings.iloc[i]['holding_id']} is in Cell {holdings.iloc[i]['cell_id']}")

    # # Count holdings per cell
    # cell_holding_counts = holdings['cell_id'].value_counts()
    # print("Holdings per Cell:")
    # print(cell_holding_counts)

    # duplicates = holdings.duplicated(subset=['easting', 'northing'], keep=False)
    # if duplicates.any():
    #     print("Error: Duplicate holdings detected in multiple cells.")
    # print(holdings[duplicates])
    # for i, cell in cells.iterrows():
    #     for j, other_cell in cells.iterrows():
    #         if i != j:
    #             if not (cell['max_x'] <= other_cell['min_x'] or cell['min_x'] >= other_cell['max_x'] or
    #                     cell['max_y'] <= other_cell['min_y'] or cell['min_y'] >= other_cell['max_y']):
    #                 print(f"Error: Overlapping cells detected between cell {i} and cell {j}.")
    print('bounding box:', bounding_box)
    num_holdings = len(holdings)
    max_time = params_sim['max_time']
    event_file_path = params_out['event_file_path']
    num_reps = 1

    #------------------------------------------------------------------------------------------
    # Set up landscape
    #------------------------------------------------------------------------------------------

    # Construct the dispersal kernel using euclidean distance
    greatest_distance = landscape.bounding_box_diagonal(bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3])

    kernel_lookup = construct_dispersal_kernel(greatest_distance)

    # # Log kernel values
    # print("Kernel Lookup Values:")
    # for dist, value in enumerate(kernel_lookup):
    #     print(f"Distance: {dist}, Kernel Value: {value}")

    # Calculate distances between every pair of holdings
    if params_land['calculate_holding_distances_flag'] == 1:
        print('Calculating holding to holding distances by cell ID...')
        distances_lookup = transmission.calculate_holding_pair_distances_by_cell(holdings, params_land)
        # distances_lookup = csr_matrix(distances_lookup).toarray()
    else:
        # Check if the distances have already been calculated
        if os.path.isfile(params_land['distance_file_path']):
            print('Loading holding to holding distances by cell ID from file...')
            # Read in file with each line a dictionary key and value
            distances_lookup = pd.read_pickle(params_land['distance_file_path'])
            # CSR to array
            # distances_lookup = csr_matrix(distances_lookup).toarray()
        else:
            print('Calculating holding to holding distances by cell ID...')
            distances_lookup = transmission.calculate_holding_pair_distances_by_cell(holdings, params_land)

    #------------------------------------------------------------------------------------------
    # Calculate the transmission matrix
    #------------------------------------------------------------------------------------------
    # Calculate the maximum transmission rate for each pair of cells
    cell_to_cell_transmiss_prob = transmission.calculate_transmission_probabilities(cells, holdings, params_epi, params_sim, params_land, kernel_lookup)

    #------------------------------------------------------------------------------------------
    # Loop over the number of replicates
    #------------------------------------------------------------------------------------------
    sim_num = 0
    for i, landscape_iteration in enumerate(params_sim['landscape_iterations']):
        for j, seed_iteration in enumerate(params_sim['seed_iterations']):
            print('Running simulation with landscape method: ', args['landscape_method'], ' and county: ', args['county'], ' for landscape iteration: ', landscape_iteration, ' and seed iteration: ', seed_iteration)
            for rep in range(num_reps):

                # seed numpy

                # # get state from numpy
                # state = [int(s) for s in list(np.random.get_state()[1])]
                # state.append(624)
                # state = tuple(state)
                # state = (3, tuple(state), None)

                # np.random.seed(params_sim['random_seed'] + rep + (1000)*landscape_iteration + seed_iteration)

                # # set state for python 
                # # rnd.setstate(state)

                # # print(rnd.random())
                # print(np.random.rand())

                # Set the seed for the RNG
                rng = np.random.default_rng(params_sim['random_seed'] + rep + (1000)*landscape_iteration + seed_iteration)
                rng_binomial = np.random.default_rng(params_sim['random_seed'] + rep + (1000)*landscape_iteration + seed_iteration)
                # print('RNG: ', rng)

                num_holdings_per_cell = cells['num_nodes']

                # if params_sim['P_CS_size'] == 'small':
                #     # Construct binomial RNGs for small size
                #     P_CS, binomial_RNG = construct_binomial_RNG_small(rng, num_holdings_per_cell)
                # elif params_sim['P_CS_size'] == 'large':
                #     # Construct binomial RNGs for large size
                #     P_CS, binomial_RNG = construct_binomial_RNG(rng, num_holdings_per_cell, 10)
                # elif params_sim['P_CS_size'] == 'xlarge':
                #     # Construct binomial RNGs for medium size
                #     P_CS, binomial_RNG = construct_binomial_RNG(rng, num_holdings_per_cell, 20)
                # elif params_sim['P_CS_size'] == 'xxlarge':
                #     # Construct binomial RNGs for medium size
                #     P_CS, binomial_RNG = construct_binomial_RNG(rng, num_holdings_per_cell, 100)
                # else:
                #     raise ValueError("Invalid P_CS_size. Choose 'small', 'large' or 'xlarge'.")
                P_CS, binomial_RNG = construct_binomial_RNG(rng_binomial, num_holdings_per_cell)
                # np.random.seed(params_sim['random_seed'] + rep + (1000)*landscape_iteration + seed_iteration)
                #------------------------------------------------------------------------------------------
                # Set up output files for each run
                #------------------------------------------------------------------------------------------
                event_file = event_file_path+'_outputtype'+str(params_out['output_flag'])+'_batch'+str(batch)+'_rep'+str(rep)+'_landscapeMethod_'+str(args['landscape_method'])+'_county_'+str(args['county'])+'_landscapeIteration_'+str(landscape_iteration)+'_seedIteration_'+str(seed_iteration)
                event_file = event_file.replace(' ', '_')
        
                greatest_distance = landscape.bounding_box_diagonal(bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3])
                kernel_lookup = construct_dispersal_kernel(greatest_distance+3)

                #------------------------------------------------------------------------------------------
                # Run the simulation
                #------------------------------------------------------------------------------------------
                total_sims = (params_sim['landscape_iterations'][-1] + 1)*(params_sim['seed_iterations'][-1] + 1)*(num_reps)
                print('Running simulation number...', sim_num+1, ' of ', total_sims)
                events, events_county, cell_status, holding_behaviour = simulation.simulate_outbreak(rng, event_file, cells, holdings, cell_to_cell_transmiss_prob, kernel_lookup, distances_lookup, params_epi, params_land, params_behav, params_sim, params_intervention, params_out, binomial_RNG, P_CS, seed_iteration, landscape_iteration)

                # Number of events[1] elements that are not Nan
                num_infections = np.count_nonzero(~np.isnan(events[1]))
                print('Number of infections: ', num_infections)

                #------------------------------------------------------------------------------------------
                # Save outputs
                #------------------------------------------------------------------------------------------
                if params_out['output_flag'] == 0:
                    df = pd.DataFrame(events).T
                    df['easting'] = holdings['easting']
                    df['northing'] = holdings['northing']
                    df['cluster_id'] = holding_behaviour[0]
                    df['vaccination_distance'] = holding_behaviour[1]
                    df['landscape_method'] = args['landscape_method']
                    df['county'] = args['county']
                    df['landscape_iteration'] = landscape_iteration
                    df['seed_iteration'] = seed_iteration
                    df['rep'] = rep
                    df.to_csv(event_file+'.csv', index=False, header=True)

                #------------------------------------------------------------------------------------------
                # Save smaller outputs
                #------------------------------------------------------------------------------------------
                if params_out['output_flag'] == 1:
                    df = pd.DataFrame(columns=['landscape_method', 'county_seed', 'landscape_iteration', 'seed_iteration', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'C_cattle', 'V_pending_holdings', 'V_pending_cattle', 'V_success_holdings', 'V_success_cattle', 'V_failure_holdings', 'V_failure_cattle'])
                    # df = pd.DataFrame(columns=['landscape_method', 'county_seed', 'landscape_iteration', 'seed_iteration', 'county', 't', 'S', 'E', 'I', 'R', 'C', 'C_a', 'V_S', 'V_S_a', 'V_E', 'V_E_a', 'V_S_S', 'V_S_S_a', 'V_S_E', 'V_S_E_a'])
                    df['county'] = events_county['county']
                    df['t'] = events_county['t']
                    df['S'] = events_county['S']
                    df['E'] = events_county['E']
                    df['I'] = events_county['I']
                    df['R'] = events_county['R']
                    df['C'] = events_county['C']
                    df['C_cattle'] = events_county['C_cattle']
                    df['V_pending_holdings'] = events_county['V_pending_holdings']
                    df['V_pending_cattle'] = events_county['V_pending_cattle']
                    df['V_success_holdings'] = events_county['V_success_holdings']
                    df['V_success_cattle'] = events_county['V_success_cattle']
                    df['V_failure_holdings'] = events_county['V_failure_holdings']
                    df['V_failure_cattle'] = events_county['V_failure_cattle']
                    # df['V_S'] = events_county['V_S']
                    # df['V_S_a'] = events_county['V_S_a']
                    # df['V_E'] = events_county['V_E']
                    # df['V_E_a'] = events_county['V_E_a']
                    # df['V_S_S'] = events_county['V_S_S']
                    # df['V_S_S_a'] = events_county['V_S_S_a']
                    # df['V_S_E'] = events_county['V_S_E']
                    # df['V_S_E_a'] = events_county['V_S_E_a']
                    df['landscape_method'] = args['landscape_method']
                    df['county_seed'] = args['county']
                    df['landscape_iteration'] = landscape_iteration
                    df['seed_iteration'] = seed_iteration

                    df.to_csv(event_file+'.csv', index=False, header=True)

                    # Write seeded holdings and their cluster_id to file
                    seed_holdings = pd.DataFrame(columns=['landscape_method', 'county_seed', 'landscape_iteration', 'seed_iteration', 'easting', 'northing', 'cluster_id', 'vaccination_distance'])
                    # Get seeds where events[1] is 0
                    seed_indices = np.where(events[1] == 0)[0]
                    # Get the corresponding holding indices
                    seed_holdings['easting'] = holdings.iloc[seed_indices]['easting']
                    seed_holdings['northing'] = holdings.iloc[seed_indices]['northing']
                    seed_holdings['cluster_id'] = holding_behaviour[0][seed_indices]
                    seed_holdings['vaccination_distance'] = holding_behaviour[1][seed_indices]
                    seed_holdings['landscape_method'] = args['landscape_method']
                    seed_holdings['county_seed'] = args['county']
                    seed_holdings['landscape_iteration'] = landscape_iteration
                    seed_holdings['seed_iteration'] = seed_iteration

                    seed_file = event_file + '_seededHoldings.csv'
                    seed_holdings.to_csv(seed_file, index=False, header=True)

                sim_num += 1

            
    return


if __name__ == "__main__":
    input_line = sys.stdin.readline().strip()
    args = input_line.split(' , ')
    batch = int(args[0])
    landscape_method = args[1]
    county = args[2]
    # landscape_iteration = args[3]
    # seed_iteration = args[4]
    main(batch, args={'landscape_method': landscape_method, 'county': county})
    # main(batch, args={'landscape_method': landscape_method, 'county': county, 'landscape_iteration': landscape_iteration, 'seed_iteration': seed_iteration})

    # batch = int(sys.argv[0])
    # landscape_method = sys.argv[1]
    # landscape_iteration = sys.argv[2]
    # county = sys.argv[3]
    # seed_iteration = sys.argv[4]
    # main(batch, args={'landscape_method': landscape_method, 'landscape_iteration': landscape_iteration, 'county': county, 'seed_iteration': seed_iteration})