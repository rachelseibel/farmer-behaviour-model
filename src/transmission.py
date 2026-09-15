#------------------------------------------------------------------------------------------
# Load the required libraries
#------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
# from scipy.spatial import distance_matrix
# from scipy.sparse import lil_matrix
import pickle

import landscape

def calculate_transmission_probabilities(cells, holdings, params_epi, params_sim, params_land, kernel_lookup):
    # Between-grid transmission probabilities
    if params_land['calculate_transmission_probabilities_flag'] == 1:
        print('Calculating transmission probabilities...')
        # Determine the number of cattle in each cell
        num_cattle = holdings['cattle']
        # Get holding-specific susceptibility and transmissibility
        holdings['susceptibility'] = (params_epi['susceptibility']) * (num_cattle**(params_epi['exp_susceptibility']))
        holdings['transmissibility'] = (params_epi['transmissibility']) * (num_cattle**(params_epi['exp_transmissibility']))

        # Get maximum susceptibility and transmissibility by cell ID
        max_values = calculate_max_susceptibility_transmissibility(holdings)
        # Map max_values to cells
        cells = cells.merge(max_values, how='left', left_on='cell_id', right_index=True)
        # Calculate distance between every cell
        num_cells = len(cells)

        # Initialize cell_to_cell_transmiss_prob with zeros
        cell_to_cell_transmiss_prob = pd.DataFrame(np.zeros((num_cells, num_cells)))
        for i in range(num_cells):
            for j in range(num_cells):
                # print('Calculating transmission probabilities for cells: ', i, j)
                if i != j:
                    # Calculate the distance between the two cells
                    dist = landscape.shortest_distance_between_squares(cells.loc[i, 'min_x'], cells.loc[i, 'min_y'], cells.loc[i, 'max_x'], cells.loc[i, 'max_y'], cells.loc[j, 'min_x'], cells.loc[j, 'min_y'], cells.loc[j, 'max_x'], cells.loc[j, 'max_y'])
                    # Distance as integer
                    dist = int(dist)
                    # Calculate the transmission probability
                    # Skip if no key for transmission probability
                    if (dist > len(kernel_lookup)) or (cells.loc[i, 'num_nodes'] == 0) or (cells.loc[j, 'num_nodes'] == 0) or (max_values['max_transmissibility'][i] == 0) or (max_values['max_susceptibility'][j] == 0) or (kernel_lookup[dist] == 0):
                        cell_to_cell_transmiss_prob.loc[i, j] = 0
                        cell_to_cell_transmiss_prob.loc[j, i] = 0
                    else:
                        cell_to_cell_transmissibilityij = max_values['max_transmissibility'][i] * max_values['max_susceptibility'][j] * kernel_lookup[dist]
                        cell_to_cell_transmiss_prob.loc[i, j] = -np.expm1(-cell_to_cell_transmissibilityij*params_sim['tstep'])
                        cell_to_cell_transmissibilityji = max_values['max_transmissibility'][j] * max_values['max_susceptibility'][i] * kernel_lookup[dist]
                        cell_to_cell_transmiss_prob.loc[j, i] = -np.expm1(-cell_to_cell_transmissibilityji*params_sim['tstep'])
                    # print(f"Cell {i} to Cell {j}: {cell_to_cell_transmiss_prob.loc[i, j]}")
                    # print(f"max_transmissibility: {max_values['max_transmissibility'][i]}")
                    # print(f"max_susceptibility: {max_values['max_susceptibility'][j]}")
                    # print(f"Kernel lookup: {kernel_lookup[dist]}")
                    # print(f"Distance: {dist}")
                else:
                    cell_to_cell_transmiss_prob.loc[i, j] = 1
                    cell_to_cell_transmiss_prob.loc[j, i] = 1

        # Save to csv file
        cell_to_cell_transmiss_prob = cell_to_cell_transmiss_prob.values
        # Set all diagonal elements to 1
        np.fill_diagonal(cell_to_cell_transmiss_prob, 1)
        pd.DataFrame(cell_to_cell_transmiss_prob).to_csv(params_land['transmission_file_path'])

    else:
        if os.path.isfile(params_land['transmission_file_path']):
            print('Loading transmission probabilities from file...')
            # Load from file
            cell_to_cell_transmiss_prob = pd.read_csv(params_land['transmission_file_path'], index_col=0)
            # Convert to numpy array
            cell_to_cell_transmiss_prob = cell_to_cell_transmiss_prob.values
            # Determine the number of cattle in each cell
            num_cattle = holdings['cattle']
            # Get holding-specific susceptibility and transmissibility
            holdings['susceptibility'] = params_epi['susceptibility'] * (num_cattle**(params_epi['exp_susceptibility']))
            holdings['transmissibility'] = params_epi['transmissibility'] * (num_cattle**(params_epi['exp_transmissibility']))
            # Get maximum susceptibility and transmissibility by cell ID
            max_values = calculate_max_susceptibility_transmissibility(holdings)
            # Map max_values to cells
            cells = cells.merge(max_values, how='left', left_on='cell_id', right_index=True)
            # Calculate distance between every cell
            num_cells = len(cells)
        else:
            print('Transmission probabilities not found. Exiting...')
            exit()

    # # Log transmission probabilities
    # print("Transmission Probabilities:")
    # for i in range(len(cell_to_cell_transmiss_prob)):
    #     for j in range(len(cell_to_cell_transmiss_prob)):
    #         print(f"Cell {i} to Cell {j}: {cell_to_cell_transmiss_prob[i, j]}")

    return cell_to_cell_transmiss_prob

def calculate_max_susceptibility_transmissibility(holdings):
    max_values = holdings.groupby('cell_id')[['susceptibility', 'transmissibility']].max()
    max_values.rename(columns={'susceptibility': 'max_susceptibility', 'transmissibility': 'max_transmissibility'}, inplace=True)
    return max_values

def calculate_holding_pair_distances_by_cell(holdings, params_land):
    # Create a sparse matrix to store distances
    holding_distances_dictionary = {}
    # print size of holding matrix
    # print('Size of holding matrix: ', holding_distances_matrix.shape)

    cell_ids_list = holdings['cell_id'].unique()
    # Sort by cell_id
    cell_ids_list.sort()

    # Iterate over cells
    for cell_holding in cell_ids_list:
        print('Calculating distances for cell: ', cell_holding)
        # Get holdings in cell_id
        cell_holdings = holdings[holdings['cell_id'] == cell_holding]
        # # Get coordinates
        # coordinates = cell_holdings[['easting', 'northing']].values
        # # Calculate distance matrix
        # distances = distance_matrix(coordinates, coordinates)

        # Update global matrix
        for i in range(len(cell_holdings)):
            for j in range(len(cell_holdings)):
                # Get holding id
                holding_id_i = int(round(cell_holdings.iloc[i]['holding_id']))
                holding_id_j = int(round(cell_holdings.iloc[j]['holding_id']))
                # Get coordinates
                easting_i = cell_holdings.iloc[i]['easting']
                northing_i = cell_holdings.iloc[i]['northing']
                easting_j = cell_holdings.iloc[j]['easting']
                northing_j = cell_holdings.iloc[j]['northing']
                # Calculate distance
                distance = landscape.holding_euclidean_distance(easting_i, northing_i, easting_j, northing_j)
                # Update dictionary
                holding_distances_dictionary[(holding_id_i, holding_id_j)] = distance
                holding_distances_dictionary[(holding_id_j, holding_id_i)] = distance

    # Save
    with open(params_land['distance_file_path'], 'wb') as f:
        pickle.dump(holding_distances_dictionary, f)
    return holding_distances_dictionary

# def calculate_holding_to_holding_distances(holdings):
#     # Convert holding coordinates to a 2D numpy array
#     holding_coordinates = holdings[['easting', 'northing']].values
#     # Calculate distances between holdings using scipy's distance_matrix function
#     holding_to_holding_distances = distance_matrix(holding_coordinates, holding_coordinates)
#     # Save to compressed npz file
#     np.savez_compressed('src/inputs/holding_to_holding_distances.npz', holding_to_holding_distances)
#     return holding_to_holding_distances

# def calculate_holding_pair_distances_by_cell(holdings, params_land):
#     # Create a sparse matrix to store distances
#     holding_distances_matrix = lil_matrix((len(holdings['holding_id']), len(holdings['holding_id'])))
#     # print size of holding matrix
#     print('Size of holding matrix: ', holding_distances_matrix.shape)

#     cell_ids_list = holdings['cell_id'].unique()
#     # Sort by cell_id
#     cell_ids_list.sort()

#     # Iterate over cells
#     for cell_holding in cell_ids_list:
#         print('Calculating distances for cell: ', cell_holding)
#         # Get holdings in cell_id
#         cell_holdings = holdings[holdings['cell_id'] == cell_holding]
#         # Get coordinates
#         coordinates = cell_holdings[['easting', 'northing']].values
#         # Calculate distance matrix
#         distances = distance_matrix(coordinates, coordinates)

#         # Update global matrix
#         for i in range(len(cell_holdings)):
#             for j in range(len(cell_holdings)):
#                 holding_distances_matrix[int(round(cell_holdings.iloc[i]['holding_id'])), int(round(cell_holdings.iloc[j]['holding_id']))] = distances[i, j]
#                 holding_distances_matrix[int(round(cell_holdings.iloc[j]['holding_id'])), int(round(cell_holdings.iloc[i]['holding_id']))] = distances[j, i]

#     # Save to pkl
#     with open(params_land['distance_file_path'], 'wb') as f:
#         pickle.dump(holding_distances_matrix.tocsr(), f)
#     return holding_distances_matrix.tocsr()