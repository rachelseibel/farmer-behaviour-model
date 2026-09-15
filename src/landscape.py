#------------------------------------------------------------------------------------------
# Load the required libraries
#------------------------------------------------------------------------------------------
import numpy as np
import math
import pandas as pd
import geopandas as gpd
from statistics import median
import os

import sys
sys.path.append('../../src')
import run_model

#------------------------------------------------------------------------------------------
# Supporting functions
#------------------------------------------------------------------------------------------

# def create_landscape(params_land, params_epi):
#     if params_land['generate_grid_flag'] == 1:
#         # Generate grid
#         holdings, cells = adaptive_grid(params_land, params_epi)
#         holdings = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle'])
#         cells = pd.DataFrame(cells, columns=['cell_id', 'min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
#         cells.to_csv(params_land['grid_file_path'], index=False)
#         holdings.to_csv(params_land['cattle_file_path'], index=False)
#     if params_land['generate_grid_flag'] == 2:
#         print('Using adaptive grid 2 to generate grid cells')
    
#         # Load landscape shapefile
#         gdf = pd.read_csv(params_land['landscape_shp_path'])
#         # Specify order of columns
#         gdf = gdf[['nation', 'region', 'county', 'easting', 'northing', 'cattle', 'iteration']]
#         # Get 0th iteration
#         gdf = gdf[gdf['iteration'] == 0]

#         # Calculate the bounding box of the data
#         min_x = min(gdf['easting'])
#         max_x = max(gdf['easting'])
#         min_y = min(gdf['northing'])
#         max_y = max(gdf['northing'])
#         # Extend bounds so it is a square
#         max_span = max(max_x - min_x, max_y - min_y)
#         min_x = min_x - (max_span - (max_x - min_x)) / 2
#         max_x = max_x + (max_span - (max_x - min_x)) / 2
#         min_y = min_y - (max_span - (max_y - min_y)) / 2
#         max_y = max_y + (max_span - (max_y - min_y)) / 2
#         num_holdings = len(gdf['northing'])
#         bounding_box_var = [min_x, min_y, max_x, max_y, num_holdings]

#         # Construct the dispersal kernel using euclidean distance
#         greatest_distance = holding_euclidean_distance(min_x, min_y, max_x, max_y)
#         kernel_lookup = run_model.construct_dispersal_kernel(greatest_distance)
#         if kernel_lookup is []:
#             kernel_lookup = run_model.construct_dispersal_kernel(greatest_distance+1)
        
#         # Get the holdings (easting, northing, cattle)
#         holdings = [(holding[0], holding[1], holding[2], holding[3], holding[4], holding[5], holding[6]) for holding in gdf.values]
#         # Get cattle for each holding in an array
#         cattle = [holding[5] for holding in holdings]
#         # susceptibility = (params_epi['susceptibility'] * np.array(cattle))**(params_epi['exp_susceptibility'])
#         # transmissibility = (params_epi['transmissibility'] * np.array(cattle))**(params_epi['exp_transmissibility'])
#         susceptibility = params_epi['susceptibility'] * (np.array(cattle)**params_epi['exp_susceptibility'])
#         transmissibility = params_epi['transmissibility'] * (np.array(cattle)**params_epi['exp_transmissibility'])

#         longest_side = get_longest_landscape_edge(min_x, min_y, max_x, max_y)

#         avg_node_per_grid_target = approx_node_per_cell_threshold_fn(susceptibility, transmissibility, kernel_lookup, 1, 50, longest_side)
#         print("avg_node_per_grid_target: ", avg_node_per_grid_target)

#         longest_landscape_edge = get_longest_landscape_edge(min_x, min_y, max_x, max_y)
#         holding_loc_X_vals = [holding[3] for holding in holdings]
#         holding_loc_Y_vals = [holding[4] for holding in holdings]
#         # Generate grid with adaptive_grid
#         holding_cell_IDs, grid_xMin, grid_xMax, grid_yMin, grid_yMax = adaptive_grid.adaptive_grid_construct_fn(avg_node_per_grid_target, bounding_box_var, longest_landscape_edge, holding_loc_X_vals, holding_loc_Y_vals)
#         holdings = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'holding_id', 'easting', 'northing', 'cattle'])
#         holdings['cell_id'] = holding_cell_IDs
#         # Reorder columns
#         holdings = holdings[['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle']]
#         # Create cells dataframe
#         cells = pd.DataFrame(columns=['cell_id', 'min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
#         cells['cell_id'] = range(len(holding_cell_IDs.unique()))
#         cells['min_x'] = grid_xMin
#         cells['min_y'] = grid_yMin
#         cells['max_x'] = grid_xMax
#         cells['max_y'] = grid_yMax
#         # For each cell_id, count the number of holdings in that cell
#         num_nodes = holdings['cell_id'].value_counts()
#         cells['num_nodes'] = cells['cell_id'].map(num_nodes)
#     else:
#         cells = pd.read_csv(params_land['grid_file_path'])
#         holdings = pd.read_csv(params_land['cattle_file_path'])
#         # Replace holdings['holding_id'] with range(len(holdings))
#         # holdings['holding_id'] = range(len(holdings))
#     # Get number of holdings per cell and assign to cells['num_nodes'] by cell_id
#     num_nodes = holdings['cell_id'].value_counts()
#     cells['num_nodes'] = cells['cell_id'].map(num_nodes)
#     # If NaN, then no holdings in cell
#     cells['num_nodes'] = cells['num_nodes'].fillna(0)
#     # Calculate bounding box for the landscape
#     bounding_box = (min(cells['min_x']), min(cells['min_y']), max(cells['max_x']), max(cells['max_y']))
#     # Get the clusters by iteration

#     return cells, holdings, bounding_box

def subdivide_cell(cell, holdings):
    """
    Split a cell into four equal quadrants.

    Inputs:
    - `cell`: Tuple containing (min_x, min_y, max_x, max_y, num_nodes).

    Outputs:
    - `child_cells`: List of four child cells, each with (min_x, min_y, max_x, max_y, num_nodes).
    """
    min_x, min_y, max_x, max_y, num_nodes = cell
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    child_cells = [
        (min_x, min_y, mid_x, mid_y, count_nodes_in_cell((min_x, min_y, mid_x, mid_y, num_nodes), holdings)),
        (min_x, mid_y, mid_x, max_y, count_nodes_in_cell((min_x, mid_y, mid_x, max_y, num_nodes), holdings)),
        (mid_x, min_y, max_x, mid_y, count_nodes_in_cell((mid_x, min_y, max_x, mid_y, num_nodes), holdings)),
        (mid_x, mid_y, max_x, max_y, count_nodes_in_cell((mid_x, mid_y, max_x, max_y, num_nodes), holdings))
    ]
    return child_cells

def count_nodes_in_cell(cell, holdings):
    """
    Count the number of holdings within a cell.

    Inputs:
    - `cell`: Tuple containing (min_x, min_y, max_x, max_y, num_nodes).
    - `holdings`: List of holdings, where each holding is a tuple (nation, region, county, easting, northing, cattle).

    Outputs:
    - `num_nodes`: Number of holdings within the cell.
    """
    min_x, min_y, max_x, max_y, _ = cell
    num_nodes = sum(
        1 for holding in holdings
        if min_x <= holding[3] < max_x and min_y <= holding[4] < max_y
    )
    return num_nodes

def calculate_variance(cell, avg_node_per_grid_target):
    """
    Calculate the cell variance.
    Inputs:
    - `cell`: Tuple containing (min_x, min_y, max_x, max_y, num_nodes).
    - `avg_node_per_grid_target`: Target average number of nodes per grid cell.
    Outputs:
    - `variance`: Variance of the number of nodes in the cell.
    """
    # print("cell: ", cell)
    # print("avg_node_per_grid_target: ", avg_node_per_grid_target)
    return (np.log(cell[4]) - np.log(avg_node_per_grid_target))**2

def adaptive_grid_construct_fn(avg_node_per_grid_target, initial_cell, holdings):
    """
    Construct an adaptive grid over the landscape.

    Inputs:
    - `avg_node_per_grid_target`: Target average number of nodes per grid cell.
    - `bounding_box`: Tuple containing (min_x, min_y, max_x, max_y, num_nodes).
    - `holdings`: List of holdings, where each holding is a tuple (nation, region, county, easting, northing, cattle).

    Outputs:
    - `cells`: List of final grid cells, each with (min_x, min_y, max_x, max_y, num_nodes).
    """
    cells = [initial_cell]
    while True:
        subdivided = False
        new_cells = []
        # print("Cells: ", cells)
        for cell in cells:
            num_nodes = cell[4]
            # if num_nodes > 0:
            parent_variance = calculate_variance(cell, avg_node_per_grid_target)
            child_cells = subdivide_cell(cell, holdings)
            children_with_nodes = 0
            child_variances = []
            for child_cell in child_cells:
                child_num_nodes = child_cell[4]
                if child_num_nodes > 0:
                    child_variances.append(calculate_variance(child_cell, avg_node_per_grid_target))
                    children_with_nodes += 1
            child_variance = np.sum(child_variances) / children_with_nodes if children_with_nodes > 0 else 0
            if parent_variance >= child_variance:
                # If the variance of the child cells is less than or equal to the parent cell, subdivide
                for child_cell in child_cells:
                    child_num_nodes = child_cell[4]
                    if child_num_nodes > 0:
                        new_cells.append((child_cell[0], child_cell[1], child_cell[2], child_cell[3], child_num_nodes))
                subdivided = True
            else:
                # If the variance of the child cells is greater than the parent cell, keep the parent cell
                new_cells.append((cell[0], cell[1], cell[2], cell[3], num_nodes))
        cells = new_cells
        if not subdivided:
            # print("Cells constructed: ", (cells))
            break
    return cells

def create_landscape(params_land, params_epi):
    if params_land['generate_grid_flag'] == 1:
        """
        Generate an adaptive grid based on the landscape and epidemiological parameters.

        Inputs:
        - `params_land`: Dictionary containing landscape parameters.
        - `params_epi`: Dictionary containing epidemiological parameters.

        Outputs:
        - `holdings`: DataFrame of holdings with assigned cell IDs.
        - `cells`: DataFrame of grid cells.
        """
        print('Using adaptive grid to generate grid cells')

        # Load landscape shapefile
        gdf = pd.read_csv(params_land['landscape_shp_path'])
        gdf = gdf[gdf['iteration'] == 0]

        # Calculate the bounding box of the data
        min_x, max_x = gdf['easting'].min(), gdf['easting'].max()
        min_y, max_y = gdf['northing'].min(), gdf['northing'].max()
        max_span = max(max_x - min_x, max_y - min_y)
        min_x -= (max_span - (max_x - min_x)) / 2
        max_x += (max_span - (max_x - min_x)) / 2
        min_y -= (max_span - (max_y - min_y)) / 2
        max_y += (max_span - (max_y - min_y)) / 2
        num_holdings = len(gdf)

        bounding_box = (min_x, min_y, max_x, max_y, num_holdings)

        # Get holdings as a list of tuples
        holdings = [
            (row['nation'], row['region'], row['county'], row['easting'], row['northing'], row['cattle'])
            for _, row in gdf.iterrows()
        ]

        # Calculate susceptibility and transmissibility
        cattle = np.array([holding[5] for holding in holdings])
        susceptibility = params_epi['susceptibility'] * (cattle ** params_epi['exp_susceptibility'])
        transmissibility = params_epi['transmissibility'] * (cattle ** params_epi['exp_transmissibility'])

        # if len(holdings) < 5000:
        #     grid_iterations = 20
        # else:
        grid_iterations = 50

        # Calculate the target average number of nodes per grid cell
        longest_side = max(max_x - min_x, max_y - min_y)
        kernel_lookup = run_model.construct_dispersal_kernel(longest_side)
        avg_node_per_grid_target = approx_node_per_cell_threshold_fn(
            susceptibility, transmissibility, kernel_lookup, 1, grid_iterations, longest_side
        )
        # avg_node_per_grid_target = 85.0680473372781
        initial_cell = (min_x, min_y, max_x, max_y, num_holdings)

        # Construct the adaptive grid
        cells = adaptive_grid_construct_fn(avg_node_per_grid_target, initial_cell, holdings)

        # Assign holdings to cells
        cell_id_map = {}
        for cell_id, cell in enumerate(cells):
            min_x, min_y, max_x, max_y, _ = cell
            for holding in holdings:
                if min_x <= holding[3] < max_x and min_y <= holding[4] < max_y:
                    if holding in cell_id_map:
                        print(f"Error: Holding {holding} assigned to multiple cells.")
                    cell_id_map[holding] = cell_id

        # # Assign holdings to cells
        # cell_id_map = {}
        # for cell_id, cell in enumerate(cells):
        #     min_x, min_y, max_x, max_y, _ = cell
        #     for holding in holdings:
        #         if min_x <= holding[3] < max_x and min_y <= holding[4] < max_y:
        #             cell_id_map[holding] = cell_id

        # Create DataFrame for holdings
        holdings_df = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'easting', 'northing', 'cattle'])
        holdings_df['cell_id'] = holdings_df.apply(
            lambda row: cell_id_map.get((row['nation'], row['region'], row['county'], row['easting'], row['northing'], row['cattle']), 0),
            axis=1
        )

        # Create DataFrame for cells
        cells_df = pd.DataFrame(cells, columns=['min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
        cells_df['cell_id'] = range(len(cells))

        # Reassign contiguous cell IDs
        unique_cell_ids = sorted(cells_df['cell_id'].unique())
        contiguous_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_cell_ids)}

        # Update cell IDs in both DataFrames
        cells_df['cell_id'] = cells_df['cell_id'].map(contiguous_id_map)
        holdings_df['cell_id'] = holdings_df['cell_id'].map(contiguous_id_map)

        # Add holding_id to holdings DataFrame
        holdings_df['holding_id'] = range(len(holdings_df))

        # Reorder columns in holdings DataFrame
        holdings_df = holdings_df[['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle']]

        cells_df.to_csv(params_land['grid_file_path'], index=False)
        holdings_df.to_csv(params_land['cattle_file_path'], index=False)
    elif params_land['generate_grid_flag'] == 2:
        # Create a single cell grid
        print('Using single cell grid')

        # Load landscape shapefile
        gdf = pd.read_csv(params_land['landscape_shp_path'])
        gdf = gdf[gdf['iteration'] == 0]
        # print(gdf)

        # Calculate the bounding box of the data
        min_x, max_x = gdf['easting'].min(), gdf['easting'].max()
        min_y, max_y = gdf['northing'].min(), gdf['northing'].max()
        max_span = max(max_x - min_x, max_y - min_y)
        min_x -= (max_span - (max_x - min_x)) / 2
        max_x += (max_span - (max_x - min_x)) / 2
        min_y -= (max_span - (max_y - min_y)) / 2
        max_y += (max_span - (max_y - min_y)) / 2
        num_holdings = len(gdf['easting'])

        bounding_box = (min_x, min_y, max_x, max_y, num_holdings)
        cells = [(min_x, min_y, max_x, max_y, num_holdings)]
        holdings = [
            (row['nation'], row['region'], row['county'], row['easting'], row['northing'], row['cattle'])
            for _, row in gdf.iterrows()
        ]
        # Create DataFrame for holdings
        holdings_df = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'easting', 'northing', 'cattle'])
        holdings_df['cell_id'] = 0  # All holdings belong to the single cell
        # Create DataFrame for cells
        cells_df = pd.DataFrame(cells, columns=['min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
        cells_df['cell_id'] = 0  # Single cell ID
        # # Reassign contiguous cell IDs
        # unique_cell_ids = sorted(cells_df['cell_id'].unique())
        # contiguous_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_cell_ids)}

        # # Update cell IDs in both DataFrames
        # cells_df['cell_id'] = cells_df['cell_id'].map(contiguous_id_map)
        # holdings_df['cell_id'] = holdings_df['cell_id'].map(contiguous_id_map)
        # Add holding_id to holdings DataFrame
        holdings_df['holding_id'] = range(len(holdings_df))
        # Reorder columns in holdings DataFrame
        holdings_df = holdings_df[['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle']]
        # Save the single cell grid to files
        cells_df.to_csv(params_land['grid_file_path'], index=False)
        holdings_df.to_csv(params_land['cattle_file_path'], index=False)
    else:
        if os.path.isfile(params_land['grid_file_path']) and os.path.isfile(params_land['cattle_file_path']):
            print('Loading grid cells and holdings from files...')
            # Load existing cells and holdings from files
            cells_df = pd.read_csv(params_land['grid_file_path'])
            holdings_df = pd.read_csv(params_land['cattle_file_path'])
            # Load landscape shapefile
            gdf = pd.read_csv(params_land['landscape_shp_path'])
            gdf = gdf[gdf['iteration'] == 0]
            # print(gdf)

            # Calculate the bounding box of the data
            min_x, max_x = gdf['easting'].min(), gdf['easting'].max()
            min_y, max_y = gdf['northing'].min(), gdf['northing'].max()
            max_span = max(max_x - min_x, max_y - min_y)
            min_x -= (max_span - (max_x - min_x)) / 2
            max_x += (max_span - (max_x - min_x)) / 2
            min_y -= (max_span - (max_y - min_y)) / 2
            max_y += (max_span - (max_y - min_y)) / 2
            num_holdings = len(gdf['easting'])

            bounding_box = (min_x, min_y, max_x, max_y, num_holdings)
        else:
            raise FileNotFoundError("Grid cells or holdings files not found.")

    return cells_df, holdings_df, bounding_box

# #------------------------------------------------------------------------------------------
# # Load the required libraries
# #------------------------------------------------------------------------------------------
# import numpy as np
# import math
# import pandas as pd
# import geopandas as gpd
# from statistics import median

# import sys
# sys.path.append('../../src')
# import run_model

# #------------------------------------------------------------------------------------------
# # Supporting functions
# #------------------------------------------------------------------------------------------
# def get_longest_landscape_edge(min_x, min_y, max_x, max_y):
#     """
#     Function to calculate the longest edge of the landscape.

#     Inputs:
#     - `min_x`: Minimum x coordinate.
#     - `min_y`: Minimum y coordinate.
#     - `max_x`: Maximum x coordinate.
#     - `max_y`: Maximum y coordinate.

#     Outputs:
#     - `longest_edge`: Length of the longest edge of the landscape.
#     """
#     longest_edge = max(max_x - min_x, max_y - min_y)
#     return longest_edge

# def create_landscape(params_land, params_epi):
#     if params_land['generate_grid_flag'] == 1:
#         # Generate grid
#         holdings, cells = adaptive_grid(params_land, params_epi)
#         holdings = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle'])
#         cells = pd.DataFrame(cells, columns=['cell_id', 'min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
#         cells.to_csv(params_land['grid_file_path'], index=False)
#         holdings.to_csv(params_land['cattle_file_path'], index=False)
#     if params_land['generate_grid_flag'] == 2:
#         print('Using adaptive grid 2 to generate grid cells')
    
#         # Load landscape shapefile
#         gdf = pd.read_csv(params_land['landscape_shp_path'])
#         # Specify order of columns
#         gdf = gdf[['nation', 'region', 'county', 'easting', 'northing', 'cattle', 'iteration']]
#         # Get 0th iteration
#         gdf = gdf[gdf['iteration'] == 0]

#         # Calculate the bounding box of the data
#         min_x = min(gdf['easting'])
#         max_x = max(gdf['easting'])
#         min_y = min(gdf['northing'])
#         max_y = max(gdf['northing'])
#         # Extend bounds so it is a square
#         max_span = max(max_x - min_x, max_y - min_y)
#         min_x = min_x - (max_span - (max_x - min_x)) / 2
#         max_x = max_x + (max_span - (max_x - min_x)) / 2
#         min_y = min_y - (max_span - (max_y - min_y)) / 2
#         max_y = max_y + (max_span - (max_y - min_y)) / 2
#         num_holdings = len(gdf['northing'])
#         bounding_box_var = [min_x, min_y, max_x, max_y, num_holdings]

#         # Construct the dispersal kernel using euclidean distance
#         greatest_distance = holding_euclidean_distance(min_x, min_y, max_x, max_y)
#         kernel_lookup = run_model.construct_dispersal_kernel(greatest_distance)
#         if kernel_lookup is []:
#             kernel_lookup = run_model.construct_dispersal_kernel(greatest_distance+1)
        
#         # Get the holdings (easting, northing, cattle)
#         holdings = [(holding[0], holding[1], holding[2], holding[3], holding[4], holding[5], holding[6]) for holding in gdf.values]
#         # Get cattle for each holding in an array
#         cattle = [holding[5] for holding in holdings]
#         # susceptibility = (params_epi['susceptibility'] * np.array(cattle))**(params_epi['exp_susceptibility'])
#         # transmissibility = (params_epi['transmissibility'] * np.array(cattle))**(params_epi['exp_transmissibility'])
#         susceptibility = params_epi['susceptibility'] * (np.array(cattle)**params_epi['exp_susceptibility'])
#         transmissibility = params_epi['transmissibility'] * (np.array(cattle)**params_epi['exp_transmissibility'])

#         longest_side = get_longest_landscape_edge(min_x, min_y, max_x, max_y)

#         avg_node_per_grid_target = approx_node_per_cell_threshold_fn(susceptibility, transmissibility, kernel_lookup, 1, 50, longest_side)
#         print("avg_node_per_grid_target: ", avg_node_per_grid_target)

#         longest_landscape_edge = get_longest_landscape_edge(min_x, min_y, max_x, max_y)
#         holding_loc_X_vals = [holding[3] for holding in holdings]
#         holding_loc_Y_vals = [holding[4] for holding in holdings]
#         # Generate grid with adaptive_grid
#         holding_cell_IDs, grid_xMin, grid_xMax, grid_yMin, grid_yMax = adaptive_grid.adaptive_grid_construct_fn(avg_node_per_grid_target, bounding_box_var, longest_landscape_edge, holding_loc_X_vals, holding_loc_Y_vals)
#         holdings = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'holding_id', 'easting', 'northing', 'cattle'])
#         holdings['cell_id'] = holding_cell_IDs
#         # Reorder columns
#         holdings = holdings[['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle']]
#         # Create cells dataframe
#         cells = pd.DataFrame(columns=['cell_id', 'min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
#         cells['cell_id'] = range(len(holding_cell_IDs.unique()))
#         cells['min_x'] = grid_xMin
#         cells['min_y'] = grid_yMin
#         cells['max_x'] = grid_xMax
#         cells['max_y'] = grid_yMax
#         # For each cell_id, count the number of holdings in that cell
#         num_nodes = holdings['cell_id'].value_counts()
#         cells['num_nodes'] = cells['cell_id'].map(num_nodes)
#     else:
#         cells = pd.read_csv(params_land['grid_file_path'])
#         holdings = pd.read_csv(params_land['cattle_file_path'])
#         # Replace holdings['holding_id'] with range(len(holdings))
#         # holdings['holding_id'] = range(len(holdings))
#     # Get number of holdings per cell and assign to cells['num_nodes'] by cell_id
#     num_nodes = holdings['cell_id'].value_counts()
#     cells['num_nodes'] = cells['cell_id'].map(num_nodes)
#     # If NaN, then no holdings in cell
#     cells['num_nodes'] = cells['num_nodes'].fillna(0)
#     # Calculate bounding box for the landscape
#     bounding_box = (min(cells['min_x']), min(cells['min_y']), max(cells['max_x']), max(cells['max_y']))
#     # Get the clusters by iteration

#     return cells, holdings, bounding_box

# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371000  # radius of Earth in meters
#     phi_1 = np.radians(lat1)
#     phi_2 = np.radians(lat2)
#     delta_phi = np.radians(lat2 - lat1)
#     delta_lambda = np.radians(lon2 - lon1)
#     a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     meters = R * c  # output distance in meters
#     return meters

# # def euclidean_distance(x1, y1, x2, y2):
# #     return np.sqrt(abs((x2 - x1)**2 + (y2 - y1)**2))

# # def holding_euclidean_distance(northing_i, easting_i, northing_j, easting_j):
# #     return math.sqrt(math.pow(northing_j - northing_i, 2) + math.pow(easting_j - easting_i, 2))

# # def holding_euclidean_distance(northing_i, easting_i, northing_j, easting_j):
#     # return euclidean((northing_i, easting_i), (northing_j, easting_j))

def holding_euclidean_distance(easting_i, northing_i, easting_j, northing_j):
    return np.sqrt((northing_i - northing_j) ** 2 + (easting_i - easting_j) ** 2)

def bounding_box_diagonal(min_x, max_x, min_y, max_y):
    return math.sqrt(math.pow(max_x - min_x, 2) + math.pow(max_y - min_y, 2))

# def cell_euclidean_distance(ix1, iy1, ix2, iy2, jx1, jy1, jx2, jy2):
#     cell1 = Polygon([(ix1, iy1), (ix2, iy1), (ix2, iy2), (ix1, iy2)])
#     cell2 = Polygon([(jx1, jy1), (jx2, jy1), (jx2, jy2), (jx1, jy2)])
#     return cell1.distance(cell2)

def shortest_distance_between_squares(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2):
    def distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # min_x1, min_y1, max_x1, max_y1 = square1
    # min_x2, min_y2, max_x2, max_y2 = square2

    # If squares overlap or touch
    if (max_x1 >= min_x2 and min_x1 <= max_x2 and max_y1 >= min_y2 and min_y1 <= max_y2):
        return 0.0  # Cells overlap or touch

    # If squares do not overlap or touch, calculate the shortest distance
    if max_x1 < min_x2:  # square1 is to the left of square2
        if max_y1 < min_y2:  # square1 is below square2
            return distance(max_x1, max_y1, min_x2, min_y2)
        elif min_y1 > max_y2:  # square1 is above square2
            return distance(max_x1, min_y1, min_x2, max_y2)
        else:  # square1 is horizontally aligned with square2
            return min_x2 - max_x1
    elif min_x1 > max_x2:  # square1 is to the right of square2
        if max_y1 < min_y2:  # square1 is below square2
            return distance(min_x1, max_y1, max_x2, min_y2)
        elif min_y1 > max_y2:  # square1 is above square2
            return distance(min_x1, min_y1, max_x2, max_y2)
        else:  # square1 is horizontally aligned with square2
            return min_x1 - max_x2
    else:  # square1 is vertically aligned with square2
        if max_y1 < min_y2:  # square1 is below square2
            return min_y2 - max_y1
        elif min_y1 > max_y2:  # square1 is above square2
            return min_y1 - max_y2

    return 0.0

def get_distance_between_cell_pairs(A_x, A_y, B_x, B_y, longest_side, grid_resolution):
    """
    Function used in approximation of optimum number of nodes per cell: Compute distances between pairs of grid cells.

    Inputs:
    - `A_x, A_y`: Position for cell A.
    - `B_x, B_y`: Position for cell B.
    - `longest_side`: Length of landscape.
    - `grid_resolution`: Number of cells per side.
    - `coord_type`: Value 1 ("Cartesian", metres), 2 ("Cartesian", km) or 3 ("LatLong").

    Outputs:
    - `cell_pair_dist`: Threshold number of nodes per cell. To be used in adaptive grid configuration construction.
    """
    a = 0
    b = 0

    if A_x == B_x:
        a = 0
    elif A_x < B_x:
        a = 1
    elif A_x > B_x:
        a = -1

    if A_y == B_y:
        b = 0
    elif A_y < B_y:
        b = 1
    elif A_y > B_y:
        b = -1

    t1 = ((A_x + a) * longest_side - B_x * longest_side) / grid_resolution
    t2 = ((A_y + b) * longest_side - B_y * longest_side) / grid_resolution
    d_sq = t1 ** 2 + t2 ** 2
    cell_pair_dist = np.sqrt(d_sq)

    return cell_pair_dist

def approx_node_per_cell_threshold_fn(susceptibility, transmissibility, kernel_lookup, delta_t, approx_node_per_cell_grid_size_threshold, longest_side):
    print("Calculating approx_node_per_cell_threshold_fn")

    get_trans = max(transmissibility)
    get_susc = max(susceptibility)

    # Set up cell resolutions to be tested
    k_to_test = np.arange(1, approx_node_per_cell_grid_size_threshold + 1)

    # Initialise variables used in main loop
    k_element = 0 # Changed this from 0 to 1
    k_calls_vector = np.zeros(len(k_to_test))
    avg_k_calls_vector = np.zeros(len(k_to_test))
    nodes_per_cell_vec = np.zeros(len(k_to_test))

    # Iterate over the different grid configuration
    for grid_resolution_itr in range(len(k_to_test)):

        # print("Grid resolution iteration: ", grid_resolution_itr)

        # Get number of cells per side of grid
        k = k_to_test[grid_resolution_itr]

        # Find number of nodes per cell, assign to storage vector
        num_nodes = len(susceptibility)
        nodes_per_cell = num_nodes / (k * k)
        nodes_per_cell_vec[grid_resolution_itr] = nodes_per_cell

        k_calls_this_k = 0  # number of kernel calls in total for this grid conf.
        if k == 1:  # No gridding (one large cell). Only pairwise.
            k_calls_this_k += (nodes_per_cell - 1)
        else:  # More than one cell present

            # Get indexes for each grid based on value of k
            x_val_all = np.arange(1, k + 1)  # x dimension
            y_val_all = np.arange(1, k + 1)  # y dimension

            # Get coordinates for cells in first quadrant
            if k % 2 == 0:  # when k is even
                quadrant_width = int(k / 2)
                quadrant_height = int(k / 2)
            else:  # when k is odd
                quadrant_width = int((k + 1) / 2)
                quadrant_height = int((k - 1) / 2)

            x_val_subset = np.arange(1, quadrant_width + 1)
            y_val_subset = np.arange(1, quadrant_height + 1)

            # Loop over all pairs of cells (first cell A x and y, and then cell B x and y).
            for A_x_itr in range(len(x_val_subset)):  # Cell A_x coordinates.
                for A_y_itr in range(len(y_val_subset)):  # Cell A_y coordinates.
                    A_x = x_val_subset[A_x_itr]
                    A_y = y_val_subset[A_y_itr]
                    # For this pair of Ax and Ay do all B cells (including A itself).
                    k_calls_A = (nodes_per_cell - 1)  # Pairwise within the cell itself.
                    for B_x_itr in range(len(x_val_all)):  # Cell B_x coordinates.
                        for B_y_itr in range(len(y_val_all)):  # Cell B_x coordinates.
                            B_x = x_val_all[B_x_itr]
                            B_y = y_val_all[B_y_itr]
                            if not (A_x == B_x and A_y == B_y):  # Only if A and B are not the same cell.
                                d = get_distance_between_cell_pairs(A_x, A_y, B_x, B_y, longest_side, k)
                                if d > longest_side:
                                    d = longest_side
                                # print("d: ", d)
                                # print("len(kernel_lookup): ", len(kernel_lookup))
                                kernel_value = kernel_lookup[int(round(d,0))]
                                exponent = -get_trans * get_susc * kernel_value * delta_t
                                p_over = -np.expm1(exponent)
                                k_calls_this_pair = (nodes_per_cell * p_over) + 1  # Expected n kernel calls for Cell(Ax, Ay)->Cell(Bx, By) plus one for the calculation of p_over.
                                k_calls_A += k_calls_this_pair  # Add the kernel calls between cells A and B to A's total.
                    k_calls_this_k += 4 * k_calls_A  # Add A's total kernel calls to the grand total (cells equivalent to A occur 4 times in the landscape).

            # When all pairs have been looped over and k is odd there still remains the central cell in the configuration.
            if k % 2 == 1:
                # Get grid index for central grid
                A_x = (k + 1) // 2
                A_y = (k + 1) // 2
                k_calls_A = (nodes_per_cell - 1)  # Pairwise within the cell itself.
                for B_x_itr in range(len(x_val_all)):  # Cell B_x coordinates.
                    for B_y_itr in range(len(y_val_all)):  # Cell B_x coordinates.
                        B_x = x_val_all[B_x_itr]
                        B_y = y_val_all[B_y_itr]
                        if not (A_x == B_x and A_y == B_y):  # Only if A and B are not the same cell.
                            d = get_distance_between_cell_pairs(A_x, A_y, B_x, B_y, longest_side, k)
                            if d > longest_side:
                                d = longest_side
                            kernel_value = kernel_lookup[int(round(d,0))]
                            exponent = -get_trans * get_susc * kernel_value * delta_t
                            p_over = -np.expm1(exponent)
                            k_calls_this_pair = (nodes_per_cell * p_over) + 1  # Expected n kernel calls for Cell(Ax, Ay)->Cell(Bx, By) plus one for the calculation of p_over.
                            k_calls_A += k_calls_this_pair  # Add the kernel calls between cells A and B to A's total
                k_calls_this_k += k_calls_A

        # Assign kernel calls for this k value to storage vector
        k_calls_vector[k_element] = k_calls_this_k

        # Assign average number of expected kernel function call per cell to storage vector
        # Total number of cells is k*k
        avg_k_calls_vector[k_element] = k_calls_this_k / (k * k)

        # Increment k_element
        k_element += 1
    # print("k_calls_vector: ", k_calls_vector)
    # print("avg_k_calls_vector: ", avg_k_calls_vector)
    # print("nodes_per_cell_vec: ", nodes_per_cell_vec)

    # Find configuration that returned the lowest number of kernel calls
    k_min_idx = np.argmin(avg_k_calls_vector)
    # print("k_min_idx: ", k_min_idx)

    # Get expected number of nodes per cell, given selected configuration
    approx_node_per_cell_threshold = nodes_per_cell_vec[k_min_idx]

    print("approx_node_per_cell_threshold: ", approx_node_per_cell_threshold)

    return approx_node_per_cell_threshold

# def should_divide(holdings, cell, depth, threshold_lambda):
#     log_lambda = np.log(threshold_lambda)
#     if cell[4] == 0:
#         return False
#     log_La = np.log(cell[4])
#     log_La_diff = (log_La - log_lambda) ** 2
    
#     B = []
#     for i in range(2):
#         for j in range(2):
#             min_x, min_y, max_x, max_y, num_nodes = cell
#             child_min_x = min_x + i * (max_x - min_x) / 2
#             child_max_x = min_x + (i + 1) * (max_x - min_x) / 2
#             child_min_y = min_y + j * (max_y - min_y) / 2
#             child_max_y = min_y + (j + 1) * (max_y - min_y) / 2
#             # Count the number of holdings within this child cell
#             child_nodes = 0
#             for holding in holdings:
#                 if holding[3] >= child_min_x and holding[3] <= child_max_x and holding[4] >= child_min_y and holding[4] <= child_max_y and holding[5] > 0:
#                     child_nodes += 1
#             child_cell = (child_min_x, child_min_y, child_max_x, child_max_y, child_nodes)
#             # if len(child_cell) > 0:
#             B.append(child_cell)
#     B_nodes = np.sum([1 if child[4] > 0 else 0 for child in B]) # Should be between 1 and 4 (number of cells with holdings in them)
#     log_Lb_sum = np.sum([(np.log(child[4]) - log_lambda) ** 2 if child[4] > 0 else 0 for child in B]) / B_nodes
    
#     return log_La_diff >= log_Lb_sum

# def assign_location_to_grid(holdings, holding, cells, cell_ids, cell_id, cell_count):
#     # Check if the grid ID already exists in the dictionary
#     if cell_id in cell_ids:
#         cell_ids[cell_id].append(holding)
#         print(holding)
#     else:
#         cell_ids[cell_id] = [holding]
#         print(f"Cell ID {cell_id} added to dictionary")
#         cell_count += 1
#     return cell_count

# def recursive_divide(holdings, cell, depth, max_depth, threshold_lambda, cells, cell_ids, assigned_count, cell_count):
#     if assigned_count >= len(holdings):
#         print(f"Assigned all holdings: {assigned_count} >= {len(holdings)}")
#         return assigned_count, cell_count
#     # Check if the cell should be divided further
#     if depth >= max_depth or not should_divide(holdings, cell, depth, threshold_lambda) or cell[4] == 0:        
#         if cell[4] != 0:
#             # Assign holdings within this cell to the current grid cell ID
#             cell = (cell[0], cell[1], cell[2], cell[3], cell[4])
#             print(cell)
#             cell_id = len(cells)
#             cells.append(cell)
#             # Check which lat longs will be assigned to this grid cell
#             for holding in holdings:
#                 if holding[3] >= cell[0] and holding[3] <= cell[2] and holding[4] >= cell[1] and holding[4] <= cell[3]:
#                     cell_count = assign_location_to_grid(holdings, holding, cells, cell_ids, cell_id, cell_count)
#                     assigned_count = assigned_count + 1
#                     print(f"Assigned count: {assigned_count}")
#     else:
#         # Divide the cell into four equal-sized children cells
#         child_cells = []
#         for i in range(2):
#             for j in range(2):
#                 min_x, min_y, max_x, max_y, num_nodes = cell
#                 child_min_x = min_x + i * (max_x - min_x) / 2
#                 child_max_x = min_x + (i + 1) * (max_x - min_x) / 2
#                 child_min_y = min_y + j * (max_y - min_y) / 2
#                 child_max_y = min_y + (j + 1) * (max_y - min_y) / 2
#                 # Convert from float to int
#                 child_min_x = int(child_min_x)
#                 child_max_x = int(child_max_x)
#                 child_min_y = int(child_min_y)
#                 child_max_y = int(child_max_y)
#                 # Count the number of holdings within this child cell
#                 child_nodes = 0
#                 for holding in holdings:
#                     if holding[3] >= child_min_x and holding[3] <= child_max_x and holding[4] >= child_min_y and holding[4] <= child_max_y and holding[5] > 0:
#                         child_nodes += 1
#                 child_cells.append((child_min_x, child_min_y, child_max_x, child_max_y, child_nodes))
 
#         # Recursively divide children cells
#         for child_cell in child_cells:
#             recursive_divide(holdings, child_cell, depth + 1, max_depth, threshold_lambda, cells, cell_ids, assigned_count, cell_count)

#     return assigned_count, cell_count

# def adaptive_grid(params_land, params_epi):
#     print('Using adaptive grid to generate grid cells')
    
#     # Load landscape shapefile
#     gdf = pd.read_csv(params_land['landscape_shp_path'])
#     # Specify order of columns
#     gdf = gdf[['nation', 'region', 'county', 'easting', 'northing', 'cattle', 'iteration']]
#     # Get 0th iteration
#     gdf = gdf[gdf['iteration'] == 0]

#     # Calculate the bounding box of the data
#     min_x = min(gdf['easting'])
#     max_x = max(gdf['easting'])
#     min_y = min(gdf['northing'])
#     max_y = max(gdf['northing'])
#     # Extend bounds so it is a square
#     max_span = max(max_x - min_x, max_y - min_y)
#     min_x = min_x - (max_span - (max_x - min_x)) / 2
#     max_x = max_x + (max_span - (max_x - min_x)) / 2
#     min_y = min_y - (max_span - (max_y - min_y)) / 2
#     max_y = max_y + (max_span - (max_y - min_y)) / 2
#     num_holdings = len(gdf['northing'])

#     initial_cell = (min_x, min_y, max_x, max_y, num_holdings)

#     # Construct the dispersal kernel using euclidean distance
#     greatest_distance = holding_euclidean_distance(min_x, min_y, max_x, max_y)
#     kernel_lookup = run_model.construct_dispersal_kernel(greatest_distance)
#     if kernel_lookup is []:
#         kernel_lookup = run_model.construct_dispersal_kernel(greatest_distance+1)
    
#     # Get the holdings (easting, northing, cattle)
#     holdings = [(holding[0], holding[1], holding[2], holding[3], holding[4], holding[5], holding[6]) for holding in gdf.values]
#     # Get cattle for each holding in an array
#     cattle = [holding[5] for holding in holdings]
#     # print("cattle: ", cattle)
#     susceptibility = params_epi['susceptibility'] * (np.array(cattle)**params_epi['exp_susceptibility'])
#     transmissibility = params_epi['transmissibility'] * (np.array(cattle)**params_epi['exp_transmissibility'])

#     longest_side = get_longest_landscape_edge(min_x, min_y, max_x, max_y)

#     # Check kernel calls
#     theta_star = approx_node_per_cell_threshold_fn(susceptibility, transmissibility, kernel_lookup, 1, 20, longest_side)
#     # theta_star = approx_node_per_cell_threshold_fn(holdings, initial_cell, susceptibility, transmissibility, kernel_lookup, 1, 50)
#     print(f"theta_star: {theta_star}")

#     # Define parameters
#     max_depth = 20  # Maximum depth for recursion
#     # theta_star = 22
#     # theta_star = 87
#     # theta_star = 200.467  # Optimal value of theta
#     eta = 1  # Constant to control grid configuration span
#     threshold_lambda = eta*theta_star  # Threshold value of lambda

#     # Initialize the list of grid cells and IDs
#     cells = []
#     cell_ids = {}  # A dictionary to store location IDs corresponding to grid cell IDs

#     # Start the recursive division
#     assigned_count = 0
#     cell_count = 0
#     print(f"Initial cell: {initial_cell}")
#     recursive_divide(holdings, initial_cell, 0, max_depth, threshold_lambda, cells, cell_ids, assigned_count, cell_count)

#     print(cells)

#     # Add cell_id to holdings
#     holdings = pd.DataFrame(holdings, columns=['nation', 'region', 'county', 'easting', 'northing', 'cattle', 'iteration'])
#     # Drop iteration column
#     holdings = holdings.drop(columns=['iteration'])

#     # Add cell_id to cells
#     cells = pd.DataFrame(cells, columns=['min_x', 'min_y', 'max_x', 'max_y', 'num_nodes'])
#     cells['cell_id'] = range(len(cells['min_x']))

#     # Assign each holding to a cell
#     holdings['cell_id'] = -1
#     holdings['holding_id'] = range(len(holdings))
#     # Sort cell_ids by cell_id
#     cell_ids = dict(sorted(cell_ids.items()))
#     for cell_id, holding_list in cell_ids.items():
#         for holding in holding_list:
#             holdings.loc[(holdings['easting'] == holding[3]) & (holdings['northing'] == holding[4]), 'cell_id'] = cell_id

#     # Specify order of columns
#     holdings = holdings[['nation', 'region', 'county', 'cell_id', 'holding_id', 'easting', 'northing', 'cattle']]

#     # Sort holdings by cell_id
#     holdings = holdings.sort_values(by='holding_id')

#     return holdings, cells