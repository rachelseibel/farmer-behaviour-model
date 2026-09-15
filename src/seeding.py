import numpy as np
from scipy.spatial import cKDTree

def seed_infection(events, t, holdings, holding_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, cell_status, holding_idxs, params_sim, county_id, landscape_iteration, seed_iteration):
    seed_flag = params_sim['seed_flag']
    num_seeds = params_sim['num_seeds']
    if seed_flag == 1:  # Seed infection in X random holdings
        print('Seeding in random holdings...')
        seeds = np.random.choice([i for i in range(len(holding_status)) if holding_status[i] == 0], num_seeds)
        holding_status[seeds] += 1
        seed_cells = holding_cell[0][seeds]
        for i in range(len(seed_cells)):
            cell_status[0][seed_cells[i]] -= 1
            cell_status[1][seed_cells[i]] += 1
        events[1][seeds] = t
        # print('Cell status: ', cell_status[1][seed_cells])
    elif seed_flag == 2:    # Seed infection in a random cluster of X holdings\
        print('Seeding in a random cluster...')
        seed = np.random.choice([i for i in range(len(holding_status)) if holding_status[i] == 0], 1)
        # Convert holding coordinates to array of points
        holding_coordinates = np.array((holding_easting, holding_northing)).T
        # Construct a KD-tree
        tree = cKDTree(holding_coordinates)
        # Query the tree for X nearest holdings to seed
        seeds = tree.query(holding_coordinates[int(seed[0])], num_seeds)
        # Get indices of nearest holdings
        seed_idxs = list(seeds[1])
        holding_status[seed_idxs] += 1
        # Get seed cells
        seed_cells = holding_cell[0][seed_idxs]
        for i in range(len(seed_cells)):
            cell_status[0][seed_cells[i]] -= 1
            cell_status[1][seed_cells[i]] += 1
        events[1][seed_idxs] = t
    elif seed_flag == 3:    # Seed infection in a random cluster of X holdings in a specific county
        print('Seeding in a specific county...')
        np.random.seed(params_sim['random_seed'] + int(county_id) + int(landscape_iteration) + int(seed_iteration))
        seed_county = params_sim['seed_county']
        seeded_holdings = holdings[holdings['county'] == seed_county]
        # Check if there are holdings in the county - if not, print error
        if seeded_holdings.empty:
            print('No holdings in the county')
            return
        seed = np.random.choice(seeded_holdings.index, 1)
        holding_coordinates = np.array((holding_easting, holding_northing)).T
        tree = cKDTree(holding_coordinates)
        seeds = tree.query(holding_coordinates[int(seed[0])], num_seeds)
        seed_idxs = list(seeds[1])
        holding_status[seed_idxs] += 1
        seed_cells = holding_cell[0][seed_idxs]
        print('Seed cells: ', seed_cells)
        print('Seed idxs: ', seed_idxs)
        for i in range(len(seed_cells)):
            cell_status[0][seed_cells[i]] -= 1
            cell_status[1][seed_cells[i]] += 1
        events[1][seed_idxs] = t
        return events, holding_status, cell_status
    elif seed_flag == 4:    # Seed infection in a specific holding
        print('Seeding in a specific holding...')
        seed_holding = params_sim['seed_holding']
        seed = seed_holding
        holding_status[seed] += 1
        seed_cells = holding_cell[0][seed]
        cell_status[0][seed_cells] -= 1
        cell_status[1][seed_cells] += 1
        events[1][seed] = t
    elif seed_flag == 5:    # Seed infection in a specific holding holding and the next nearest 2 holdings
        print('Seeding in a specific holding and the next nearest 2 holdings...')
        seed_holding = params_sim['seed_holding']
        seed = seed_holding
        holding_status[seed] += 1
        seed_cells = holding_cell[0][seed]
        cell_status[0][seed_cells] -= 1
        cell_status[1][seed_cells] += 1
        events[1][seed] = t
        # Convert holding coordinates to array of points
        holding_coordinates = np.array((holding_easting, holding_northing)).T
        # Construct a KD-tree
        tree = cKDTree(holding_coordinates)
        # Query the tree for X nearest holdings to seed
        seeds = tree.query(holding_coordinates[int(seed)], num_seeds)
        # Get indices of nearest holdings
        seed_idxs = list(seeds[1])
        holding_status[seed_idxs] += 1
        # Get seed cells
        seed_cells = holding_cell[0][seed_idxs]
        for i in range(len(seed_cells)):
            cell_status[0][seed_cells[i]] -= 1
            cell_status[1][seed_cells[i]] += 1
        events[1][seed_idxs] = t
        return events, holding_status, cell_status
    else:
        print('Infection seeding not implemented')
        pass
    return