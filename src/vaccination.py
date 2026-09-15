from scipy.spatial import cKDTree
import numpy as np

def vaccinate_holdings(t, events, holding_status, holding_vax_status, holding_cell, holding_easting, holding_northing, holding_transmissibility, holding_susceptibility, holding_behaviour, cell_status, cell_vax_status, vaccination_event_ID, day_infectious, day_reported, params_behav, params_epi):
    """
    Vaccinate holdings based on farmer behaviour
    """
    # Unpack behavioural parameters
    epidemic_stage_distances = params_behav['epidemic_stage_distances']
    day_culled = params_epi['day_culled']

    # Get the current reported infections
    reported_infections = np.where((holding_status >= day_reported) & (holding_status <= day_culled))[0]

    if len(reported_infections) == 0:
        return
    
    # Build a KDTree of the reported infections
    reported_infections_tree = cKDTree(list(zip(holding_easting[reported_infections], holding_northing[reported_infections])))

    # Vaccinate the holdings based on vaccination_distance
    for i, stage in enumerate(epidemic_stage_distances):

        vaccination_distance = params_behav['epidemic_stage_distances'][i]

        # Based on holding_behaviour[1] which is the vaccination_stage, determine which holdings are to be vaccinated
        stage_holdings = np.where((holding_behaviour[1] == stage) & (holding_status < day_infectious) & (holding_vax_status == -1))[0]

        if len(stage_holdings) == 0:
            continue

        # Get the coordinates of the holdings in the behavioural group
        stage_holdings_coords = list(zip(holding_easting[stage_holdings], holding_northing[stage_holdings]))

        # Query the KDTree for the nearest reported infection to each holding in the behavioural group
        distances, indices = reported_infections_tree.query(stage_holdings_coords)

        # Vaccinate the holdings that are within the vaccination distance
        for i, (distance, index) in enumerate(zip(distances, indices)):
            if distance <= vaccination_distance:
                holding = stage_holdings[i]
                # Check if holding_status is still susceptible
                if holding_status[holding] == -1:
                    holding_vax_status[holding] += 1
                    cell_vax_status[0][holding_cell[0][holding]] -= 1
                    cell_vax_status[1][holding_cell[0][holding]] += 1
                    events[vaccination_event_ID][holding] = t
                # Check if holding_status is infected
                if ((holding_status[holding] > -1) & (holding_status[holding] < day_infectious)):
                    holding_vax_status[holding] = 6
                    cell_vax_status[0][holding_cell[0][holding]] -= 1
                    cell_vax_status[6][holding_cell[0][holding]] += 1
                    events[vaccination_event_ID+6][holding] = t

    return