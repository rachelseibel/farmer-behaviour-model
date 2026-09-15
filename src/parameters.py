#------------------------------------------------------------------------------------------
# Define the parameters for the model
#------------------------------------------------------------------------------------------
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths are read from the environment so the model runs from a clone without
# editing this file. Defaults point at the repository's own sampledata/ folder,
# which tools/make_sample_landscape.py fills with synthetic data.
#
#   LIVESTOCK_INPUT_DIR   grid cells, holdings, distances, transmission probs
#   LIVESTOCK_LANDSCAPE_DIR  behavioural landscape realisations
#   LIVESTOCK_STATUS_DIR  where run-status files are written
#   LIVESTOCK_COUNTY_NAMES  county name lookup csv
#
# No restricted data lives in this repository; see DATA.md.
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.environ.get("LIVESTOCK_INPUT_DIR", os.path.join(_REPO, "sampledata"))
LANDSCAPE_DIR = os.environ.get("LIVESTOCK_LANDSCAPE_DIR", os.path.join(_REPO, "sampledata"))
STATUS_DIR = os.environ.get("LIVESTOCK_STATUS_DIR", os.path.join(_REPO, "status"))
COUNTY_NAMES = os.environ.get("LIVESTOCK_COUNTY_NAMES",
                              os.path.join(_REPO, "sampledata", "county_names.csv"))


def parameter_loading(args):

    path = STATUS_DIR if STATUS_DIR.endswith(os.sep) else STATUS_DIR + os.sep
    extension = ''
    extension_no_score = extension.replace('_', '')

    # Epidemiological parameters
    params_epi = {
        'susceptibility': 1,
        'transmissibility': 1e6,
        'exp_susceptibility': 0.41,
        'exp_transmissibility': 0.42,
        'day_infectious': 6,
        'day_reported': 10,
        'day_culled': 14,
        'vaccine_effectiveness_delay': 5,
    }
    # Landscape parameters
    params_land = {
        'generate_grid_flag': 0,
        'calculate_holding_distances_flag': 0,
        'calculate_transmission_probabilities_flag': 0,
        'input_folder': INPUT_DIR + os.sep,
        'landscape_shp_path': os.path.join(LANDSCAPE_DIR, args['landscape_method'] + '.csv'),
        'grid_file_path': os.path.join(INPUT_DIR, 'grid_cells' + extension + '.csv'),
        'cattle_file_path': os.path.join(INPUT_DIR, 'grid_holdings' + extension + '.csv'),
        'transmission_file_path': os.path.join(INPUT_DIR, 'transmission_probabilities' + extension + '.csv'),
        'distance_file_path': os.path.join(INPUT_DIR, 'holding_pair_distances_by_cell' + extension + '.pkl')
    }
    # Behavioural parameters
    params_behav = {
        'behaviour_flag': 1,
        'behavioural_group_names': [1, 2, 3, 4],
        # 'behavioural_group_names': [1],
        'epidemic_stage_names': ['early', 'mid', 'late', 'never'],
        # 'epidemic_stage_names': ['late'],
        'epidemic_stage_distances': [1e10, 320000, 50000, -1], # distances to nearest infected herd in metres
        # 'epidemic_stage_distances': [50000], # distances to nearest infected herd in metres
        # Dictionary by behavioural group and epidemic stage with proportion of farmers vaccinating
        # early - From first emergence in GB (i.e. from outset of simulation)
        # mid - Within 200 miles (we use 320km)
        # late - Within 30 miles (we use 50km)
        # never - Never (non-users)
        'vaccination_by_group': {
            1: {'early': 0.47, 'mid': 0.38, 'late': 0.15, 'never': 0.0},
            2: {'early': 0.40, 'mid': 0.40, 'late': 0.20, 'never': 0.0},
            3: {'early': 0.40, 'mid': 0.10, 'late': 0.50, 'never': 0.0},
            4: {'early': 0.22, 'mid': 0.34, 'late': 0.22, 'never': 0.22},
        },
    }
    # Simulation parameters
    params_sim = {
        'landscape_iterations': np.arange(20),
        'seed_iterations': np.arange(1),
        'random_seed': 314159,
        'max_time': 10*365,
        'tstep': 1,
        'seed_flag': 3,
        'seed_holding': 1938,
        # 'seed_holding': 41081,
        'seed_county': str(args['county']),
        'num_seeds': 3,
        'county_names_file_path': COUNTY_NAMES,
        # 'binomial_flag': 'right',
        # 'P_CS_size': 'small',
    }
    # Intervention parameters
    params_intervention = {
        'vaccinate_flag': 1,
        'vaccine_effectiveness': 1,
        'vaccination_event_ID': 15,
    }
    # Output parameters
    params_out = {
        'tstep': 1,
        'event_file_path': path + 'output_summary',
        'time_file_path': path + 'output_times',
        'output_flag': 0,
    }

    # Write parameters to a file
    with open(os.path.join(INPUT_DIR, 'parameters.txt'), 'w') as f:
        f.write('# Epidemiological parameters\n')
        for key, value in params_epi.items():
            f.write('%s:%s\n' % (key, value))
        f.write('# Landscape parameters\n')
        for key, value in params_land.items():
            f.write('%s:%s\n' % (key, value))
        f.write('# Behavioural parameters\n')
        for key, value in params_behav.items():
            f.write('%s:%s\n' % (key, value))
        f.write('# Simulation parameters\n')
        for key, value in params_sim.items():
            f.write('%s:%s\n' % (key, value))
        f.write('# Intervention parameters\n')
        for key, value in params_intervention.items():
            f.write('%s:%s\n' % (key, value))
        f.write('# Output parameters\n')
        for key, value in params_out.items():
            f.write('%s:%s\n' % (key, value))
        f.write('# Input parameters\n')
        for key, value in args.items():
            f.write('%s:%s\n' % (key, value))

    return  params_epi, params_land, params_behav, params_sim, params_intervention, params_out