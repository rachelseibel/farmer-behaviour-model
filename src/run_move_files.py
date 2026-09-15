import os
import pandas as pd

def move_files():
    path = '/home/rachelseibel/farmer_behaviour/'
    ## Organise data into folders
    path_no_vax = path + 'simulations_no_vax/'
    path_idw = path + 'simulations_vax_idw_all/'
    path_softmax = path + 'simulations_vax_softmax_all/'
    county_names_path = path + '/scripts/inputs/holdings_counties.csv'
    landscape_method_folders = ['GB', 'Nation', 'County', 'blocking_method', 'interpolation_method']
    county_names = pd.read_csv(county_names_path)
    county_folders = county_names['county'].unique()
    # Replace spaces with underscores in county names
    county_folders = [county.replace(' ', '_') for county in county_folders]

    # Create folders for each landscape method
    for method in landscape_method_folders:
        if method == 'GB':
            os.makedirs(os.path.join(path_no_vax, method), exist_ok=True)
        os.makedirs(os.path.join(path_idw, method), exist_ok=True)
        os.makedirs(os.path.join(path_softmax, method), exist_ok=True)
        # Create folders for each county
        for county in county_folders:
            if method == 'GB':
                os.makedirs(os.path.join(path_no_vax+method+'/', county), exist_ok=True)
            os.makedirs(os.path.join(path_idw+method+'/', county), exist_ok=True)
            os.makedirs(os.path.join(path_softmax+method+'/', county), exist_ok=True)
    # Move files to the appropriate folders
    for method in landscape_method_folders:
        for county in county_folders:
            # Move files for no_vax
            if method == 'GB':
                for file in os.listdir(path_no_vax):
                    if 'landscapeMethod_'+method in file and county in file:
                        os.rename(os.path.join(path_no_vax, file), os.path.join(path_no_vax+method+'/', county, file))
            # Move files for idw
            for file in os.listdir(path_idw):
                if 'landscapeMethod_'+method in file and county in file:
                    os.rename(os.path.join(path_idw, file), os.path.join(path_idw+method+'/', county, file))
            # Move files for softmax
            for file in os.listdir(path_softmax):
                if 'landscapeMethod_'+method in file and county in file:
                    os.rename(os.path.join(path_softmax, file), os.path.join(path_softmax+method+'/', county, file))
    return

if __name__ == "__main__":
    print("Moving files...")
    move_files()
    print("Files moved successfully.")