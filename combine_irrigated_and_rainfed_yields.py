import numpy as np
import os

def import_file_list(path,file_extension):
# Imports a list of file names with file_extension ('.csv') from the folder defined by path. 
    listings = []
    for file in os.listdir(path):
        if file.endswith(file_extension):
            listings.append(file)            
    return listings


def import_file_lists_AgMIPcsv(input_prod,input_area,aggreg_type):
# Imports lists of file names for irrigated and rainfed crops to combine.
# input_prod: path to the folder where the production data csvs are stored,
# input_area: path to the folder where the harvested area csvs are stored,
# aggreg_type: FPUs here
    
    climate_inputs = ['AgMERRA','Princeton']
    
# Crops included, order: maize, rice, soy, wheat
    crops = ['mai','ric','soy','whe']
    
# GGCMs included are: pDSSAT, EPIC-Boku, EPIC-IIASA	, GEPIC, pAPSIM, PEGASUS	
# LPJ-GUESS, LPJmL, CGMS-WOFOST, EPIC-TAMU, ORCHIDEE-crop, PEPIC   
    GGCMs = ['pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
             'lpj-guess','lpjml','cgms-wofost','epic-tamu',\
             'orchidee-crop','pepic']
# List of different model configurations.
    runtypes = ['default','fullharm','harmnon'] # model setup
# List of the harvest year set-ups included
    aggreg_season_types = ['annual_harvest','may_sowing']
# 
    filenames_area = import_file_list(input_area,'.csv')
    filenames_prod = import_file_list(input_prod,'.csv')
    
    listings_file_firr_prod = []
    listings_file_noirr_prod = []
    listings_file_firr_area = []
    listings_file_noirr_area = []
# Loop goes through all the variable lists specified above and creates a list of file names.
# The file names are in the same order for the harvested area and production data as well as
# irrigated ('firr') and rainfed ('noirr') data.
    for climate in climate_inputs:
        for crop in crops:
            for GGCM in GGCMs:
                for aggreg in aggreg_type:
                    for aggreg_season in aggreg_season_types:
                        for run in runtypes:
                            os.chdir(input_prod)
                            for file in filenames_area:
                                if 'area' in file and aggreg_season in file and crop in file and GGCM in file and run in file and 'firr' in file and climate in file and aggreg in file:
                                    listings_file_firr_area.append(file)
                                if 'area' in file and aggreg_season in file and crop in file and GGCM in file and run in file and 'noirr' in file and climate in file and aggreg in file:
                                    listings_file_noirr_area.append(file)
                            os.chdir(input_prod)
                            for file in filenames_prod:
                                if 'prod' in file and aggreg_season in file and crop in file and GGCM in file and run in file and 'firr' in file and climate in file and aggreg in file:
                                    listings_file_firr_prod.append(file)
                                if 'prod' in file and aggreg_season in file and crop in file and GGCM in file and run in file and 'noirr' in file and climate in file and aggreg in file:
                                    listings_file_noirr_prod.append(file)

# Function exports lists of file names for irrigated and rainfed production and harvested areas.                                                        
    return listings_file_firr_area, listings_file_noirr_area, listings_file_firr_prod, listings_file_noirr_prod


def combine_AgMIPggcm_yields(input_prod,input_area,output_yield,aggreg_type):
# Function combines irrigated and rainfed yields, as an harvested area weighted average
# and exports it as a csv file.
    
# Export lists of file names for irrigated and rainfed production and harvested areas.                                                         
    listings_file_firr_area, listings_file_noirr_area, listings_file_firr_prod, listings_file_noirr_prod = import_file_lists_AgMIPcsv(input_prod,input_area,aggreg_type)

# Check that the file lists are of same length.
    print(len(listings_file_firr_area))
    print(len(listings_file_noirr_area))
    
# Loops through the file lists exported with the function import_file_lists_AgMIPcsv()
    for firr_area, noirr_area, firr_prod, noirr_prod in zip(listings_file_firr_area, listings_file_noirr_area, listings_file_firr_prod, listings_file_noirr_prod):
        
# If statement checks that the data is in correct order:
        if firr_area == noirr_area.replace('noirr','firr'):       
# Area
# Change working directory to the one with harvested areas data            
            os.chdir(input_area)
# Import the irrigated ('firr') and rainfed ('noirr') harvested areas
# data for each FPU
            data_firr_area = np.genfromtxt(firr_area, delimiter=';')
            data_noirr_area = np.genfromtxt(noirr_area, delimiter=';')
# For summation, change nans into zeros.
            data_firr_area[np.isnan(data_firr_area)] = 0.0
            data_noirr_area[np.isnan(data_noirr_area)] = 0.0
            data_combined_area = data_firr_area + data_noirr_area
# If both irrigated and rainfed data is zero, change back to nan.
            data_combined_area[data_combined_area == 0] = np.nan
# Production
# Change working directory to the one with production data            
            os.chdir(input_prod)
# Import the irrigated ('firr') and rainfed ('noirr') production
# data for each FPU
            data_firr_prod = np.genfromtxt(firr_prod, delimiter=';')
            data_noirr_prod = np.genfromtxt(noirr_prod, delimiter=';')
# For summation, change nans into zeros.
            data_firr_prod[np.isnan(data_firr_prod)] = 0.0
            data_noirr_prod[np.isnan(data_noirr_prod)] = 0.0
            data_combined_prod = data_firr_prod + data_noirr_prod
            
            
# Calcualte combined yield by dividing production with harvested area.
            yield_combined = data_combined_prod[:,1:] / data_combined_area[:,1:]
            yield_combined = np.hstack((data_firr_area[:,0,np.newaxis],yield_combined))
    
# Create file name for the combined data.
            yield_combined_filename = firr_area.replace('area','yield')
            yield_combined_filename = yield_combined_filename.replace('firr','combined')
            
            os.chdir(output_yield)
            np.savetxt(yield_combined_filename,yield_combined, delimiter=";")
        else:
            print('no match')


# Paths to where production and harvested areas (input) data are stored.
input_path_production = r'D:\work\data\modified_crop_data\__GGCM_actual_cropland_review1_final\prod'
input_path_area = 'D:\work\data\modified_crop_data\__GGCM_actual_cropland_review1_final\ha'
# Path to where yield (output) data are stored.
output_path_combined_yield = 'D:\work\data\modified_crop_data\__GGCM_actual_cropland_review1_final\yield'
# Data aggregated to 573 FPUs.
aggreg_type = ['573']
combine_AgMIPggcm_yields(input_path_production,input_path_area,output_path_combined_yield,aggreg_type)

