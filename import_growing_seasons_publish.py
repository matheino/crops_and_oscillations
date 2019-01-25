import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import os
from osgeo import gdal

def import_fpu_raster(directory, filename):
# Import raster data from path (directory) and filename
    os.chdir(directory)
    fpu_raster = gdal.Open(filename)
    fpu_array = fpu_raster.GetRasterBand(1)
    fpu_array = fpu_array.ReadAsArray().astype('f')
# Change all values below or equal to zero into nan
    fpu_array[fpu_array <= 0] = np.nan
    return fpu_array

def import_mirca_mask(datatype,resolution,filename):
# Imports MIRCA2000 harvested area data for maize, rice, soy, and wheat,
# respectively.
    os.chdir(r'D:\work\data\MIRCA2000\harvested_area_grids')
    
    listings = os.listdir('D:\work\data\MIRCA2000\harvested_area_grids')
# Crop order: maize, rice, soy, wheat
    mirca_data = np.zeros((int(180/resolution),int(360/resolution),4))
# Change information of resolution to correspond to MIRCA data
    if resolution == 0.5:
        resolution = '30'
        crop_id = ['02','03','08','01']
        
# Loops through crop_id variable and listings variable (all files in folder)
# and selects the HA data - based on the if statement - and saves the HA data
# into mirca_data variable.
    i = 0
    for crop in crop_id:
        for file in listings:
            if datatype in file and crop in file and resolution in file and (file.endswith('.asc') or file.endswith('.tif')):
                data_temp = gdal.Open(file)                    
                mirca_data[:,:,i] = data_temp.ReadAsArray().astype('f')
                i += 1
                del data_temp
                
    crop_list = ['mai','ric','soy','wheat']

# Create a boolean array about the growing areas for each crop
    for i in range(0,len(crop_list)):
        if crop_list[i] in filename.lower():
            mirca_data_out = mirca_data[:,:,i] > 0

    return mirca_data_out

def aggregate_growing_season(ids,map_raster,harvest,mirca):
# Create empty variables, where each row represents a unit defined in ids
# and each column represents a year.
    map_raster = np.repeat(map_raster[:,:,np.newaxis],2,axis=2)
    gs_table_boolean = np.zeros((ids.shape[0],5)).astype(bool)    
    
# Loop through each FPU id
    index = -1
    for id in ids:
        index = index+1
        harvest_temp = np.copy(harvest)
        harvest_temp[map_raster != id] = np.nan
        gs_area_temp = np.zeros((5,2))

# Loop through the irrigation set-ups
        for irrig in range(0,2):
# Loop through the harvest seasons
            for season in range(0,5):
# Create a boolean array, of the areas, where crops are harvested in the season in question
                harvest_season_temp = (harvest_temp[:,:,irrig] == season).astype(float)
# Mask the harvested areas data with the boolean array created above, and sum it into a single value
                harvest_temp_area = harvest_season_temp*mirca[:,:,irrig]
                harvest_temp_tot_area = np.nansum(harvest_temp_area)
# Save the extent of harvested areas on the season in question into gs_area_temp
                gs_area_temp[season,irrig] = harvest_temp_tot_area

# Sum irrigated and rainfed harvested areas for each season
        gs_area_temp = np.sum(gs_area_temp,axis = 1)
# If harvested areas exist for the FPU in question, check which season has the largest harvested area,
# and change the boolean value corresponding that season to True in gs_table_boolean
        if np.nansum(gs_area_temp) > 0:
            gs_temp = np.argmax(gs_area_temp).astype(int)
            gs_table_boolean[index,gs_temp] = True
        
# Combine id data with growing season data   
    growing_season = np.hstack((ids[:,np.newaxis],gs_table_boolean))
    
    return growing_season


def import_growing_seasons():

# Import file list from the folder that contains the growing season information        
    file_list = os.listdir(r'D:\work\data\crop_calendar_isi_mip')
    
    crop_list = ['mai','ric','soy','whe']
# Create an array with the number of days for each month.
    days_per_month = np.array((31,28,31,30,31,30,31,31,30,31,30,31))
# Import FPU information
    fpu_573 = import_fpu_raster('D:\work\data\map_files\FPUs','raster_fpu_573.tif')
    fpu_573_ids = np.unique(fpu_573)
    fpu_573_ids = fpu_573_ids[~np.isnan(fpu_573_ids)]

    for crop in crop_list:
        for file in file_list:
            if crop in file.lower() and 'ir' in file.lower() and file.endswith('nc4'):
                os.chdir(r'D:\work\data\crop_calendar_isi_mip')
                file_ir = file
                file_rf = file.replace('ir','rf')
            
# Import the nc-file and extract the variables.
                data_ir = nc.Dataset(file_ir, mode = 'r', format = 'NETCDF4')
                data_rf = nc.Dataset(file_rf, mode = 'r', format = 'NETCDF4')
                plant_ir = data_ir.variables['planting day'][:][:]
                harvest_ir = data_ir.variables['harvest day'][:][:]
                plant_rf = data_rf.variables['planting day'][:][:]
                harvest_rf = data_rf.variables['harvest day'][:][:]

# Make sure there isn't any strange values in the data           
                data = np.dstack((plant_ir,harvest_ir,plant_rf,harvest_rf))
                data = ma.filled(data,np.nan)
                data[data > 365] = -99
                data[data < 0] = np.nan
                data = np.flip(data,0)
# Here only harvest information is used, thus select that data to the harvest variable.
# Also, change nan values to -99 to make conditional operations without warning messages
                harvest = data[:,:,[1,3]]
                harvest[np.isnan(harvest)] = -99
                
# Import the harvested area of irrigated and rainfed crops and combine those into a single variable
                mirca_irc = import_mirca_mask('irc',0.5,file_ir) # irrigated areas
                mirca_rfc = import_mirca_mask('rfc',0.5,file_rf) # rainfed areas
                mirca_stack = np.dstack((mirca_irc,mirca_rfc))                
                
                harvest_season = np.zeros(harvest.shape)*np.nan
                
                print np.nanmax(harvest)

# Loops through both irrigation set-ups and 5 harvest seasons, and checks which     
# which season the harvest time falls in. The five seasons included are:
# i) January, February; ii) March, April, May; iii) June, July, August;
# iv) September, October, November; v) December. These seasons were selected,
# correspond to the times for which the seasonal oscillation indices were calculated for.
                for j in range(0,2):
                    for i in range(0,5):
                        if i == 0:
                            season_start = 1
                            season_end = np.sum(days_per_month[0:2])
                        elif i > 0 and i < 4:
                            season_start = np.sum(days_per_month[0:(2+(i-1)*3)])+1
                            season_end = np.sum(days_per_month[0:(2+(i)*3)])
                        elif i == 4:
                            season_start = np.sum(days_per_month[:-1])+1
                            season_end = 365.0
                            
# Isolete the areas, where the crop in question is harvested during the interval in question
                        harvest_index = np.all(np.dstack((harvest[:,:,j] >= season_start, harvest[:,:,j] <= season_end)),axis = 2)  
                        harvest_season[harvest_index,j] = i
                
                plt.imshow(harvest_season[:,:,1])
                plt.colorbar()
                plt.show()
# Aggregate the seasonal harvest season to FPU level, by investigating which season has the
# largest harvested area in the FPU in question.
                growing_season = aggregate_growing_season(fpu_573_ids,fpu_573,harvest_season,mirca_stack)
                os.chdir(r'D:\work\data\crop_calendar_isi_mip')
# Save the tabulated FPU level harvest season data.
                np.savetxt(crop+'_growing_season.csv',growing_season,delimiter = ';')
                                       
import_growing_seasons()
             
                        

