import xarray as xr
import numpy as np
import os
from osgeo import gdal
                   

def import_fpu_raster(directory, filename):
# Import FPU raster data from path (directory) and filename
    os.chdir(directory)
    fpu_raster = gdal.Open(filename)
    fpu_array = fpu_raster.GetRasterBand(1)
    fpu_array = fpu_array.ReadAsArray().astype('f')
# Change all values below or equal to zero into nan
    fpu_array[fpu_array <= 0] = np.nan
# Returns a numpy array of the FPUs
    return fpu_array

def import_mirca_dict(datatype,resolution):
# Imports MIRCA2000 harvested area data for maize, rice, soy, and wheat,
# respectively, at 0.5 degree resolution. datatype
# specifies whether rainfed ('rfc') or irrigated ('irc') cropland is wanted.
    
    os.chdir('D:\work\data\MIRCA2000\harvested_area_grids')
    
# Create listings variable, that includes all files, that are included under
# the path put as input.
    listings = os.listdir('D:\work\data\MIRCA2000\harvested_area_grids')
    
# For 0.5 degree resolution
    if resolution == 0.5:
        resolution = '30'
        # Crop order: maize, rice, soy, wheat
        crop_ids = ['02','03','08','01']
        crops = ['maize', 'rice', 'soybeans', 'wheat']
        
# Loops through crop_id variable and listings variable (all files in folder)
# and selects the HA data - based on the if statement - and saves the HA data
# into mirca_data variable.
    i = 0
    mirca_data = {}
    for crop_id, crop in zip(crop_ids, crops):
        for file in listings:
            if datatype in file and crop_id in file and resolution in file and file.endswith('.asc'):
                data_temp = gdal.Open(file)                    
                mirca_data_temp = data_temp.ReadAsArray().astype('f')
                mirca_data.update({crop:mirca_data_temp})
                
                i += 1
                del data_temp
                                
    return mirca_data

def aggregate_clim(ids,map_raster,data_ha,data_prod):
# Aggregates raster level weather anomaly data into 
# spatial units defined as map_raster. Each area in map_raster corresponds to
# a unit in the variable ids.
    raster_temp = np.repeat(map_raster[np.newaxis,:,:], data_prod.shape[0], axis=0)

# Create empty variables, where each row represents a unit defined in ids
# and each column represents a year.
    prod = np.zeros((ids.shape[0],data_prod.shape[0]))
    area = np.zeros((ids.shape[0],data_prod.shape[0]))
    
# Loop aggregates harvested areas data and weighed anomaly data data for each timestep and 
# spatial unit defined by map_raster (and ids).
    index = -1
    for id in ids:
        index += 1
        prod_temp = np.copy(data_prod)
        area_temp = np.copy(data_ha)
        prod_temp[raster_temp != id] = 0.0
        area_temp[raster_temp != id] = 0.0
        prod[index,:] = np.sum(prod_temp,axis = (1,2))
        area[index,:] = np.sum(area_temp,axis = (1,2))

# Transform all area zero-values in prod and area to np.nan
    prod[area == 0] = np.nan
    area[area == 0] = np.nan
    
    return area, prod


def aggregate_climate_data(output_path, growing_season):

# Import fpu raster
    fpu_573 = import_fpu_raster('D:\work\data\map_files\FPUs','raster_fpu_573.tif')
# Extract fpu ids from the fpu raster data
    fpu_573_ids = np.unique(fpu_573)
    fpu_573_ids = fpu_573_ids[~np.isnan(fpu_573_ids)]
    
# import mirca data, unit is in ha;  order: maize, rice, soy, wheat
    mirca_rfc_all_crops = import_mirca_dict('rfc',0.5) # rainfed areas 
    mirca_irc_all_crops = import_mirca_dict('irc',0.5) # irrigated areas
    
    
#    plt.imshow(mirca_irc[:,:,1])
#    plt.show()

    crops = ['maize','rice','soybeans','wheat']

    path = r'D:\work\data\modified_clim_data'

    for crop in crops:
# Temperature anomaly and soil moisture data:
        GDD_ir = xr.open_dataset(os.path.join(path,'Temperature',crop+'_ir_'+growing_season+'_GDD.nc'))['__xarray_dataarray_variable__'].fillna(0).values
        GDD_rf = xr.open_dataset(os.path.join(path,'Temperature',crop+'_rf_'+growing_season+'_GDD.nc'))['__xarray_dataarray_variable__'].fillna(0).values
        soil_moisture_ir = xr.open_dataset(os.path.join(path,'Soil_moisture',crop+'_ir_'+growing_season+'_soil_moisture.nc'))['__xarray_dataarray_variable__'].fillna(0).values
        soil_moisture_rf = xr.open_dataset(os.path.join(path,'Soil_moisture',crop+'_rf_'+growing_season+'_soil_moisture.nc'))['__xarray_dataarray_variable__'].fillna(0).values
        
        mirca_rfc = mirca_rfc_all_crops[crop]
        mirca_irc = mirca_irc_all_crops[crop]
        
        mirca_rfc = np.repeat(mirca_rfc[np.newaxis,:,:], GDD_ir.shape[0], axis = 0,)
        mirca_irc = np.repeat(mirca_irc[np.newaxis,:,:], GDD_ir.shape[0], axis = 0,)
        
        GDD_ir_areaweight = GDD_ir * mirca_irc
        GDD_rf_areaweight = GDD_rf * mirca_rfc
        GDD_areaweight = GDD_ir_areaweight + GDD_rf_areaweight
        soil_moisture_ir_areaweight = soil_moisture_ir * mirca_irc
        soil_moisture_rf_areaweight = soil_moisture_rf * mirca_rfc
        soil_moisture_areaweight = soil_moisture_ir_areaweight + soil_moisture_rf_areaweight
        
        mirca_tot = mirca_irc + mirca_rfc
# Aggregate to fpu level. 
        fpu_573_area, fpu_573_GDD_areaweight = aggregate_clim(fpu_573_ids,fpu_573,mirca_tot,GDD_areaweight)      
        fpu_573_area, fpu_573_soil_moisture_areaweight = aggregate_clim(fpu_573_ids,fpu_573,mirca_tot,soil_moisture_areaweight)
# Calculate yield data
        fpu_573_GDD = fpu_573_GDD_areaweight/fpu_573_area
        fpu_573_soil_moisture = fpu_573_soil_moisture_areaweight/fpu_573_area
# Add id to FPU level data        
        fpu_573_GDD = np.hstack((fpu_573_ids[:,np.newaxis],fpu_573_GDD))
        fpu_573_soil_moisture = np.hstack((fpu_573_ids[:,np.newaxis],fpu_573_soil_moisture))
# Save data in csv format.
        os.chdir(output_path)
        np.savetxt('fpu_573_temperature_'+crop+'.csv', fpu_573_GDD, delimiter=";")
        np.savetxt('fpu_573_soil_moisture_'+crop+'.csv', fpu_573_soil_moisture, delimiter=";")

# Run the codes with different configurations        
output_path = r'D:\work\data\modified_clim_data'
growing_season = 'may_sowing'

aggregate_climate_data(output_path, growing_season)






