import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import os
from osgeo import gdal
                   
def import_file_lists():
# import_file_lists() function imports lists of file names from the downloaded
# GGCMI Globus archive based on the parameters specified (below).
    
# Import simulated data that uses AgMERRA and Princeton data as climate
# inputs.
    climate_inputs = ['AgMERRA','Princeton']

    
# Crops to be imported: maize, rice, soy, wheat
# The crops have different names in the file (crops) and folder (crops_path) names 
    crops = ['mai','ric','soy','whe']
    crops_path = ['maize', 'rice', 'soy', 'wheat']
    
# The data is imported for the following GGCMs: pDSSAT, EPIC-Boku, EPIC-IIASA,
# GEPIC, pAPSIM,	PEGASUS, LPJ-GUESS, LPJmL, CGMS-WOFOST, EPIC-TAMU, ORCHIDEE-crop,
# PEPIC   
    GGCMs = ['pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
             'lpj-guess','lpjml','cgms-wofost','epic-tamu',\
             'orchidee-crop','pepic']

# The input GGCMI simulation set-up types that are included: default, fullharm,
# harm-suffN (harmnon)
    runtypes = ['default','fullharm','harmnon'] # model setup    

# Fully irrigated (firr) and rainfed (noirr) data is imported
    irrigtypes = ['firr','noirr']
    
# Create a list of all the files that are under the path ''D:\work\data\Globus'
    filenames = []
    for root, dirs, files in os.walk('D:\work\data\Globus'):
        for name in files:
            if name.endswith((".nc4")):
                filenames.append(name)

# Create empty lists of variables that describe the each file in the list
    listings_path = []
    listings_file = []
    crop_list = []
    irrig_list = []
    ggcm_list = []
    runtype_list = []
    climate_list = []
    crop_index = []

# Loop goes through the variables created above and creates lists of file names,
# that correspond to the variable combinations. Also creates lists for the variables
# (crops, irrigation set-ups, etc.) that corresponds to the file name in each position in the list.
    for climate in climate_inputs:
        for crop in crops:
            crop_index = crops.index(crop)
            for GGCM in GGCMs:
                for irrig in irrigtypes:
                    for run in runtypes:
                        for file in filenames:
# Conditions based on which variable names are appended to the lists.
# As Orchidee-crop model has data starting from 1979 and 1980 for AgMERRA climate input, only the one starting 1980 is selected.
# Further, as Orchidee-crop has data starting from 1948 and 1979 for Princeton climate input for only rice 
# (other crops have data start from 1979), the one starting 1979 is selected for rice as well.
# Further, PEPIC has data for an additional 'plim' set-up, which is not included here.
                            if 'yield' in file and crop in file and GGCM in file and run in file and irrig in file and climate.lower() in file and \
                            'plim' not in file and not ('1979' in file and 'agmerra' in file) and not ('1948' in file and 'orchidee-crop' in file):
                                listings_path.append(os.path.join('D:\work\data\Globus',climate,crops_path[crop_index]))
                                listings_file.append(file)
                                crop_list.append(crop)
                                irrig_list.append(irrig)
                                ggcm_list.append(GGCM)
                                runtype_list.append(run)
                                climate_list.append(climate)                    
    
# Function returns a list of files as well as the corresponding path, crop, irrigation set-up, etc.
    return listings_path, listings_file, crop_list, irrig_list, ggcm_list, runtype_list, climate_list
         
   
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

def import_mirca(datatype,resolution):
# Imports MIRCA2000 harvested area data for maize, rice, soy, and wheat,
# respectively, at 0.5 degree resolution. datatype
# specifies whether rainfed ('rfc') or irrigated ('irc') cropland is wanted.
    
    os.chdir('D:\work\data\MIRCA2000\harvested_area_grids')
    
# Create listings variable, that includes all files, that are included under
# the path put as input.
    listings = os.listdir('D:\work\data\MIRCA2000\harvested_area_grids')
    mirca_data = np.zeros((int(180/resolution),int(360/resolution),4))
    
# For 0.5 degree resolution
    if resolution == 0.5:
        resolution = '30'
        # Crop order: maize, rice, soy, wheat
        crop_id = ['02','03','08','01']
        
# Loops through crop_id variable and listings variable (all files in folder)
# and selects the HA data - based on the if statement - and saves the HA data
# into mirca_data variable.
    i = 0
    for crop in crop_id:
        for file in listings:
            if datatype in file and crop in file and resolution in file and file.endswith('.asc'):
                data_temp = gdal.Open(file)                    
                mirca_data[:,:,i] = data_temp.ReadAsArray().astype('f')
                i += 1
                del data_temp
                                
    return mirca_data

def aggregate_raster_ha_prod(ids,map_raster,data_ha,data_prod):
# Aggregates raster level crop production and harvested area data into 
# spatial units defined as map_raster. Each area in map_raster corresponds to
# a unit in the variable ids.
    raster_temp = np.repeat(map_raster[:,:,np.newaxis], data_prod.shape[2], axis=2)

# Create empty variables, where each row represents a unit defined in ids
# and each column represents a year.
    prod = np.zeros((ids.shape[0],data_prod.shape[2]))
    area = np.zeros((ids.shape[0],data_prod.shape[2]))
    
# Loop aggregates harvested areas data and production data for each timestep and 
# spatial unit defined by map_raster (and ids).
    index = -1
    for id in ids:
        index += 1
        prod_temp = np.copy(data_prod)
        area_temp = np.copy(data_ha)
        prod_temp[raster_temp != id] = 0.0
        area_temp[raster_temp != id] = 0.0
        prod[index,:] = np.sum(prod_temp,axis = (0,1))
        area[index,:] = np.sum(area_temp,axis = (0,1))

# Transform all area zero-values in prod and area to np.nan
    prod[area == 0] = np.nan
    area[area == 0] = np.nan
    
    return area, prod

def check_file_existance(file_name,path):
# In case of re-booting, checks which files have already been imported, in order
# to not have to do it again.
    file_list_temp = os.listdir(path)
    for file_temp in file_list_temp:
        if file_temp == file_name:
            exist_or_not = True
            break
        exist_or_not = False
        
    if file_list_temp == []:
        exist_or_not = False
# Returns False, if file doesn't exist under path, and True, if it does.
    return exist_or_not

def import_harvest_shift_info(file_name,season):
    
# Crop harvest dates are different for the four crops (maize, rice, soy, wheat,
# as well as different irrigation set-ups (firr & noirr)
    crop_list = ['mai','ric','soy','whe']
    irrigtypes = ['firr','noirr']
    
# Extract the crop and irrigation set-up for the file specified in file_name.
# Define variables (irrig_gs and crop_gs) that correspond to the names 
# of the growing season files.
    for crop in crop_list:
        for irrig in irrigtypes:
            if crop in file_name.lower() and irrig in file_name.lower():
                if irrig == 'firr':
                    irrig_gs = 'ir'
                elif irrig == 'noirr':
                    irrig_gs = 'rf'

                if crop == 'mai':    
                    crop_gs = 'Maize'
                elif crop == 'ric':
                    crop_gs = 'Rice'
                elif crop == 'soy':
                    crop_gs = 'Soybeans'
                elif crop == 'whe':
                    crop_gs = 'Wheat'
                    
    os.chdir(r'D:\work\data\crop_calendar_isi_mip')

# Import nc harvest and planting day data and change its format to numpy array
    gs_data = nc.Dataset(crop_gs+'_'+irrig_gs+'_growing_season_dates_v1.25.nc4', mode = 'r', format = 'NETCDF4')
    gs_plant = gs_data.variables['planting day'][:][:]
    gs_harvest = gs_data.variables['harvest day'][:][:]
    
# Set strange values (DOY > 365 or DOY < 0) to -999
    data = np.dstack((gs_plant,gs_harvest))
    data = ma.filled(data,-999)
    data[data > 365] = -999
    data[data < 0] = -999
    data = np.flip(data,0)
    
    gs_plant = data[:,:,0]
    gs_harvest = data[:,:,1]
 
# If either harvest or planting data is strange set it to -999 in both
    gs_plant[gs_harvest == -999] = -999
    gs_harvest[gs_plant == -999] = -999
    
# Initialize gs_shift_cell as same shape array as harvest and planting data data
    gs_shift_cell = np.zeros((gs_plant.shape)).astype(bool)

# If temporal aggregation is into annual calendar year values ('annual_harvest'), see where
# planting DOY is larger than harvest DOY (growing season goes over boreal winter).
# If aggregation is into seasonal aggregations (e.g. 'september_aggreg'), change
# strange data values from -999 to 999 as it's more convenient later in the script.
    if season == 'annual_harvest':
        gs_shift_cell[gs_plant > gs_harvest] = True
    else:
        gs_harvest[gs_harvest == -999] = 999
        gs_plant[gs_plant == -999] = 999

# Array of the number of days for each month
    days_per_month = np.array((31,28,31,30,31,30,31,31,30,31,30,31))
    
# If temporal aggregation is conducted so that the crops are harvested annually
# based on a specific month, e.g. starting from september and going to the end
# of august in 'september_aggreg', find where the harvest is conducted prior to the first
# data of the aggregation time span.
    if season == 'december_harvest':
        gs_shift_cell[gs_harvest <= np.sum(days_per_month[:11])] = True
    elif season == 'march_harvest':    
        gs_shift_cell[gs_harvest <= np.sum(days_per_month[:2])] = True
    elif season == 'june_harvest':
        gs_shift_cell[gs_harvest <= np.sum(days_per_month[:5])] = True
    elif season == 'september_harvest':
        gs_shift_cell[gs_harvest <= np.sum(days_per_month[:8])] = True
    
#    plt.imshow(gs_shift_cell)
#    plt.colorbar()
#    plt.show()
    
# If temporal aggregation is conducted so that the crops are sown annually
# based on a specific month, e.g. starting from may and going to the end
# of april in as in 'may_sowing', find where the harvest is conducted prior to the first
# data of the aggregation time span. Only 'may_sowing'
# is used in the study in question.
    if season == 'may_sowing':
        gs_shift_cell[gs_plant <= np.sum(days_per_month[:4])] = True
    elif season == 'june_sowing':
        gs_shift_cell[gs_plant <= np.sum(days_per_month[:5])] = True
    elif season == 'september_sowing':  
        gs_shift_cell[gs_plant <= np.sum(days_per_month[:8])] = True
    elif season == 'december_sowing':  
        gs_shift_cell[gs_plant <= np.sum(days_per_month[:11])] = True
    
    return gs_shift_cell

        
def shift_yield_data(data_yield, file_name, season):
    
    
    if season == 'sowing_annual':
        return data_yield

# Shift data by one year in areas where sowing is conducted starting from a specific date,
# e.g. may.
    elif 'sowing' in season:
        shift_data_seasonal = import_harvest_shift_info(file_name,season)
        data_yield_updated = np.zeros(data_yield.shape)*np.nan
        
        for k in range(0,data_yield.shape[2]-1):
            yield_temp_t0 = np.copy(data_yield[:,:,k])
            yield_temp_t1 = np.copy(data_yield[:,:,k+1])
            
            yield_temp_t0[shift_data_seasonal] = yield_temp_t1[shift_data_seasonal]
            data_yield_updated[:,:,k] = yield_temp_t0
    
        data_yield = data_yield_updated

    else:
# Shift data by one year in areas where harvest date is smaller than planting data (as DOY), so that
# harvests in different areas have the same calendar years. This is done because initially, in GGCMI, the first 
# layer of the data cube shows the yield for the first harvest in each cell. As the first harvest might occur
# in the second year of the simulation (i.e. for crops that grow through winter), the harvests are shifter
# so that they match in terms of calendar years for each raster layer.        
        shift_data_annual = import_harvest_shift_info(file_name,'annual_harvest')
        
        data_yield_updated = np.zeros(data_yield.shape)*np.nan
        for k in range(0,data_yield.shape[2]-1):
            yield_temp_t0 = np.copy(data_yield[:,:,k])
            yield_temp_t1 = np.copy(data_yield[:,:,k+1])
            
            yield_temp_t1[shift_data_annual] = yield_temp_t0[shift_data_annual]
            data_yield_updated[:,:,k+1] = yield_temp_t1
            
        data_yield = data_yield_updated

        
# If harvests are temporally aggregated starting from a specific month
# i.e. september or december for this study.
        if season != 'annual_harvest':
            shift_data_seasonal = import_harvest_shift_info(file_name,season)
            data_yield_updated = np.zeros(data_yield.shape)*np.nan
            for k in range(0,data_yield.shape[2]-1):
                yield_temp_t0 = np.copy(data_yield[:,:,k])
                yield_temp_t1 = np.copy(data_yield[:,:,k+1])
                
                yield_temp_t0[shift_data_seasonal] = yield_temp_t1[shift_data_seasonal]
                data_yield_updated[:,:,k] = yield_temp_t0
    
            data_yield = data_yield_updated
    
    return data_yield




def import_AgMIP_GGCM_data_matched_growing_season_optional_cropland_and_seasonal_aggregation(cropland,season,output_path):
    
    listings_path, listings_file, crop_list, irrig_list, ggcm_list, runtype_list, climate_list \
    = import_file_lists()

# Import fpu raster
    fpu_573 = import_fpu_raster('D:\work\data\map_files\FPUs','raster_fpu_573.tif')
# Extract fpu ids from the fpu raster data
    fpu_573_ids = np.unique(fpu_573)
    fpu_573_ids = fpu_573_ids[~np.isnan(fpu_573_ids)]
    
# import mirca data, unit is in ha;  order: maize, rice, soy, wheat
    mirca_rfc = import_mirca('rfc',0.5) # rainfed areas 
    mirca_irc = import_mirca('irc',0.5) # irrigated areas
#    plt.imshow(mirca_irc[:,:,1])
#    plt.show()

    crops = ['mai','ric','soy','whe']

# loop through each file in listings_file (& listings path)
    for i in range(0,len(listings_path)):

        print(r'AgMIP '+crop_list[i]+' '+str(i))
        print(listings_file[i])

# Check whether file already exists in folder. If file already exists, goes directly to the next item in the loop.
        file_existance = check_file_existance('fpu_573_area_'+crop_list[i]+'_'+climate_list[i]+'_'+irrig_list[i]+'_'+ggcm_list[i]+'_'+runtype_list[i]+'_'+season+'.csv',output_path+'\ha')   
        if file_existance == True:
            continue
        
# import yield data, which is in netcdf format
        os.chdir(listings_path[i])
        data_temp = nc.Dataset(listings_file[i], mode = 'r', format = 'NETCDF4')
        data_yield = data_temp.variables['yield_' + crop_list[i]][:][:][:]
        data_yield = np.rollaxis(np.rollaxis(data_yield,2),2)
        
# fill empty matrix values into nan
        data_yield = ma.filled(data_yield,np.nan)

        data_yield = shift_yield_data(data_yield, listings_file[i], season)


# See which crop is in question; crop_list[i] corresponds to the crop in listings_path[i],
# and retreive the correct harvested areas array.
        j = -1
        for crop in crops:
            j += 1
            if crop in crop_list[i]:
                if cropland == 'full_cropland':
                    mirca_temp = mirca_irc[:,:,j]+mirca_rfc[:,:,j]
                elif cropland == 'actual_cropland':
                    if 'firr' == irrig_list[i]:
                        mirca_temp = mirca_irc[:,:,j]
                    elif 'noirr' in irrig_list[i]:
                        mirca_temp = mirca_rfc[:,:,j]

# Multiply yield with harvested area to obtain production                       
        mirca_temp = np.repeat(mirca_temp[:,:,np.newaxis], data_yield.shape[2], axis=2) 
        rast_prod = data_yield*mirca_temp

# Change nans to 0.0 for faster computation.
        mirca_temp[np.isnan(mirca_temp)] = 0.0
        rast_prod[np.isnan(rast_prod)] = 0.0
# Aggregate to fpu level. 
        fpu_573_area, fpu_573_prod = aggregate_raster_ha_prod(fpu_573_ids,fpu_573,mirca_temp,rast_prod)      

# Calculate yield data
        fpu_573_yield = fpu_573_prod/fpu_573_area

# Add id to FPU level data        
        fpu_573_prod = np.hstack((fpu_573_ids[:,np.newaxis],fpu_573_prod))
        fpu_573_area = np.hstack((fpu_573_ids[:,np.newaxis],fpu_573_area))
        fpu_573_yield = np.hstack((fpu_573_ids[:,np.newaxis],fpu_573_yield))
# Save data in csv format.
        os.chdir(output_path+'\yield')
        np.savetxt('fpu_573_yield_'+crop_list[i]+'_'+climate_list[i]+'_'+irrig_list[i]+'_'+ggcm_list[i]+'_'+runtype_list[i]+'_'+season+'.csv', fpu_573_yield, delimiter=";")

        os.chdir(output_path+'\prod')
        np.savetxt('fpu_573_prod_'+crop_list[i]+'_'+climate_list[i]+'_'+irrig_list[i]+'_'+ggcm_list[i]+'_'+runtype_list[i]+'_'+season+'.csv', fpu_573_prod, delimiter=";")

        os.chdir(output_path+'\ha')
        np.savetxt('fpu_573_area_'+crop_list[i]+'_'+climate_list[i]+'_'+irrig_list[i]+'_'+ggcm_list[i]+'_'+runtype_list[i]+'_'+season+'.csv', fpu_573_area, delimiter=";")
        


# Run the codes with different configurations        
output_path = r'D:\work\data\modified_crop_data\__GGCM_actual_cropland_review1_final'
import_AgMIP_GGCM_data_matched_growing_season_optional_cropland_and_seasonal_aggregation('actual_cropland','may_sowing',output_path)
import_AgMIP_GGCM_data_matched_growing_season_optional_cropland_and_seasonal_aggregation('actual_cropland','annual_harvest',output_path)   

output_path = r'D:\work\data\modified_crop_data\__GGCM_full_cropland_review1_final'
import_AgMIP_GGCM_data_matched_growing_season_optional_cropland_and_seasonal_aggregation('full_cropland','may_sowing',output_path)









