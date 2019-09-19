# -*- coding: utf-8 -*-
import xarray as xr
import os
import numpy as np
import time
import warnings
import glob
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

   
def import_growing_season(crop,irrig,path,growing_season):
# Create boolean growing season data based on the growing season information used in GGCMI simulations.
    os.chdir(path+'data/crop_calendar_isi_mip')
    growing_season_raw = xr.open_dataset(str(crop)+'_'+irrig+'_growing_season_dates_v1.25.nc4',decode_times = False)
    planting_day = np.flipud(growing_season_raw['planting day'].values-1)
    harvest_day = np.flipud(growing_season_raw['harvest day'].values-1)
    gs_length = np.flipud(growing_season_raw['growing season length'].values)
    
    
    days_per_month = np.array((31,28,31,30,31,30,31,31,30,31,30,31))

    if growing_season == 'june_sowing':
        planting_day[planting_day < np.sum(days_per_month[:5])] = planting_day[planting_day < np.sum(days_per_month[:5])]+365
        harvest_day[planting_day < np.sum(days_per_month[:5])] = harvest_day[planting_day < np.sum(days_per_month[:5])]+365
    elif growing_season == 'may_sowing':
        planting_day[planting_day < np.sum(days_per_month[:4])] = planting_day[planting_day < np.sum(days_per_month[:4])]+365
        harvest_day[planting_day < np.sum(days_per_month[:4])] = harvest_day[planting_day < np.sum(days_per_month[:4])]+365  
        
    else:
        planting_day[planting_day < harvest_day] = planting_day[planting_day < harvest_day]+365
        harvest_day[planting_day < harvest_day] = harvest_day[planting_day < harvest_day]+365
    
    gs_365_365 = np.zeros((365*2,360,720)).astype(bool)          
        
    for lat_i in range(0,planting_day.shape[0]):
        for lon_i in range(0,planting_day.shape[1]):
            planting_day_i = planting_day[lat_i,lon_i]
            harvest_day_i = planting_day[lat_i,lon_i]+gs_length[lat_i,lon_i]
            if np.isnan(planting_day_i) or np.isnan(harvest_day_i):
                continue
            else:
                planting_day_i = planting_day_i.astype(int)
                harvest_day_i = harvest_day_i.astype(int)
                gs_365_365[planting_day_i:harvest_day_i,lat_i,lon_i] = 1
                            
    gs_365_366 = np.vstack((gs_365_365,np.zeros((1,360,720)).astype(bool)))

    
    gs_366_365 = np.zeros((365*2+1,360,720)).astype(bool)          

    planting_day = planting_day + 1
    harvest_day = harvest_day + 1
    
    for lat_i in range(0,planting_day.shape[0]):
        for lon_i in range(0,planting_day.shape[1]):
            planting_day_i = planting_day[lat_i,lon_i]
            harvest_day_i = planting_day[lat_i,lon_i]+gs_length[lat_i,lon_i]
            if np.isnan(planting_day_i) or np.isnan(harvest_day_i):
                continue
            else:
                planting_day_i = planting_day_i.astype(int)
                harvest_day_i = harvest_day_i.astype(int)
                gs_366_365[planting_day_i:harvest_day_i,lat_i,lon_i] = 1
                
    return gs_365_366, gs_366_365, gs_365_365


def write_data_to_netcdf(data,crop,irrig,growing_season):
    
    lats_nc = np.linspace(89.75,-89.75,360)
    lons_nc = np.linspace(-179.75,179.75,720)
    years_nc = np.linspace(1981,2009,29).astype(int)
    
    data_xarray = xr.DataArray(data,
                               dims = ('time','latitude','longitude'),
                               coords={'time': years_nc,
                                       'latitude': lats_nc,
                                       'longitude': lons_nc
                                       })
    
    os.chdir(r'D:\work\data\modified_clim_data\Soil_moisture')
    data_xarray.to_netcdf(crop.lower()+'_'+irrig.lower()+'_'+growing_season.lower()+'_soil_moisture.nc')    

if __name__== "__main__":

    path = r'D:/work/'

    print('Soil moisture: process started')

    # Initialize variables related to temporal span, crops, and irrigation
    years = np.linspace(1981,2009,29).astype(int)
    crops = ['Maize','Rice','Soybeans','Wheat']
    growing_season_list = ['may_sowing']

    irrig_setup = ['ir','rf']
    
    latitude_new = np.linspace(89.75,-89.75,360)
    longitude_new = np.linspace(-179.75,179.75,720)
    
    SM_datatype = 'surf'
    os.chdir(r'D:\work\data\gleam\v3.2a')
    SM_filenames = glob.glob(r'D:\work\data\gleam\v3.2a\*\SM'+SM_datatype+'_*.nc')
    SM_filenames.sort()
#    print(SM_filenames)

    SM_data = xr.open_mfdataset(SM_filenames).interp(lon=longitude_new, lat=latitude_new, method = 'linear')['SM'+SM_datatype].transpose('time','lat','lon')
#    print(SM_data)
#    sys.stdout.flush()
    SM_data.sel(time = '2009-01-01').plot()
    # Set up a distributed computing platform
    year = 1990
    crop = 'Maize'
    irrig = 'rf'
    day = 10
    for growing_season in growing_season_list:
        for crop in crops:
            for irrig in irrig_setup:
                gs_365_366, gs_366_365, gs_365_365 = import_growing_season(crop,irrig,path,growing_season)
                sm_data = np.zeros((years.shape[0],latitude_new.shape[0],longitude_new.shape[0]))
                
                for year in years:
                    start_t_yr = time.time()
                                        
                    if 'sowing' in growing_season:
                        sm_yrly = SM_data.sel(time=slice(str(year)+'-01-01', str(year+1)+'-12-31')).values
                        year_length_t1 = SM_data['time'].sel(time = str(year)).shape[0]
                        year_length_t2 = SM_data['time'].sel(time = str(year+1)).shape[0]
                    else:
                        sm_yrly = SM_data.sel(time=slice(str(year-1)+'-01-01', str(year)+'-12-31')).values
                        year_length_t1 = SM_data['time'].sel(time = str(year-1)).shape[0]
                        year_length_t2 = SM_data['time'].sel(time = str(year)).shape[0]
                    
                    if year_length_t1 == 366:
                        sm_yrly[~gs_366_365] = np.nan
                    elif year_length_t2 == 366:
                        sm_yrly[~gs_365_366] = np.nan
                    else:
                        sm_yrly[~gs_365_365] = np.nan
                    
                    sm_yrly_avg = np.nanmean(sm_yrly,axis = 0)
        
                    sm_data[year-1981,:,:] = sm_yrly_avg               
                    print('year '+str(year)+' took '+str(time.time()-start_t_yr)+' to run')          

        
                sm_data_anom = (sm_data - np.mean(sm_data,axis=0)) / np.std(sm_data,axis = 0)
                write_data_to_netcdf(sm_data_anom,crop,irrig,growing_season)
        
        
        
        
        
        