import xarray as xr
import os
import numpy as np
import pandas as pd
import time
import warnings
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


def write_data_to_netcdf(data,crop,irrig,growint_season,path):
    
    lats_nc = np.linspace(89.75,-89.75,360)
    lons_nc = np.linspace(-179.75,179.75,720)
    years_nc = np.linspace(1981,2009,29).astype(int)
    
    data_xarray = xr.DataArray(data,
                               dims = ('time','latitude','longitude'),
                               coords={'time': years_nc,
                                       'latitude': lats_nc,
                                       'longitude': lons_nc
                                       })
    
    os.chdir(path+'/data/modified_clim_data/Temperature')
    data_xarray.to_netcdf(crop.lower()+'_'+irrig.lower()+'_'+growing_season.lower()+'_GDD.nc')    

if __name__== "__main__":
    
    print('Temperature: process started')
    # Initialize variables related to temporal span, crops, and irrigation
    
    years = np.linspace(1981,2009,29).astype(int)
    crops = ['Maize','Rice','Soybeans','Wheat']
    growing_season_list = ['may_sowing']
    
    irrig_setup = ['ir','rf']
    
    path = r'D:/work/'
    os.chdir(path+'data/agmerra')
    latitude_new = np.linspace(89.75,-89.75,360)
    longitude_new = np.linspace(0.25,359.75,720)

    T_avg = xr.open_mfdataset('*_tavg.nc4',decode_times = False).interp(longitude=longitude_new, latitude=latitude_new, method = 'linear')['tavg']#.roll(longitude=360, roll_coords = True)
    
    T_avg['time'] = pd.to_datetime(T_avg['time'].values, unit = 'D',origin = pd.Timestamp('1980-1-1'))
    
    T_avg = T_avg.roll(longitude=360, roll_coords = True)
    T_avg = T_avg.assign_coords(longitude=((T_avg.longitude.values + 180) % 360) - 180)
    
#    T_avg.sel(time = '1990-01-01',method = 'nearest').plot()
        
    
    year = 1990
    crop = 'Maize'
    irrig = 'rf'
    day = 10
    for growing_season in growing_season_list:
        for crop in crops:
            for irrig in irrig_setup:
                gs_365_366, gs_366_365, gs_365_365 = import_growing_season(crop,irrig,path,growing_season)
                GDD_data = np.zeros((years.shape[0],latitude_new.shape[0],longitude_new.shape[0]))
    
                for year in years:
                    start_t_yr = time.time()
                                        
                    if 'sowing' in growing_season:
                        T_avg_yrly = T_avg.sel(time=slice(str(year)+'-01-01', str(year+1)+'-12-31')).values
                        year_length_t1 = T_avg['time'].sel(time = str(year)).shape[0]
                        year_length_t2 = T_avg['time'].sel(time = str(year+1)).shape[0]
                    else:
                        T_avg_yrly = T_avg.sel(time=slice(str(year-1)+'-01-01', str(year)+'-12-31')).values
                        year_length_t1 = T_avg['time'].sel(time = str(year-1)).shape[0]
                        year_length_t2 = T_avg['time'].sel(time = str(year)).shape[0]

                    if year_length_t1 == 366:
                        T_avg_yrly[~gs_366_365] = np.nan
                    elif year_length_t2 == 366:
                        T_avg_yrly[~gs_365_366] = np.nan
                    else:
                        T_avg_yrly[~gs_365_365] = np.nan
                        
                    GDD_yrly = np.nansum(T_avg_yrly,axis = 0)
    
                    GDD_data[year-1981,:,:] = GDD_yrly               
                    print('year '+str(year)+' took '+str(time.time()-start_t_yr)+' s to run')
                    
                GDD_data_anom = (GDD_data - np.mean(GDD_data,axis=0)) / np.std(GDD_data,axis = 0)
                write_data_to_netcdf(GDD_data_anom,crop,irrig,growing_season,path)
            
            

            
                
            
            
            
            
            
            
            
            
            
    