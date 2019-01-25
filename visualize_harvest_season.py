#!/usr/bin/env python
# -*- coding: utf-8 -*- import pylab
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap


def fpu_data_to_raster(fpu_ids,fpu_raster,data):
# Disaggregate tabulated data into raster.
    path = r'D:\work\data\map_files\FPUs'
    os.chdir(path)
    
    fpu_raster = gdal.Open(fpu_raster)
    fpu_array = fpu_raster.GetRasterBand(1)
    fpu_array = fpu_array.ReadAsArray()
    fpu_array = fpu_array.astype('f')
    data_array = np.zeros((fpu_array.shape[0],fpu_array.shape[1]))
    
    i = -1
    for fpu in fpu_ids:
        i += 1
        data_array[fpu_array == fpu] = data[i]
        
    return data_array

def import_mirca_data_visualization_masking(file_name):
# Import harvested areas data, based on the irrigation setup and crop
# specified in the file name
    os.chdir(r'D:\work\data\MIRCA2000\harvested_area_grids')
    if 'mai' in file_name:
        data_temp_firr = gdal.Open('annual_area_harvested_irc_crop02_ha_30mn.asc')
        data_temp_noirr = gdal.Open('annual_area_harvested_rfc_crop02_ha_30mn.asc')
    elif 'ric' in file_name:
        data_temp_firr = gdal.Open('annual_area_harvested_irc_crop03_ha_30mn.asc')
        data_temp_noirr = gdal.Open('annual_area_harvested_rfc_crop03_ha_30mn.asc')
    elif 'soy' in file_name:
        data_temp_firr = gdal.Open('annual_area_harvested_irc_crop08_ha_30mn.asc')
        data_temp_noirr = gdal.Open('annual_area_harvested_rfc_crop08_ha_30mn.asc')
    elif 'whe' in file_name:
        data_temp_firr = gdal.Open('annual_area_harvested_irc_crop01_ha_30mn.asc')
        data_temp_noirr = gdal.Open('annual_area_harvested_rfc_crop01_ha_30mn.asc')
    
    mirca_firr = data_temp_firr.ReadAsArray().astype('f')
    mirca_noirr = data_temp_noirr.ReadAsArray().astype('f')
    
    if 'firr' in file_name:
        mirca_data = mirca_firr + mirca_noirr
    elif 'noirr' in file_name:
        mirca_data = mirca_firr + mirca_noirr
    elif 'combined' in file_name or 'est' in file_name:
        mirca_data = mirca_firr + mirca_noirr
        
    return mirca_data


def visualize_raster(crop,osc,raster_data,raster_mask,savepath,save,clim,data_type,title):

# Initialize the map figure using Basemap
    m = Basemap(projection='cyl', resolution='c',
        llcrnrlat=-60, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180)
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)        

# Create the colorscale        
    if data_type == 'harvest_season':
        col0 = np.array([0.9,0.9,0.9])
        col1 = '#7aafcc'
        col2 = '#dae123'
        col3 = '#7cb542'
        col4 = '#c65c3c'
        col5 = '#9d65aa'
        
        colors = [col0,col1,col2,col3,col4,col5]
        
        cbar_label = 'Robustness of results'
        cmap = ListedColormap(colors)
    
# Modify settings for the raster visualization
    raster_data = np.flipud(raster_data)
    raster_mask = np.flipud(raster_mask)
    mask_index = raster_mask * np.isnan(raster_data)
    raster_data[mask_index] = 0
    cs = m.imshow(raster_data[60:,:],clim=clim,cmap=cmap)   
    m.drawcoastlines(linewidth=0.25)
#    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='white',lake_color='white',zorder = 0)
    m.drawmapboundary(fill_color='White')
    
# Save the figure
    if save == 1:
        os.chdir(savepath)
        plt.savefig(crop+'_'+data_type+'.png', dpi=300, bbox_inches='tight')
    plt.show(m)

# If colorbar for sensitivity doesn't exist in folder, create it.
    if data_type not in os.listdir(savepath):
        plt.figure()
        cs = m.imshow(raster_data[60:,:],clim=clim,cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(cs, extend = 'both',orientation='horizontal')
        cbar.set_label(cbar_label, fontsize=12)
        plt.savefig(data_type, dpi = 300, bbox_inches='tight')
        

model = 'all_models'
aggreg = '573'
irrig= 'combined' # ['noirr','combined']
climate = 'AgMERRA'
crop_list = ['mai','ric','soy','whe']
oscillation_list = ['enso','iod','nao']
savepath = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity_gs_figs'
inputpath = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity_gs_figs'
os.chdir(inputpath)

for crop in crop_list:
# Import data about harvest seasons, and extract the FPU ids and season information into arrays
        season_raw = np.genfromtxt(r'sensitivity_gs_seasons_enso_'+crop+'_combined_fullharm_all_models_AgMERRA_GGCM_573.csv',delimiter = ';')
        ids = season_raw[:,0]
        season = season_raw[:,1]+1
# Disaggregate tabulated data into raster
        season_raster = fpu_data_to_raster(ids,'raster_fpu_573.tif',season)
# Import mirca cropland mask and set values where the crop in question is not grown as nan.
        mirca_mask = import_mirca_data_visualization_masking(crop+'combined')        
        season_raster[mirca_mask == 0] = np.nan
#        plt.imshow(season_raster)
#        plt.show()        
        mirca_mask = mirca_mask > 0
# Visualize and save the raster data        
        visualize_raster(crop,oscillation,season_raster,mirca_mask,savepath,1,(0,5),'harvest_season',crop+oscillation)
        