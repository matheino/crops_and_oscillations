#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.colors import LinearSegmentedColormap
os.environ['PROJ_LIB'] = r'C:\Users\heinom2\AppData\Local\conda\conda\pkgs\proj4-4.9.3-hfa6e2cd_8\Library\share'
from mpl_toolkits.basemap import Basemap



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

def import_mirca_data_visualization_masking_combined(file_name):
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

    mirca_data = mirca_firr + mirca_noirr
        
    return mirca_data

def obtain_raster_data(file_name, path, aggregation):
# Import tablulated data, based on file_name and path to the directory in question.
    os.chdir(path)
    print(file_name)
    data = np.genfromtxt(file_name,delimiter = ';')    
    
    if aggregation == '573':
        ids = data[:,0]
        sens = data[:,1]
        pval = data[:,2]   
        rast_sens = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens)
        rast_pval = fpu_data_to_raster(ids,'raster_fpu_573.tif',pval)
        
    mirca_data = import_mirca_data_visualization_masking_combined(file_name)
     
    rast_sens[mirca_data == 0] = np.nan
    rast_pval[mirca_data == 0] = 1
    
    mirca_data = mirca_data > 0
    
    return rast_sens, rast_pval, mirca_data


def extract_raster_sensitivity(alpha,file_name,aggregation,clim_type,input_path,savepath,save):

# Import information of sensitivity as raster
    rast_sens, rast_pval, rast_ha_bol = obtain_raster_data(file_name,input_path,aggregation)
    rast_sens[rast_pval > alpha] = np.nan
    
# Initialize the map figure using Basemap
    m = Basemap(projection='cyl', resolution='c',
        llcrnrlat=-60, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180, )
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


# Create the colorscale, using RGB information and the function LinearSegmentedColormap
    
    if clim_type == 'Soil_moisture':
    
        cdict = {'blue':  ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.49, 0.8, 0.9),
                       (0.51, 0.9, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 0.4, 1.0)),
    
             'green': ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.49, 0.9, 0.9),
                       (0.51, 0.9, 0.9),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'red':  ((0.0, 0.0, 0.4),
                       (0.25, 1.0, 1.0),
                       (0.49, 1.0, 0.9),
                       (0.51, 0.9, 0.8),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
                }
        
    elif clim_type == 'Temperature':
        
        cdict = {'blue': ((0.0, 0.0, 0.4),
                       (0.25, 1.0, 1.0),
                       (0.49, 1.0, 0.9),
                       (0.51, 0.9, 0.8),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'green': ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.49, 0.9, 0.9),
                       (0.51, 0.9, 0.9),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'red':  ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.49, 0.8, 0.9),
                       (0.51, 0.9, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 0.4, 1.0))
                }
        
        
        
        
    cmap = LinearSegmentedColormap('cmap', cdict)
    clim = (-1,1)
# Modify settings for the raster visualization
    rast_sens = np.flipud(rast_sens)
    rast_ha_bol = np.flipud(rast_ha_bol)
    crop_index = rast_ha_bol * np.isnan(rast_sens)
    rast_sens[crop_index] = 0

    m.imshow(rast_sens[60:,:],clim=clim,cmap=cmap)    
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='white',lake_color='white',zorder = 0)
    m.drawmapboundary(fill_color='White')
    
#    title, model_title = create_title(file_name)
#    plt.title(title, loc='left',fontsize=20)
#    plt.title(model_title, loc='left',fontsize=20)

# Save the figure
    if save == 1:
        os.chdir(savepath)
        plt.savefig(file_name.replace('.csv','.png'), dpi=300, bbox_inches='tight')
    plt.show(m)
    print(file_name)

# If colorbar for sensitivity doesn't exist in folder, create it.
    if 'colorbar_sens_T.png' not in os.listdir(savepath) and clim_type == 'Temperature':
        plt.figure()
        img = plt.imshow(rast_sens[60:,:],clim=clim,cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(img, extend = 'both',orientation='horizontal')
        cbar.set_label('Tempearature anomaly per standard deviation index change', fontsize=12)
        plt.savefig('colorbar_sens_T.png', dpi = 300, bbox_inches='tight')
        plt.show()
        
    elif 'colorbar_sens_SM.png' not in os.listdir(savepath) and clim_type == 'Soil_moisture':
        plt.figure()
        img = plt.imshow(rast_sens[60:,:],clim=clim,cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(img, extend = 'both',orientation='horizontal')
        cbar.set_label('Soil moisture anomaly per standard deviation index change', fontsize=12)
        plt.savefig('colorbar_sens_SM.png', dpi = 300, bbox_inches='tight')
        plt.show()
        
    

def sens_files_to_visualize(alpha,crop_list,oscillation_list,climate_list,clim_type_list, aggregation,input_path,savepath,save):
    
    file_list = os.listdir(input_path)
# Loop throught the paramters and selects the file that is going to be visualized.
    for climate in climate_list:
        for crop in crop_list:
            for osc in oscillation_list:
                for clim_type in clim_type_list:
                    for file in file_list:
                        if crop in file and osc in file and climate in file and clim_type in file and file.endswith('.csv'):
                            extract_raster_sensitivity(alpha,file,aggregation[0],clim_type,input_path,savepath,save)


aggregation_list = ['573']
irrig_setup_list = ['combined']
climate_list = ['AgMERRA']
crop_list = ['mai','whe','ric','soy']
#oscillation_list = ['enso_djf_adj','iod_son_partial','nao_djf_adj']#,'enso_mam','enso_jja','enso_son','enso_djf_plus','max_enso','iod_djf_reg','iod_mam','iod_jja','iod_son','iod_djf_plus','max_iod','nao_djf_reg','nao_mam','nao_jja','nao_son','nao_djf_plus','max_nao']
oscillation_list = ['enso_multiv','iod_multiv','nao_multiv']

clim_type_list = ['Temperature','Soil_moisture']

alpha = 0.1
save = 1

input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\clim_corr'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\clim_corr\figures'

sens_files_to_visualize(alpha,crop_list,oscillation_list,climate_list, clim_type_list, aggregation_list, input_path,savepath,save)






















    