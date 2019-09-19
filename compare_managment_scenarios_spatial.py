#!/usr/bin/env python
# -*- coding: utf-8 -*- import pylab
import os
os.environ['PROJ_LIB'] = r'C:\Users\heinom2\AppData\Local\conda\conda\pkgs\proj4-4.9.3-hfa6e2cd_8\Library\share'
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap



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

def obtain_raster_data_difference(input_path1,input_path2,aggregation,alpha,file1,file2):
    
# Import the data for the two scenarios.
    os.chdir(input_path1)
    data = np.genfromtxt(file1[0],delimiter = ';')     
    if aggregation[0] == '573':
        ids = data[:,0]
        sens1 = data[:,1]
        pval1 = data[:,2]
    
    os.chdir(input_path2)
    data = np.genfromtxt(file2[0],delimiter = ';')    
    if aggregation[0] == '573':
        sens2 = data[:,1]
        pval2 = data[:,2]
    
# sens_data (pval_data) includes sensitivity (p-value) results for all the oscillations for both scenarios.
    sens_data = np.hstack((sens1[:,np.newaxis], sens2[:,np.newaxis]))
    pval_data = np.hstack((pval1[:,np.newaxis], pval2[:,np.newaxis]))
# Remove those FPUS that don't grow the crop in question or where both scenarios show non-significant sensitivity.
# Remove also those FPUs where either scenario has a nan value for the sensitivity (can happen e.g. if a model has all zero for yield).
    keep_these_indices_pval = np.any((pval_data <= alpha),axis = 1)
    sens_data[~keep_these_indices_pval,:] = 0.0
    remove_these_indices_nan = np.any(np.isnan(sens_data),axis = 1)                      
    sens_data[remove_these_indices_nan,:] = 0.0
    x = sens_data[:,0]
    y = sens_data[:,1]
    
# Calculate the difference in absolute sensitivity between the scenarios.
# Change the values previously changed to zero into nan.
    sens_diff  = np.abs(y) - np.abs(x)
    sens_diff[~keep_these_indices_pval] = np.nan
    sens_diff[remove_these_indices_nan] = np.nan
# Change tabulated data to raster format.
    if aggregation[0] == '573':
        rast_sens_diff = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens_diff)
    
# Mask raster based on cropland data and important cropland mask as well.
    mirca_data = import_mirca_data_visualization_masking(file1[0])
    rast_sens_diff[mirca_data == 0] = np.nan
    mirca_data = mirca_data > 0
    
    return rast_sens_diff, mirca_data
        
def visualize_ensemble_raster(crop,osc,raster_data,raster_mask,savepath,save,colorbar_name,data_type):
    
# Initialize the map figure using Basemap
    m = Basemap(projection='cyl', resolution='c',
        llcrnrlat=-60, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180, )
    
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
      
# Create the colorscale, using RGB information and the function LinearSegmentedColormap
    cdict = {'red':  ((0.0, 0.0, 0.0),
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

         'blue':  ((0.0, 0.0, 0.4),
                   (0.25, 1.0, 1.0),
                   (0.49, 1.0, 0.9),
                   (0.51, 0.9, 0.8),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
            }
    clim=(-5,5)
    cmap = LinearSegmentedColormap('cmap', cdict)
    
# Modify settings for the raster visualization
    raster_data = np.flipud(raster_data)
    raster_mask = np.flipud(raster_mask)
    mask_index = raster_mask * np.isnan(raster_data)
    raster_data[mask_index] = 0
    cs = m.imshow(raster_data[60:,:]*100,clim=clim,cmap=cmap)
        
    m.drawcoastlines(linewidth=0.25)
#    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='white',lake_color='white',zorder = 0)
    
    m.drawmapboundary(fill_color='White')

# Save the figure
    if save == 1:
        os.chdir(savepath)
        plt.savefig(crop+'_'+osc+'_'+data_type+'.png', dpi=300, bbox_inches='tight')
    plt.show(m)

# If colorbar for sensitivity doesn't exist in folder, create it.
    if colorbar_name not in os.listdir(savepath):
        # add colorbar
        plt.figure()
        cs = plt.imshow(raster_data[60:,:]*100,clim=clim,cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(cs, extend = 'both',orientation='horizontal')
        cbar.set_label('Difference in sensitivity magnitude', fontsize=12)
        plt.savefig(colorbar_name, dpi = 300, bbox_inches='tight')

def difference_files_to_visualize(alpha,model,aggregation,setup1,setup2,crop_list,oscillation_list,climate,input_path1,input_path2,savepath,save,additional_info):

# Create a list of files in the folder specified by input_path1, and then make a
# copy of that list, but replace the setup identification for the copied list.
    file_list1 = os.listdir(input_path1)
    file_list2 = []
    for file in file_list1:
        file_list2.append(file.replace(setup1,setup2))
    
    file_save1 = []
    file_save2 = []
    
    for osc in oscillation_list:
        for crop in crop_list: 
# Loop through all the file names and check, that the input parameters match
# for the scanerios being compared. Then append the file name to import it later.
            for file1, file2 in zip(file_list1,file_list2):
                if model[0] in file1 and aggregation[0] in file1 and setup1 in file1 and crop in file1 and osc in file1 and climate[0] in file1 and file1.endswith('.csv') and 'may_sowing' in file1 and 'nino34' not in file1:
                    file_save1.append(file1)
                if model[0] in file2 and aggregation[0] in file2 and setup2 in file2 and crop in file2 and osc in file2 and climate[0] in file2 and file2.endswith('.csv') and 'may_sowing' in file1 and 'nino34' not in file1:
                    file_save2.append(file2)
            print(file_save1)
            print(file_save2)
# Import the data specified in the file_save variables in raster format.
            rast_difference, mirca_data = obtain_raster_data_difference(input_path1,input_path2,aggregation,alpha,file_save1,file_save2)
# Visualize the raster using the specified parameters
            visualize_ensemble_raster(crop,osc,rast_difference,mirca_data,savepath,save,'colorbar_diff.png',additional_info)     

            file_save1 = []
            file_save2 = []
            
            
            
            

aggregation_list = ['573']
climate_list = ['AgMERRA']
crop_list = ['mai','ric','soy','whe']
oscillation_list = ['enso_multiv','iod_multiv','nao_multiv']
alpha = 0.1
save = 1

model_list_main = ['all_models']
input_path1 = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sens_difference'
difference_files_to_visualize(alpha,model_list_main,aggregation_list,'combined','firr',crop_list,oscillation_list,climate_list,input_path1,input_path2,savepath,save,'fullharm_combined_firr')

input_path1 = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_review_v1\noirr_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sens_difference'
difference_files_to_visualize(alpha,model_list_main,aggregation_list,'combined','noirr',crop_list,oscillation_list,climate_list,input_path1,input_path2,savepath,save,'fullharm_combined_noirr')

input_path1 = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_review_v1\noirr_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sens_difference'
difference_files_to_visualize(alpha,model_list_main,aggregation_list,'firr','noirr',crop_list,oscillation_list,climate_list,input_path1,input_path2,savepath,save,'fullharm_firr_noirr')

model_list_main = ['all_fertilizer_models']
input_path1 = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_harmnon\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sens_difference'
difference_files_to_visualize(alpha,model_list_main,aggregation_list,'combined_actfert','combined_fullfert',crop_list,oscillation_list,climate_list,input_path1,input_path2,savepath,save,'fullharm_combined_harmnon_combined')

input_path1 = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_harmnon\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sens_difference'
difference_files_to_visualize(alpha,model_list_main,aggregation_list,'combined_actfert','firr_fullfert',crop_list,oscillation_list,climate_list,input_path1,input_path2,savepath,save,'fullharm_combined_harmnon_firr')


































    