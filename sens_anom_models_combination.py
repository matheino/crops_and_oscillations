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


def visualize_raster(crop,osc,raster_data,raster_mask,savepath,save,cmap,clim,colorbar_name,data_type,title):

# Initialize the map figure using Basemap
    m = Basemap(projection='cyl', resolution='c',
        llcrnrlat=-60, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180)
    
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
        
# Create the colorscale   
    if cmap == 'sens_anom_models_combined':
        
        col0 = np.array([0.9,0.9,0.9])
        col1 = '#fdd98c'
        col2 = '#c587bb'
        col3 = '#7cb542'
        col4 = '#f49a7a'
        col5 = '#b45aa3'
        col6 = '#2e7b3c'
        col7 = '#d45350'
        col8 = '#833190'
        col9 = '#0e542b'
        
        colors = [col0,col1,col2,col3,col4,col5,col6,col7,col8,col9]
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
    
#    plt.title(title, loc='left',fontsize=20)

# Save the figure
    if save == 1:
        os.chdir(savepath)
        plt.savefig(crop+'_'+osc+'_'+data_type+'.png', dpi=300, bbox_inches='tight')
    plt.show(m)
# If colorbar for sensitivity doesn't exist in folder, create it.
    if colorbar_name not in os.listdir(savepath):
        # add colorbar
        plt.figure()
        cs = m.imshow(raster_data[60:,:],clim=clim,cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(cs, extend = 'both',orientation='horizontal')
        cbar.set_label(cbar_label, fontsize=12)
        plt.savefig(colorbar_name, dpi = 300, bbox_inches='tight')



model = 'all_models'
aggreg = '573'
irrig= 'combined' # ['noirr','combined']
climate = 'AgMERRA'
crop_list = ['mai','ric','soy','whe']
oscillation_list = ['enso_djf_adj','iod_son_adj','nao_djf_adj']
alpha = 0.1
savepath = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anom_sens_agreement_figs'


# Loop through all oscillations and crops
for crop in crop_list:
    for oscillation in oscillation_list:
# Import information about the how well the individual models agree with the ensemble sensitivity result
        os.chdir(r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\min_max_median_agreement')
        model_comp = np.genfromtxt('sensitivity_agreement_'+oscillation+'_'+crop+'.csv',delimiter = ';')

# Import information about the average anomalies during strong, positive and negative, oscillation phases.
        files_anoms = os.listdir(r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anomalies')
        os.chdir(r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anomalies')
        for file in files_anoms:
            if crop in file and model in file and aggreg in file and irrig in file and climate in file and oscillation in file and crop in file and file.endswith('.csv'):
                if 'pos' in file:
                    anom_pos = np.genfromtxt(file,delimiter = ';')
                elif 'neg' in file:
                    anom_neg = np.genfromtxt(file,delimiter = ';')

# Import information about sensitivity information.
        files_sens = os.listdir(r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity')
        os.chdir(r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity')
        for file in files_sens:
            if crop in file and model in file and aggreg in file and irrig in file and climate in file and oscillation in file and crop in file and file.endswith('.csv'):
                sens = np.genfromtxt(file,delimiter = ';')

# Set anomalies and sensitivity (1st index) to zero in FPUs where they're not significant
# (p-value > alpha; 2nd index). Also, set anomalies to zero in areas, where sensitivity is insiginificant.
# ids include information about the FPUs.
        ids = model_comp[:,0]
        anom_pos[anom_pos[:,2] > alpha,1] = 0.0
        anom_neg[anom_neg[:,2] > alpha,1] = 0.0
        anom_pos[sens[:,2] > alpha,1] = 0.0
        anom_neg[sens[:,2] > alpha,1] = 0.0
        sens[sens[:,2] > alpha,1] = 0.0            
# Select the extent of model agreement, anomaly and sensitivity numbers from the arrays.
        model_comp = model_comp[:,1]
        anom_pos = anom_pos[:,1]
        anom_neg = anom_neg[:,1]
        sens_pval = sens[:,2]
        sens = sens[:,1] 
        
# Initialize arrays for checking if anomaly and sensitivity results agree within each FPU.
        anom_sens_agr_1 = np.zeros(sens.shape[0])
        anom_sens_agr_2 = np.zeros(sens.shape[0])
        anom_sens_agr_3 = np.zeros(sens.shape[0])
# Loop through all the FPUs
        for i in range(0,sens.shape[0]):
# Set dummy variable number 3 as 1, if both negative and positive phase give a result that is consistent with
# the sensitivity result
            if (sens[i] > 0 and anom_pos[i] > 0 and anom_neg[i] < 0) or (sens[i] < 0 and anom_pos[i] < 0 and anom_neg[i] > 0):
                anom_sens_agr_1[i] = 0
                anom_sens_agr_2[i] = 0
                anom_sens_agr_3[i] = 1
# Set dummy variable number 2 as 1, if either negative or positive phase give a result that is consistent with
# the sensitivity result
            elif (sens[i] > 0 and (anom_pos[i] > 0 or anom_neg[i] < 0)) or (sens[i] < 0 and (anom_pos[i] < 0 or anom_neg[i] > 0)):
                anom_sens_agr_1[i] = 0
                anom_sens_agr_2[i] = 1
                anom_sens_agr_3[i] = 0
# Set dummy variable number 3 as 1, if neither negative or positive phase give a result that is consistent with
# the sensitivity result
            elif (sens[i] > 0 and anom_pos[i] < 0 and anom_neg[i] > 0) or (sens[i] <= 0 and anom_pos[i] >= 0 and anom_neg[i] <= 0):
                anom_sens_agr_1[i] = 1
                anom_sens_agr_2[i] = 0
                anom_sens_agr_3[i] = 0
        
# Categorize FPUs into 9 categories based on the alignment of anomaly results to the sensitivity results
# and the agreement of the results from individual models to the ensemble result.
        cat1 = np.all(np.hstack((model_comp[:,np.newaxis] <= 1.0/3.0, anom_sens_agr_1[:,np.newaxis])),axis = 1)
        cat2 = np.all(np.hstack((model_comp[:,np.newaxis] <= 1.0/3.0, anom_sens_agr_2[:,np.newaxis])),axis = 1)
        cat3 = np.all(np.hstack((model_comp[:,np.newaxis] <= 1.0/3.0, anom_sens_agr_3[:,np.newaxis])),axis = 1)
        cat4 = np.all(np.hstack((model_comp[:,np.newaxis] > 1.0/3.0, model_comp[:,np.newaxis] <= 2.0/3.0, anom_sens_agr_1[:,np.newaxis])),axis = 1)
        cat5 = np.all(np.hstack((model_comp[:,np.newaxis] > 1.0/3.0, model_comp[:,np.newaxis] <= 2.0/3.0, anom_sens_agr_2[:,np.newaxis])),axis = 1)
        cat6 = np.all(np.hstack((model_comp[:,np.newaxis] > 1.0/3.0, model_comp[:,np.newaxis] <= 2.0/3.0, anom_sens_agr_3[:,np.newaxis])),axis = 1)
        cat7 = np.all(np.hstack((model_comp[:,np.newaxis] > 2.0/3.0, anom_sens_agr_1[:,np.newaxis])),axis = 1)
        cat8 = np.all(np.hstack((model_comp[:,np.newaxis] > 2.0/3.0, anom_sens_agr_2[:,np.newaxis])),axis = 1)
        cat9 = np.all(np.hstack((model_comp[:,np.newaxis] > 2.0/3.0, anom_sens_agr_3[:,np.newaxis])),axis = 1)
        
# Set values into the aggreg_results variable, based on the categorization results obtained above.
        aggreg_results = np.zeros(sens.shape[0])*np.nan
        aggreg_results[cat1] = 1.0
        aggreg_results[cat2] = 2.0     
        aggreg_results[cat3] = 3.0        
        aggreg_results[cat4] = 4.0        
        aggreg_results[cat5] = 5.0
        aggreg_results[cat6] = 6.0
        aggreg_results[cat7] = 7.0
        aggreg_results[cat8] = 8.0
        aggreg_results[cat9] = 9.0
        aggreg_results[sens_pval > alpha] = np.nan
     
# Disaggregate the tabulated results into a raster.
        aggreg_results_raster = fpu_data_to_raster(ids,'raster_fpu_573.tif',aggreg_results)
# Import a harvested areas mask based on the crop in question
        mirca_mask = import_mirca_data_visualization_masking(crop+'combined')
# Change the values in the raster to nan, if the crop in question is not grown there
        aggreg_results_raster[mirca_mask == 0] = np.nan
# Create a boolean array about where the crop is grown
        mirca_mask = mirca_mask > 0
# Visualize the raster based on the parameters given.
        visualize_raster(crop,oscillation,aggreg_results_raster,mirca_mask,savepath,1,'sens_anom_models_combined',(0,9),'aggreg_cbar','aggreg_results',crop+oscillation)
        
