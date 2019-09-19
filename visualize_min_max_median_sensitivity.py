#!/usr/bin/env python
# -*- coding: utf-8 -*- import pylab
import os
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


def obtain_raster_data_ensemble(file_list, path, aggregation,oscillation_list,crop_list,alpha,savepath):
    os.chdir(path)
    i = 0

# Import the sensitivity and related p-value results for the files in file_list
    for file_name in file_list:
        data = np.genfromtxt(file_name,delimiter = ';')     
        if aggregation == '573':
            ids = data[:,0]
            sens = data[:,1]
            pval = data[:,2]
            
        if i == 0:
            sens_ensemble = sens[:,np.newaxis]
            pval_ensemble = pval[:,np.newaxis]
            i = 1
        else:
            sens_ensemble = np.hstack((sens_ensemble, sens[:,np.newaxis]))
            pval_ensemble = np.hstack((pval_ensemble, pval[:,np.newaxis]))

# Change sensitivity values into nan where p-value is larger than alpha (0.1).
    sens_ensemble[pval_ensemble >= alpha] = np.nan
# Select the ensemble sensitivity, and replicate that array as many times as there are other models.
    sens_ensemble_all = sens_ensemble[:,0]
    sens_ensemble_all = np.repeat(sens_ensemble_all[:,np.newaxis],sens_ensemble.shape[1]-1,axis = 1)
# Select the sensitivity results of the individual model results only.
    sens_ensemble_models = sens_ensemble[:,1:]

# Multiply the sensitivity result arrays of ensemble with individual results.
# If they agree in sign, the result of the multiplication is positive.
# Thus, check if the result is larger than zero.
# Also, first get rid of nans to dodge any warnings during running the code
    sens_ensemble_all_nonan = np.copy(sens_ensemble_all)
    sens_ensemble_models_nonan = np.copy(sens_ensemble_models)
    sens_ensemble_all_nonan[np.isnan(sens_ensemble_all_nonan)] = 0
    sens_ensemble_models_nonan[np.isnan(sens_ensemble_models_nonan)] = 0
    sens_ensemble_sign = sens_ensemble_all_nonan*sens_ensemble_models_nonan > 0
# Calculate the percentage of models that have a True value from the calculation above.
    sens_ensemble_sign_sum = np.nansum(sens_ensemble_sign,axis = 1).astype('f')
    sens_ensemble_size = sens_ensemble_sign.shape[1]
    sens_ensemble_sign_tot = sens_ensemble_sign_sum/sens_ensemble_size
    
# Combine and save the result in the array created above, as a csv file.
    sens_ensemble_sign_tot_txt = np.hstack((ids[:,np.newaxis],sens_ensemble_sign_tot[:,np.newaxis]))
    for osc_type in oscillation_list:
        for crop in crop_list:
            if crop in file_list[0] and osc_type in file_list[0]:
                print(crop)
                print(osc_type)
                print(file_list[0])
                os.chdir(savepath)
                np.savetxt('sensitivity_agreement_'+osc_type+'_'+crop+'.csv',sens_ensemble_sign_tot_txt, delimiter=";")
    
# Obtain the median sensitivity of the individual models
    sens_ensemble_models_median = np.nanmedian(sens_ensemble_models,axis = 1)
    sens_ensemble_models_median[np.isnan(sens_ensemble_all[:,0]),np.newaxis] = np.nan
    
# Check maximum and minimum magnitude only for those models and FPUs, where significant sensitivity
# of same sign as ensemble result is observed.
# Get rid of nans to get rid off warnings during running the code.
    
    sens_sign_any_model = np.any(sens_ensemble_sign,axis=1)
    sens_ensemble_models_temp = sens_ensemble_models[sens_sign_any_model,:] 
    sens_ensemble_models_max_temp = np.nanmax(abs(sens_ensemble_models_temp),axis = 1)
    sens_ensemble_models_min_temp = np.nanmin(abs(sens_ensemble_models_temp),axis = 1)
    
    sens_ensemble_models_max = np.zeros(sens_sign_any_model.shape)
    sens_ensemble_models_min = np.zeros(sens_sign_any_model.shape)
    sens_ensemble_models_max[sens_sign_any_model] = sens_ensemble_models_max_temp
    sens_ensemble_models_min[sens_sign_any_model] = sens_ensemble_models_min_temp
    

# As above the magnitude of sensitivity was obtained, correct the values where actual sensitivity
# is negative by multiplying those values by -1.
# Get rid of nans to get rid off warnings during running the code.
    sens_sign_temp = np.copy(sens_ensemble_all[:,0])
    sens_sign_temp[np.isnan(sens_sign_temp)] = 0
    sens_ensemble_models_max[sens_sign_temp < 0] = sens_ensemble_models_max[sens_sign_temp < 0] * -1
    sens_ensemble_models_min[sens_sign_temp < 0] = sens_ensemble_models_min[sens_sign_temp < 0] * -1

# Change tabulated data into raster format.
    if aggregation == '573':
        rast_agreement = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens_ensemble_sign_tot)
        rast_max = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens_ensemble_models_max)
        rast_min = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens_ensemble_models_min)
        rast_median = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens_ensemble_models_median)
# Obtain harvested areas data.
    mirca_data = import_mirca_data_visualization_masking(file_name)
    
# Change areas where no harvesed areas exist to nan.
    rast_agreement[mirca_data == 0] = np.nan
    rast_max[mirca_data == 0] = np.nan
    rast_min[mirca_data == 0] = np.nan
    rast_median[mirca_data == 0] = np.nan
    
# Create a cropland mask.
    mirca_data = mirca_data > 0
    
    return rast_agreement, rast_max, rast_min, rast_median, mirca_data
        
def visualize_ensemble_raster(crop,osc,raster_data,raster_mask,savepath,save,cmap,colorbar_name,data_type):
    
# Initialize the map figure using Basemap
    m = Basemap(projection='cyl', resolution='c',
        llcrnrlat=-60, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180, )
    
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
# Create the colorscale, using RGB information and the function LinearSegmentedColormap  
    if cmap == 'redblue':
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
        clim=(-5,5)
        cbar_label = 'Crop yield deviation (%) per unit index change'
        cmap = LinearSegmentedColormap('cmap', cdict)

    elif cmap == 'whiteblue':
        cdict = {'blue':  ((0.0, 0.8, 0.9),
                       (0.01, 0.9, 1.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 0.4, 1.0)),
    
             'green': ((0.0, 0.9, 0.9),
                       (0.01, 0.9, 0.9),
                       (0.5, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'red':  ((0.0, 1.0, 0.9),
                       (0.01, 0.9, 0.8),
                       (0.5, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
             }
        clim=(0,100)
        cbar_label = 'Proportion of individual models that agree with the ensemble sensitivity sign'
        cmap = LinearSegmentedColormap('cmap', cdict)
    
    
# Modify settings for the raster visualization.
    raster_data = np.flipud(raster_data)
    raster_mask = np.flipud(raster_mask)
    mask_index = raster_mask * np.isnan(raster_data)
    raster_data[mask_index] = 0
    cs = m.imshow(raster_data[60:,:]*100,clim=clim,cmap=cmap)
        
    m.drawcoastlines(linewidth=0.25)
#    m.drawcountries(linewidth=0.25)
    m.fillcontinents(color='white',lake_color='white',zorder = 0)
    
    m.drawmapboundary(fill_color='White')
    
# Save the figure.
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
        cbar.set_label(cbar_label, fontsize=12)
        plt.savefig(colorbar_name, dpi = 300, bbox_inches='tight')
        

 

def ensemble_files_to_visualize(alpha,model_list,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,input_path,savepath,save):
    
    file_list_temp = os.listdir(input_path)
    file_list = []
    
# Loop through the parameters and files in file_list_temp, and create a list of files,
# where the sensitivity result of each model is listed in the order specified by model_list_main.
    for osc in oscillation_list:
        for crop in crop_list:           
            for climate in climate_list:
                for model in model_list:    
                    for aggregation in aggregation_list:
                        for irrig in irrig_setup_list:
                            for file in file_list_temp:
                                if model in file and aggregation in file and irrig in file and crop in file and osc in file and climate in file and file.endswith('.csv') and 'annual_harvest' not in file:
                                    file_list.append(file)
# Import raster data about the agreement of the individual models compared to the individual model results
# as well as maximum, minimum and median results of the individual model results. Also, import
# information about ahrvested areas.
            rast_agreement, rast_max, rast_min, rast_med, mirca_data = obtain_raster_data_ensemble(file_list,input_path,aggregation,oscillation_list,crop_list,alpha,savepath)
# Create visualization about the rasters imported above useing the specified parameters.
            visualize_ensemble_raster(crop,osc,rast_max,mirca_data,savepath,save,'redblue','colorbar_sens.png','max')
            visualize_ensemble_raster(crop,osc,rast_min,mirca_data,savepath,save,'redblue','colorbar_sens.png','min')     
            visualize_ensemble_raster(crop,osc,rast_agreement,mirca_data,savepath,save,'whiteblue','colorbar_agreement.png','agreement')     
            visualize_ensemble_raster(crop,osc,rast_med,mirca_data,savepath,save,'redblue','colorbar_sens.png','median')     

            file_list = []


model_list_main = ['all_models','pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
             'lpj-guess','lpjml','cgms-wofost','epic-tamu',\
             'orchidee-crop','pepic']
aggregation_list = ['573']
irrig_setup_list = ['combined'] # ['noirr','combined']
climate_list = ['AgMERRA']
crop_list = ['mai','ric','soy','whe']
oscillation_list = ['enso_multiv','iod_multiv','nao_multiv']
alpha = 0.1
save = 1
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\min_max_median_agreement'

ensemble_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,input_path,savepath,save)






































    