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

def obtain_raster_data(file_name, path, aggregation):
# Import tablulated data, based on file_name and path to the directory in question.
    os.chdir(path)

    data = np.genfromtxt(file_name,delimiter = ';')

    if aggregation == '573':
        ids = data[:,0]
        sens = data[:,1]
        if 'rsquared' in file_name:
            pval = np.zeros(sens.shape)
        else:
            pval = data[:,2]
            
        rast_sens = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens)
        rast_pval = fpu_data_to_raster(ids,'raster_fpu_573.tif',pval)
        
    mirca_data = import_mirca_data_visualization_masking(file_name)
     
    rast_sens[mirca_data == 0] = np.nan
    rast_pval[mirca_data == 0] = 1
    
    mirca_data = mirca_data > 0
    
    return rast_sens, rast_pval, mirca_data


def create_title(file_name):
# Create a title for the figure, based on the information in the file-name.
    if 'iod' in file_name:
        osc_name = 'IOD'
    elif 'nao' in file_name:
        osc_name = 'NAO'
    elif 'enso' in file_name:
        osc_name = 'ENSO'
        
    if 'whe' in file_name:
        crop_name = 'Wheat'
    elif 'soy' in file_name:
        crop_name = 'Soybean'
    elif 'mai' in file_name:
        crop_name = 'Maize'
    elif 'ric' in file_name:
        crop_name = 'Rice'
    
    model_list_main = ['all_models','all_fertilizer_models','pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
             'lpj-guess','lpjml','cgms-wofost','epic-tamu',\
             'orchidee-crop','pepic']

    for model_name in model_list_main:
        if model_name in file_name:
            if model_name == 'all_models' or model_name == 'all_fertilizer_models':
                title = osc_name + ', ' + crop_name
                model_title = model_name.lower()
            else:
                title = osc_name + ', ' + crop_name + ', ' + model_name.lower()
                model_title = model_name.lower()
      
    return title, model_title

        
def extract_raster_sensitivity(alpha,file_name,aggregation,input_path,savepath,save):

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

    if 'rsquared' not in file_name:
# Create the colorscale, using RGB information and the function LinearSegmentedColormap    
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
        cmap = LinearSegmentedColormap('cmap', cdict)
        cbarmin, cbarmax = -10,10
        clim = (cbarmin,cbarmax)
        
    elif 'rsquared' in file_name:
        cbarmin, cbarmax = 0,25
        cmap = 'Greens'
        clim = (cbarmin, cbarmax)
        
# Modify settings for the raster visualization
    rast_sens = np.flipud(rast_sens)
    rast_ha_bol = np.flipud(rast_ha_bol)
    crop_index = rast_ha_bol * np.isnan(rast_sens)
    rast_sens[crop_index] = 0

    m.imshow(rast_sens[60:,:]*100,clim=clim,cmap=cmap)    
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='white',lake_color='white',zorder = 0)
    m.drawmapboundary(fill_color='White')
    
    if 'all_models' not in file_name and 'all_fertilizer_models' not in file_name:
        title, model_title = create_title(file_name)
        plt.title(title, loc='left',fontsize=20)
        plt.title(model_title, loc='left',fontsize=20)

# Save the figure
    file_name = file_name.replace('jmasst_','')
        
    if save == 1:
        os.chdir(savepath)
        plt.savefig(file_name.replace('.csv','.png'), dpi=300, bbox_inches='tight')
    plt.show(m)
    print(file_name)

# If colorbar for sensitivity doesn't exist in folder, create it.
    if 'colorbar_sens.png' not in os.listdir(savepath) or 'colobar_rsquared.png' not in os.listdir(savepath) :
        plt.figure()
        img = plt.imshow(rast_sens[60:,:]*100,clim=clim,cmap=cmap)
        plt.gca().set_visible(False)


        if 'rsquared' in file_name:
            cbar = plt.colorbar(img, extend = 'both',orientation='horizontal', ticks = np.arange(int(cbarmin), int(cbarmax+5), int(5)))
            cbar.set_label('% of yield variability explained by the oscillations', fontsize=12)
            plt.savefig('colobar_rsquared.png', dpi = 300, bbox_inches='tight')   
            cbar.ax.set_xticklabels(np.arange(int(cbarmin), int(cbarmax+5), int(5)))
            
        else:
            cbar = plt.colorbar(img, extend = 'both',orientation='horizontal', ticks = np.arange(int(cbarmin), int(cbarmax+2), int(2)))
            cbar.set_label('Crop yield deviation (%) per standard deviation index change', fontsize=12)
            plt.savefig('colorbar_sens.png', dpi = 300, bbox_inches='tight')
            cbar.ax.set_xticklabels(np.arange(int(cbarmin), int(cbarmax+2), int(2)))
        plt.show()
    

def sens_files_to_visualize(alpha,model_list,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,growing_season,input_path,savepath,save):
    
    file_list = os.listdir(input_path)
# Loop throught the paramters and selects the file that is going to be visualized.
    for climate in climate_list:
        for model in model_list:    
            for aggregation in aggregation_list:
                for irrig in irrig_setup_list:
                    for crop in crop_list:
                        for osc in oscillation_list:
                            for file in file_list:
                                if model in file and aggregation in file and irrig in file and crop in file and osc in file and climate in file and growing_season in file and file.endswith('.csv'):
                                    extract_raster_sensitivity(alpha,file,aggregation,input_path,savepath,save)



def extract_aggregated_sensitivity(file_name, path, aggregation, alpha):
# Calculate the total absolute and porportional harvested area in areas, where
# sensitivity is significant and negative or postive.
    
# Import sensitivity data.
    os.chdir(path)
    data = np.genfromtxt(file_name,delimiter = ';')
    ids = data[:,0]
    sens = data[:,1]
    pval = data[:,2]
       
# Disaggregate tabulated data into raster
    if aggregation == '573':
        ids = data[:,0]
        sens = data[:,1]
        pval = data[:,2]
        rast_sens = fpu_data_to_raster(ids,'raster_fpu_573.tif',sens)
        rast_pval = fpu_data_to_raster(ids,'raster_fpu_573.tif',pval)
        
# Import harvested areas data
    mirca_data = import_mirca_data_visualization_masking(file_name)
     
# Change values without cropland into 0 and 1 for sensitivity and p-value, respectively.
    rast_sens[mirca_data == 0] = 0.0
    rast_pval[mirca_data == 0] = 1.0
# Change nan values into 0 and 1 for sensitivity and p-value, respectively.
    rast_sens[np.isnan(rast_sens)] = 0.0
    rast_pval[np.isnan(rast_pval)] = 1.0
        
# Calculate the total harvested area where sensitivity is positive and negative.
    mirca_data_pval = np.copy(mirca_data)
    mirca_data_pval[rast_pval > alpha] = 0.0
    
    total_ha_impacted_neg = np.nansum(mirca_data_pval[rast_sens < 0])
    total_ha_impacted_pos = np.nansum(mirca_data_pval[rast_sens > 0])

# Calculate the proportional harvested area where sensitivity is positive and negative.
    total_ha_impacted_prop_neg = total_ha_impacted_neg/np.nansum(mirca_data)
    total_ha_impacted_prop_pos = total_ha_impacted_pos/np.nansum(mirca_data)

    return total_ha_impacted_neg, total_ha_impacted_pos, total_ha_impacted_prop_neg, total_ha_impacted_prop_pos


def check_row_col(file_name):
# Based on the name of the file, assign it to the correct row and column, in the table.
    col_idx = 0
    row_idx = 0
    
    if 'iod' in file_name:
        col_idx = col_idx + 2        
    elif 'nao' in file_name:
        col_idx = col_idx + 4
    
    if 'ric' in file_name:
        row_idx = row_idx + 1
    elif 'soy' in file_name:
        row_idx = row_idx + 2
    elif 'whe' in file_name:
        row_idx = row_idx +3
        
    return row_idx, col_idx

def write_output_table(output_table,aggregation,climate_input,irrig,model,alpha,savepath):
# Add information about the table columns and rows. Then export the table.
    column_header = ['ENSO-','ENSO+','IOD-','IOD+','NAO-','NAO+']
    row_header = [['nan'],['Maize'],['Rice'],['Soy'],['Wheat'],['Maize prop'],['Rice prop'],['Soy prop'],['Wheat prop']]
    
    output_table = np.vstack((column_header,output_table))
    output_table = np.hstack((row_header,output_table))
    
    print(output_table)
    print('sensitivity_aggregated_table_'+model+'_'+aggregation+'_'+irrig+'_'+climate_input+'_'+'_alpha_'+str(alpha)+'.csv')
    
    os.chdir(savepath)
    
    np.savetxt('sensitivity_aggregated_table_'+model+'_'+aggregation+'_'+irrig+'_'+climate_input+'_'+'_alpha_'+str(alpha)+'.csv', output_table, delimiter=";",fmt="%s")


def sens_files_to_aggregate_table(alpha,model_list,aggregation_list,irrig_setup_list,crop_list,oscillation_list,input_path,savepath,climate_input, growing_season):
    
    file_list = os.listdir(input_path)
# Loop throught the paramters and select the file that is going to be visualized.
    for model in model_list:
        for aggregation in aggregation_list:
            for irrig in irrig_setup_list:
                os.chdir(savepath)
                output_table = np.zeros((8,6))
                for file in file_list:
                    for crop in crop_list:
                        for osc in oscillation_list:
                            if aggregation in file and irrig in file and crop in file and osc in file and model in file and climate_input in file and growing_season in file and 'nino34' not in file:
                                print(file)
# Calculate how much harvested areas have significant sensitivity.
                                total_neg, total_pos, prop_neg, prop_pos = extract_aggregated_sensitivity(file,input_path,aggregation,alpha)
# Check where to put the values in the table.
                                row, col = check_row_col(file)
                                output_table[row,col] = total_neg
                                output_table[row,col+1] = total_pos
                                output_table[row+4,col] = prop_neg
                                output_table[row+4,col+1] = prop_pos
# Write out the table with the specified parameters.
            write_output_table(output_table,aggregation,climate_input,irrig,model,alpha,savepath)

        
        

aggregation_list = ['573']
irrig_setup_list = ['combined']
climate_list = ['AgMERRA']
crop_list = ['mai','whe','ric','soy']
oscillation_list = ['enso_multiv','iod_multiv','nao_multiv', 'rsquared']

alpha = 0.1
save = 1

model_list_main = ['all_models','all_fertilizer_models']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity_figs_multiv'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

oscillation_list = ['enso_multiv','iod_multiv','nao_multiv']
model_list_main = ['pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
             'lpj-guess','lpjml','cgms-wofost','epic-tamu','orchidee-crop','pepic']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity_figs_models'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)


oscillation_list = ['enso_multiv','iod_multiv','nao_multiv']
model_list_main = ['all_models']
climate_input = 'AgMERRA'
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity_table'
sens_files_to_aggregate_table(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,input_path,savepath,climate_input,'may_sowing')

model_list_main = ['all_models']
climate_list = ['Princeton']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

model_list_main = ['all_models']
climate_list = ['AgMERRA']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_default\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_default\sensitivity_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

model_list_main = ['all_models']
irrig_setup_list = ['firr']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_fullharm\sensitivity_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

model_list_main = ['all_models']
irrig_setup_list = ['noirr']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\noirr_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\noirr_fullharm\sensitivity_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

model_list_main = ['all_fertilizer_models']
irrig_setup_list = ['firr']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_harmnon\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\firr_harmnon\sensitivity_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

model_list_main = ['all_fertilizer_models']
irrig_setup_list = ['combined']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_harmnon\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_harmnon\sensitivity_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'may_sowing',input_path,savepath,save)

model_list_main = ['all_models']
irrig_setup_list = ['combined']
oscillation_list = ['growing_season_enso','growing_season_iod','growing_season_nao']
input_path = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity_gs_figs'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\combined_fullharm\sensitivity_gs_figs'
sens_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,'growing_season',input_path,savepath,save)






































    