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
    elif 'combined' in file_name:
        mirca_data = mirca_firr + mirca_noirr
        
    return mirca_data

def obtain_raster_data(file_name, path, aggregation):
# Import tablulated data, based on file_name and path to the directory in question.
    os.chdir(path)

    data = np.genfromtxt(file_name,delimiter = ';')

    if aggregation == '573':
        ids = data[:,0]
        anom = data[:,1]
        pval = data[:,2]   
        rast_anom = fpu_data_to_raster(ids,'raster_fpu_573.tif',anom)
        rast_pval = fpu_data_to_raster(ids,'raster_fpu_573.tif',pval)
        
    mirca_data = import_mirca_data_visualization_masking(file_name)
     
    if aggregation == '573':
        rast_anom[mirca_data == 0] = np.nan
        rast_pval[mirca_data == 0] = 1
    
    mirca_data = mirca_data > 0
    
    return rast_anom, rast_pval, mirca_data


def create_title(file_name):
# Create a title for the figure, based on the information in the file-name.
    if 'neg' in file_name:
        if 'iod' in file_name:
            osc_name = 'Negative IOD'
        elif 'nao' in file_name:
            osc_name = 'Negative NAO'
        elif 'enso' in file_name:
            osc_name = u'La Niña'
    elif 'pos' in file_name:
        if 'iod' in file_name:
            osc_name = 'Positive IOD'
        elif 'nao' in file_name:
            osc_name = 'Positive NAO'
        elif 'enso' in file_name:
            osc_name = u'El Niño'
         
    if 'whe' in file_name:
        crop_name = 'Wheat'
    elif 'soy' in file_name:
        crop_name = 'Soybean'
    elif 'mai' in file_name:
        crop_name = 'Maize'
    elif 'ric' in file_name:
        crop_name = 'Rice'
    
    model_list_main = ['all_models','all_fertilizer_models','pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
             'lpj-guess','lpjml','cgms-wofost','clm-crop','epic-tamu',\
             'orchidee-crop','pepic','prysbi2']
    
    for model_name in model_list_main:
        if model_name in file_name:
            if model_name == 'all_models' or model_name == 'all_fertilizer_models':
                title = osc_name + ', ' + crop_name
            else:
                title = osc_name + ', ' + crop_name + ', ' + model_name
    
    return title

        
def extract_raster_anomalies(alpha,file_name,aggregation,input_path,savepath,save):
# Import information of sensitivity as raster
    rast_anom, rast_pval, rast_ha_bol = obtain_raster_data(file_name,input_path,aggregation)
    rast_anom[rast_pval > alpha] = np.nan

# Initialize the map figure using Basemap
    m = Basemap(projection='cyl', resolution='c',
        llcrnrlat=-60, urcrnrlat=90,
        llcrnrlon=-180, urcrnrlon=180, )
    
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
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
    clim = (-5,5)
# Modify settings for the raster visualization
    
    rast_anom = np.flipud(rast_anom)
    rast_ha_bol = np.flipud(rast_ha_bol)
    crop_index = rast_ha_bol * np.isnan(rast_anom)
    rast_anom[crop_index] = 0
    cs = m.imshow(rast_anom[60:,:]*100,clim=clim,cmap=cmap)
        
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='white',lake_color='white',zorder = 0)
    
    m.drawmapboundary(fill_color='White')
    
#    title = create_title(file_name)
#    plt.title(title, loc='left',fontsize=20)

# Save the figure
    if save == 1:
        os.chdir(savepath)
        plt.savefig(file_name.replace('.csv','.png'), dpi=300, bbox_inches='tight')
    plt.show(m)
# If colorbar for anomalies doesn't exist in folder, create it.
    if 'colorbar_anom.png' not in os.listdir(savepath):
        plt.figure()
        cs = m.imshow(rast_anom[60:,:]*100,clim=clim,cmap=cmap)
        plt.gca().set_visible(False)
        cbar = plt.colorbar(cs, extend = 'both',orientation='horizontal')
        cbar.set_label('Median crop yield deviation (%) during strong oscillation phases', fontsize=12)
        plt.savefig('colorbar_anom.png', dpi = 300, bbox_inches='tight')
#        
    

def anom_files_to_visualize(alpha,model_list,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list, input_path,savepath,save):
    
    file_list = os.listdir(input_path)
# Loop throught the paramters and selects the file that is going to be visualized.
    for file in file_list:
        for model in model_list:
            for climate in climate_list:
                for aggregation in aggregation_list:
                    for irrig in irrig_setup_list:
                        for crop in crop_list:
                            for osc in oscillation_list:
                                if model in file and aggregation in file and irrig in file and crop in file and osc in file and climate in file and file.endswith('.csv'):
                                    extract_raster_anomalies(alpha,file,aggregation,input_path,savepath,save)



def extract_aggregated_anomalies(file_name, path, aggregation, alpha):
# Calculate the total and proportional area with significant crop yield anomalies during
# the strong oscillation phases.
    
# Import anomaly data.
    os.chdir(path)
    data = np.genfromtxt(file_name,delimiter = ';')
    ids = data[:,0]
    anom = data[:,1]
    pval = data[:,2]
        
# Disaggregate tabulated data into raster.
    if aggregation == '573':
        rast_anom = fpu_data_to_raster(ids,'raster_fpu_573.tif',anom)
        rast_pval = fpu_data_to_raster(ids,'raster_fpu_573.tif',pval)
     
# Change values without cropland into 0 and 1 for anomaly and p-value, respectively.
    mirca_data = import_mirca_data_visualization_masking(file_name)
    rast_anom[mirca_data == 0] = 0
    rast_pval[mirca_data == 0] = 1.0
    
# Calculate the total and proportional harvested area where significant anomaly can be found.
    total_ha_impacted = np.nansum(mirca_data[rast_pval < alpha])
    total_ha_impacted_prop = total_ha_impacted/np.nansum(mirca_data)

    return total_ha_impacted, total_ha_impacted_prop


def check_row_col(file_name):
# Based on the name of the file, assign it to the correct row and column, in the table.
    col_idx = 0
    row_idx = 0
    
    if 'neg' in file_name:
        col_idx = col_idx + 1
        
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

def write_output_table(output_table,aggregation,model,irrig,alpha,savepath):
# Add information about the table columns and rows. Then export the table.     
    column_header = ['El Nino','La Nina','Positive IOD','Negative IOD','Positive NAO','Negative NAO']
    row_header = [['nan'],['Maize'],['Rice'],['Soy'],['Wheat'],['Maize prop'],['Rice prop'],['Soy prop'],['Wheat prop']]
    
    output_table = np.vstack((column_header,output_table))
    output_table = np.hstack((row_header,output_table))
    
    print output_table
    print 'aggregated_table_'+aggregation+'_'+irrig+'.csv'
    
    os.chdir(savepath)
    
    np.savetxt('anomaly_table_'+model+'_'+aggregation+'_'+irrig+'_alpha_'+str(alpha)+'.csv', output_table, delimiter=";",fmt="%s")


def anom_files_to_aggregate_table(alpha,model_list,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,input_path,savepath):
    file_list = os.listdir(input_path)
# Loop throught the paramters and select the file that is going to be visualized.
    for model in model_list:
        for aggregation in aggregation_list:
            for climate in climate_list:
                for irrig in irrig_setup_list:
                    os.chdir(savepath)
                    output_table = np.zeros((8,6))
                    for file in file_list:
                        for crop in crop_list:
                            for osc in oscillation_list:
                                if aggregation in file and irrig in file and crop in file and osc in file and model in file and climate in file:
                                    print file
# Calculate how much harvested areas have a significant anomaly during the oscillation phase.
                                    total_ha_impacted, total_ha_impacted_prop = extract_aggregated_anomalies(file,input_path,aggregation,alpha)
# Check where to put the values in the table.
                                    row, col = check_row_col(file)
                                    output_table[row,col] = total_ha_impacted
                                    output_table[row+4,col] = total_ha_impacted_prop
# Write out the table with the specified parameters.
            write_output_table(output_table,aggregation,irrig,model,alpha,savepath)


model_list_main = ['all_models']
aggregation_list = ['573']
irrig_setup_list = ['combined']
crop_list = ['whe','mai','ric','soy']
oscillation_list = ['enso_djf_adj','iod_son_adj','nao_djf_adj']
climate_list = ['AgMERRA']
alpha = 0.1
save = 1
input_path = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anomalies'
savepath = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anomalies_figs'
anom_files_to_visualize(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,input_path,savepath,save)



   
model_list_main = ['all_models']
aggregation_list = ['573']
irrig_setup_list = ['combined'] # ['noirr','combined']
crop_list = ['whe','mai','ric','soy']
oscillation_list = ['enso_djf_adj','iod_son_adj','nao_djf_adj']
climate_list = ['AgMERRA']
alpha = 0.1
save = 1
input_path = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anomalies'
savepath = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\anomalies_table'   
             
anom_files_to_aggregate_table(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,input_path,savepath)















































    