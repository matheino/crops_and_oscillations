import os
import numpy as np

def import_file_lists(path, irrig_setup, model_list,runtype,aggregation,crop,climate_list,aggreg_season):
# Function creates a list of files in a folder (specified by path) that include the input specifications
# of models included (model_list), model configurations (runtype), aggregation (573 FPUs), crop, 
# weather forcing (climate_list), and temporal harvest aggregation (aggreg_season).

# If a list of all models or all models that simulate nutrient stress are included, the input specifications of
# 'all_models' and 'all_fertilizer_models' are used, respectively.
    if model_list[0] == 'all_models':
        model_list = ['pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
         'lpj-guess','lpjml','cgms-wofost','epic-tamu',\
         'orchidee-crop','pepic']
        
    if model_list[0] == 'all_fertilizer_models':
        model_list = ['pdssat','epic-boku','epic-iiasa','gepic','papsim','pegasus',\
        'epic-tamu','orchidee-crop','pepic']
        
    runtype_temp = runtype[:]
    
# Adjust runtype parameter, to work in case lpjml and lpj-guess are included
# in the file list. 
    if 'default' in runtype_temp:
        runtype_temp.append('default')
    elif 'fullharm' in runtype_temp:
        runtype_temp.append('harmnon')
    elif 'harmnon' in runtype_temp:
        runtype_temp.append('harmnon')
    
# Loop though the different model parameters to create file lists for data to be imported later in the script.
    filenames = []
    for file in os.listdir(path):
        for climate in climate_list:
            for model in model_list:
                if file.endswith(".csv") and irrig_setup[0] in file and model in file and 'yield' in file and \
                (runtype_temp[0] in file or (runtype_temp[1] in file and ('lpjml' in file or 'lpj-guess' in file)))  and \
                aggregation[0] in file and crop[0] in file and climate[0] in file and aggreg_season in file:
                    filenames.append(file)
    
    print(filenames)
    
# Export list of filenames, where all files that have the required configurations are included.
    return filenames


def detrend_data(filename,path):
# Detrends data by substracting a five year (three-year at the ends of the timeseries) mean.
    os.chdir(path)
    data = np.genfromtxt(filename, delimiter=';') # np.loadtxt??
    fpu_ids = data[:,0]
# Remove first column (id), second column (no data for all cell-crop combinations) and last column (no data for all cell-crop combinations).
    data = data[:,2:-1]
# If the model is orchidee-crop and climate forcing is princeton, adjust the time span, so that it matches with the others
    if 'Princeton' in filename and 'orchid' in filename:
        data = data[:,:-2]
# Create an empty 
    detrended_data_prop = np.zeros((data.shape[0],data.shape[1]-2))*np.nan
    
# Loop goes through each year of data, starting from first year + 1,
# and calculates the proportional and absolute deviation from the 5-year 
# (3-year at both ends of the time series) mean.
    for i in range(0, data.shape[1]-2):
        
        if i == 0 or i == range(0, data.shape[1]-1)[-2]:
            yrly_indices = np.linspace(i,i+2,3).astype(int) # Indices for 3 years surrounding the year in question.
        else:
            yrly_indices = np.linspace(i-1,i+3,5).astype(int) # Indices for 5 years surrounding the year in question.

        baseline_mean = np.mean(data[:,yrly_indices],axis=1)
        yrly_val = data[:,i+1]
        
# To avoid dividing with zero, or having NaN values in the calculation
# all nan values are ignored in the calculations and zero values in 
# the baseline_mean
        check_for_nans = np.hstack((~np.isnan(baseline_mean[:,np.newaxis]),~np.isnan(yrly_val[:,np.newaxis]), baseline_mean[:,np.newaxis] != 0))
        not_nan_array = np.all(check_for_nans, axis = 1)                
        detrended_data_prop_temp = (yrly_val[not_nan_array] - baseline_mean[not_nan_array]) / baseline_mean[not_nan_array]
        detrended_data_prop[not_nan_array,i] = detrended_data_prop_temp
# If all data in an array is zero, change values into nan.
        all_data_zero = np.all(detrended_data_prop == 0, axis = 1)
        detrended_data_prop[all_data_zero,:] = np.nan
                
    return fpu_ids, detrended_data_prop
        

def import_oscillation_index(oscillation_index,climate_input,aggreg_season):
# Imports each oscillation index (ENSO, IOD, NAO).
    os.chdir('D:\work\data\oscillation_indices')
# Imports the raw oscillation index data based on the oscillation in question
# defined by oscillation_index.
    if 'enso' in oscillation_index[0]:
        osc_raw = np.genfromtxt('jmasst1868-today.filter-5')
    elif 'iod' in oscillation_index[0]:
        osc_raw = np.genfromtxt('dmi.long.data',skip_header = 1,skip_footer = 6)
    elif 'nao' in oscillation_index[0]:
        osc_raw = np.genfromtxt('nao_hurrell_pc_monthly.txt',skip_header = 1)

# Select the appropriate years of oscillation index data based on the climate
# forcing of the crop yield data in question.
    if 'AgMERRA' in climate_input[0]:
        years_to_select = np.all([osc_raw[:,0] >= 1980, osc_raw[:,0] <= 2010],axis = 0)
        osc_raw = osc_raw[years_to_select,:]     
    elif 'Princeton' in climate_input[0]:
        years_to_select = np.all([osc_raw[:,0] >= 1948, osc_raw[:,0] <= 2008],axis = 0)
        osc_raw = osc_raw[years_to_select,:]

# Remove strong El Niño years of 1982 and 1997 from IOD data.
#    if 'iod' in oscillation_index[0]:
#        years_to_select_iod = np.all([osc_raw[:,0] != 1982, osc_raw[:,0] != 1997],axis = 0)
#        osc_raw = osc_raw[years_to_select_iod,:]
        
# Calculate the seasonal average of the index based on the parameters specified
# as the inputs of the function, i.e. aggreg_season and oscillation_index.
    if aggreg_season == 'may_sowing' and 'djf_adj' in oscillation_index[0]:
        osc_raw_months = np.vstack((osc_raw[2:-2,12],osc_raw[3:-1,1],osc_raw[3:-1,2])).T
        osc_idx = np.mean(osc_raw_months,axis = 1)
    elif aggreg_season == 'may_sowing' and 'son_adj' in oscillation_index[0]:
        osc_raw_months = osc_raw[2:-2,[9,10,11]]
        osc_idx = np.mean(osc_raw_months,axis = 1)        

# Standardize the oscillation index.  
    osc_idx_standardized = (osc_idx - np.mean(osc_idx)) / np.std(osc_idx)
# Add the corresponding year to the standardized oscillation index array,
# and remove the oscillation index data at the tails, where no crop yield data exist
    osc_idx_standardized = np.vstack((osc_raw[2:-2,0],osc_idx_standardized)).T

# Isolate the years when the oscillation index is larger / smaller than 75% of years.
    pos_yrs = osc_idx_standardized[osc_idx_standardized[:,1] > np.percentile(osc_idx_standardized[:,1],75),0]
    neg_yrs = osc_idx_standardized[osc_idx_standardized[:,1] < np.percentile(osc_idx_standardized[:,1],25),0]
    osc_yrs = np.vstack((pos_yrs,neg_yrs))
        
    return osc_yrs

def bootstrap_difftest(data,N):
# Initialize a matrix with sampled values for each fpu/raster cell, N is bootstrap sample size
    bstrp_means = np.zeros((data.shape[0],N))
    
    not_nan_array = np.all(~np.isnan(data), axis = 1)    
    
# Calculate median of bootsrap sample N times.
    for j in range(0,N):
        rand_idx_data = np.random.choice(data.shape[1],data.shape[1],replace = True)
# Calculate the median value for the bootsrap sample in question
        sample = data[:,rand_idx_data]      
        bstrp_means[not_nan_array,j] = np.median(sample[not_nan_array,:],axis = 1)

# Calculate how many of the sampled median values are 
# smaller or larger than zero,
    bstrp_test = np.zeros((bstrp_means.shape[0],2))
    bstrp_test[:,0] = np.sum(bstrp_means > 0, axis = 1)
    bstrp_test[:,1] = np.sum(bstrp_means < 0, axis = 1)
# and divide that with the total number of samples. Then substract this value from 1,
# and multiply this value by two (two-sided test) to get the p-value.
    bstrp_test_pval = (1 - np.amax(bstrp_test,axis = 1) / N)*2.0
        
    return bstrp_test_pval


def isolate_osc_anomalies_GGCM(model_list,runtype,irrig_setup,crop,aggregation,oscillation_type,climate_input,aggreg_season, input_path,savepath):

    osc_yrs = import_oscillation_index(oscillation_type, climate_input, aggreg_season)    
# Change directory where crop yield data is imported from based on the irrigation setup:
    if irrig_setup[0] == 'combined':
        input_path = os.path.join(input_path, r'__GGCM_actual_cropland_review1_final\yield')
    else:
        input_path = os.path.join(input_path, r'__GGCM_full_cropland_review1_final\yield') 
    
    
# import list of files with yield information specified by parameters
# irrig_setup, model_list,runtype,irrig_setup,aggregation,crop        
    filelist = import_file_lists(input_path, irrig_setup, model_list,runtype,aggregation,crop,climate_input,aggreg_season)

    
# Initialize size of anomaly table
    N_rows = 548
    pos_anom = np.empty((N_rows,0), float)
    neg_anom = np.empty((N_rows,0), float)
    
# Loop goes through filelist and calculates median yield anomaly of a model ensemble
# (specified by parameters: runtype, irrig_setup, crop, aggregation) anomaly during
# different oscillation phases (specified by osc_yrs);
# p-value of the anomalies is assessed by bootsrapping;
    for file in filelist:
                
        ids, detrended_data = detrend_data(file,input_path)  
        
    # Control for time spans: different models and climate inputs have various time spans
        if 'AgMERRA' in file:
            pos_yrs_temp = osc_yrs[0,osc_yrs[0,:] >= 1982] - 1982
            neg_yrs_temp = osc_yrs[1,osc_yrs[1,:] >= 1982] - 1982
        elif 'Princeton' in file and 'orchid' in file:
            pos_yrs_temp = osc_yrs[0,osc_yrs[0,:] >= 1981] - 1981
            neg_yrs_temp = osc_yrs[1,osc_yrs[1,:] >= 1981] - 1981
        else:
            pos_yrs_temp = osc_yrs[0,osc_yrs[0,:] <= 2005] - 1950
            neg_yrs_temp = osc_yrs[1,osc_yrs[1,:] <= 2005] - 1950
        
        
        pos_anom_new = detrended_data[:,pos_yrs_temp.astype(int)]
        neg_anom_new = detrended_data[:,neg_yrs_temp.astype(int)]
                
        pos_anom = np.hstack((pos_anom,pos_anom_new))
        neg_anom = np.hstack((neg_anom,neg_anom_new))
            
    if len(filelist) > 0:
    
        bstrp_N = 1000
# Calculate the significance of the median anomaly by boottrapping
        pos_sign = bootstrap_difftest(pos_anom,bstrp_N)
        neg_sign = bootstrap_difftest(neg_anom,bstrp_N)       
# Calculate median anomaly of the sample
        
        not_nan_array_pos = np.all(~np.isnan(pos_anom), axis = 1)
        not_nan_array_neg = np.all(~np.isnan(neg_anom), axis = 1)

        pos_anom_median_temp = np.median(pos_anom[not_nan_array_pos],axis = 1)
        neg_anom_median_temp = np.median(neg_anom[not_nan_array_neg],axis = 1)       

        pos_anom_median = np.zeros((pos_anom.shape[0]))*np.nan
        neg_anom_median = np.zeros((neg_anom.shape[0]))*np.nan
        
        pos_anom_median[not_nan_array_pos] = pos_anom_median_temp
        neg_anom_median[not_nan_array_neg] = neg_anom_median_temp

# Combine the results into a single array
        pos_anom_stack = np.hstack((ids[:,np.newaxis],pos_anom_median[:,np.newaxis],pos_sign[:,np.newaxis]))
        neg_anom_stack = np.hstack((ids[:,np.newaxis],neg_anom_median[:,np.newaxis],neg_sign[:,np.newaxis]))
    
# Update path, where the anomaly results are stored for visualization
# Save results with the file name depending on the function parameters
        os.chdir(os.path.join(savepath, irrig_setup[0]+'_'+runtype[0],'anomalies'))

        if runtype[0] == 'default':
            np.savetxt('pos_anom_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'.csv',pos_anom_stack, delimiter=";")
            np.savetxt('neg_anom_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'.csv',neg_anom_stack, delimiter=";")
        if runtype[0] == 'fullharm':
            np.savetxt('pos_anom_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'actfert'+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'.csv',pos_anom_stack, delimiter=";")
            np.savetxt('neg_anom_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'actfert'+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'.csv',neg_anom_stack, delimiter=";")
        if runtype[0] == 'harmnon':
            np.savetxt('pos_anom_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'fullfert'+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'.csv',pos_anom_stack, delimiter=";")
            np.savetxt('neg_anom_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'fullfert'+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'.csv',neg_anom_stack, delimiter=";")


def isolate_anomalies_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, aggreg_season, input_path, savepath):
    for model_main in model_list_main:
        for runtype_main in runtype_list_main:
            for aggregation_main in aggregation_list_main:
                for irrig_setup_main in irrig_setup_list_main:
                    for crop_main in crop_list_main:
                        for oscillation_main in oscillation_list_main:
                            for climate_main in climate_list_main:
                                isolate_osc_anomalies_GGCM(model_main,runtype_main,irrig_setup_main,crop_main,aggregation_main,oscillation_main,climate_main,aggreg_season,input_path,savepath)




# ISOLATE ANOMALIES
# GGCM
aggregation_list_main = [['573']]
irrig_setup_list_main = [['combined']]
crop_list_main = [['whe'],['mai'],['ric'],['soy']]
climate_list_main = [['AgMERRA']]
input_path = r'D:\work\data\modified_crop_data'
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1'


runtype_list_main = [['fullharm']]
model_list_main = [['all_models']]
aggreg_season = 'may_sowing'
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj'],['iod_son_adj']]
isolate_anomalies_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, aggreg_season, input_path, savepath)


model_list_main = [['all_fertilizer_models']]
runtype_list_main = [['fullharm']]
aggreg_season = 'may_sowing'
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj'],['iod_son_adj']]
isolate_anomalies_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, aggreg_season, input_path, savepath)


runtype_list_main = [['harmnon']]
aggreg_season = 'may_sowing'
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj'],['iod_son_adj']]
isolate_anomalies_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, aggreg_season, input_path, savepath)



