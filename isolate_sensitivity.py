import os
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as dg
import scipy.stats as stat

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
    
    print filenames
    
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

# Calculate the seasonal average of the index based on the parameters specified
# as the inputs of the function, i.e. aggreg_season and oscillation_index.
    if aggreg_season == 'december_aggreg' and 'djf_adj' in oscillation_index[0]:
        osc_raw_months = np.vstack((osc_raw[2:-2,12],osc_raw[3:-1,1],osc_raw[3:-1,2])).T
        osc_idx = np.mean(osc_raw_months,axis = 1)
    elif aggreg_season == 'september_aggreg' and 'son_adj' in oscillation_index[0]:
        osc_raw_months = osc_raw[2:-2,[9,10,11]]
        osc_idx = np.mean(osc_raw_months,axis = 1)
    elif aggreg_season == 'annual_aggreg':
        if 'djf_reg' in oscillation_index[0]:
            osc_raw_months = np.vstack((osc_raw[1:-3,12],osc_raw[2:-2,1],osc_raw[2:-2,2])).T
            osc_idx = np.mean(osc_raw_months,axis = 1)
        elif 'mam' in oscillation_index[0]:
            osc_raw_months = osc_raw[2:-2,[3,4,5]]
            osc_idx = np.mean(osc_raw_months,axis = 1)
        elif 'jja' in oscillation_index[0]:
            osc_raw_months = osc_raw[2:-2,[6,7,8]]
            osc_idx = np.mean(osc_raw_months,axis = 1)
        elif 'son' in oscillation_index[0]:
            osc_raw_months = osc_raw[2:-2,[9,10,11]]
            osc_idx = np.mean(osc_raw_months,axis = 1)
        elif 'djf_plus' in oscillation_index[0]:
            osc_raw_months = np.vstack((osc_raw[2:-2,12],osc_raw[3:-1,1],osc_raw[3:-1,2])).T
            osc_idx = np.mean(osc_raw_months,axis = 1)

# Standardize the oscillation index.  
    osc_idx_standardized = (osc_idx - np.mean(osc_idx)) / np.std(osc_idx)
# Add the corresponding year to the standardized oscillation index array,
# and remove the oscillation index data at the tails, where no crop yield data exist
    osc_idx_standardized = np.vstack((osc_raw[2:-2,0],osc_idx_standardized)).T
    
    return osc_idx_standardized

    
def calculate_slope(data,osc_idx):
# Initialization of an array where slope values are saved
    slope = np.zeros((data.shape[0],1))
# Initialize an array, to save the p-value of
# harvey-collier test for linearity
    linearity_pval = np.zeros((data.shape[0],1))*np.nan
# If, crop yield data exists, calculate the slope coefficient for each FPU
# with ordinary linear least squareas regression
    for k in range(0,slope.shape[0]):
       x = osc_idx[0,:]
       y = data[k,:]
       if all(~np.isnan(y)):
           X = np.hstack((x[:,np.newaxis],np.ones((x[:,np.newaxis].shape))))
           lm_model = sm.OLS(y,X).fit()
           coeffs = lm_model.params
           slope[k,0] = coeffs[0]
           linearity_pval[k,0] = dg.linear_harvey_collier(lm_model).pvalue

       else:
           slope[k,0] = np.nan      
           
#    print np.sum(linearity_pval>0.1).astype(float)/np.sum(~np.isnan(linearity_pval)).astype(float)
    
    return slope[:,0]
    
def pval_pearson_correlation_bootstrap(data,osc_idx,N):
# Initialization of matrix with sampled values for each fpu, N is bootstrap sample size.
    bstrp_test = np.zeros((data.shape[0],2))

# Create arrays data and bootsrap sample arrays for FPUs that have data for all years.
    not_nan_array = np.all(~np.isnan(data), axis = 1) 
    data_temp = data[not_nan_array,:]
    bstrp_corrcoef_temp = np.zeros((data_temp.shape[0],N))
    
# For loop calculates Pearson's correlation for N bootstrap samples
    for j in range(0,N):
# Randomly select a sample (with replacement) of the original data
        rand_idx_data = np.random.choice(data.shape[1],data.shape[1],replace = True)
        data_sample = data_temp[:,rand_idx_data]
        osc_idx_sample = osc_idx[0,rand_idx_data]
# Calculate Pearson's correlation for each FPU and randomly selected sample,
# and save the information into bstrp_corrcoef_temp, which is initialized above
        for k in range(0,data_sample.shape[0]):
            y = osc_idx_sample.T
            x = data_sample[k,:].T
            coeff,pval = stat.pearsonr(x,y)
            bstrp_corrcoef_temp[k,j] = coeff

# Calculate how many of the sampled Pearson's correlation values are 
# smaller or larger than zero
    bstrp_test_temp = np.zeros((bstrp_corrcoef_temp.shape[0],2)) 
    bstrp_test_temp[:,0] = np.sum(bstrp_corrcoef_temp > 0, axis = 1)
    bstrp_test_temp[:,1] = np.sum(bstrp_corrcoef_temp < 0, axis = 1)

# Allocate the bootrapped Pearson's correlation values to the correct places (FPUs)
# into the bstrp_corrcoef variable
    bstrp_test[not_nan_array,:] = bstrp_test_temp

# Select the maximum of the array that stores the number of sampled correlations
# larger or smaller than zero, and divide that with the total number of samples.
# Then substract this value from 1, and multiply this value by two (two-sided test) to get the p-value.
    bstrp_test_pval = (1 - np.amax(bstrp_test,axis = 1) / N)*2.0
        
    return bstrp_test_pval



def isolate_osc_sensitivity_GGCM(model_list,runtype,irrig_setup,crop,aggregation,oscillation_type,climate_input,aggreg_season,input_path,savepath):

# Import information about the oscillations
    osc_idx = import_oscillation_index(oscillation_type,climate_input,aggreg_season)

# Change directory where crop yield data is imported from based on the irrigation setup:
    if irrig_setup[0] == 'combined':
        input_path = os.path.join(input_path, r'_GGCM_actual_cropland_final\yield')
    else:
        input_path = os.path.join(input_path, r'_GGCM_full_cropland_final\yield')    
# Import list of files with yield information specified by parameters, i.e.
# path to input directory, irrigation setup, models, model configuration, level of aggregation (FPUs),
# crop, climate forcing, and method of temporal aggregation, respectively.    
    filelist = import_file_lists(input_path, irrig_setup, model_list,runtype,aggregation,crop,climate_input,aggreg_season)
    
# Loop goes through filelist and calculates crop yield sensitivity to the oscilaltion index, and then exports the
# information to the folder specified by savepath as well as irrigation and model setup.
    i = 0
    for file in filelist:
# Import FPU id information and detrended crop yield data
        ids, detrended_data = detrend_data(file,input_path)
        
# Control for time spans: models and climate inputs have different amounts of
# simulated data
        osc_idx_temp = osc_idx[:,1]
        if 'Princeton' in file and 'orchid' in file:
            osc_idx_temp = osc_idx[osc_idx[:,0] >= 1981,1]

# Combine the data for detrended crop yield and oscillation index data for different models.
        if i == 0:
            osc_idx_data = osc_idx_temp[:,np.newaxis]
            crop_yield_data = detrended_data
            i = 1
        else:
            osc_idx_data = np.vstack((osc_idx_data,osc_idx_temp[:,np.newaxis]))
            crop_yield_data = np.hstack((crop_yield_data,detrended_data))                
    
# Checks, if data exists for the specified parameters
    if len(filelist) > 0:
        osc_idx_data = osc_idx_data.T
        
        print osc_idx_data.shape
        print crop_yield_data.shape
# Calculate slope values foe each FPU
        slopes = calculate_slope(crop_yield_data,osc_idx_data)

# Calculate p-value (H0: no linear relationship) for each FPU        
        pvals = pval_pearson_correlation_bootstrap(crop_yield_data,osc_idx_data,1000)
# Combine the results into a single array
        slope_stack = np.hstack((ids[:,np.newaxis],slopes[:,np.newaxis],pvals[:,np.newaxis]))

# Update path, where the sensitivity results are stored for visualization
        os.chdir(os.path.join(savepath, irrig_setup[0]+'_'+runtype[0]+'\sensitivity'))
# Save results with the file name depending on the function parameters
        if runtype[0] == 'default':
            np.savetxt('sensitivity_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack, delimiter=";")
        if runtype[0] == 'fullharm':
            np.savetxt('sensitivity_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'actfert'+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack, delimiter=";")
        if runtype[0] == 'harmnon':
            np.savetxt('sensitivity_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'fullfert'+'_'+model_list[0]+'_'+climate_input[0]+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack, delimiter=";")

def isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, aggreg_season, input_path, savepath):
    for model_main in model_list_main:
        for runtype_main in runtype_list_main:
            for aggregation_main in aggregation_list_main:
                for irrig_setup_main in irrig_setup_list_main:
                    for crop_main in crop_list_main:
                        for oscillation_main in oscillation_list_main:
                            for climate_main in climate_list_main:
                                isolate_osc_sensitivity_GGCM(model_main,runtype_main,irrig_setup_main,crop_main,aggregation_main,oscillation_main,climate_main,aggreg_season,input_path,savepath)




# ISOLATE SENSITIVITY
# Combined, fullharm (Actual scenario).
# List of models that want to be included
model_list_main = [['all_models'],['all_fertilizer_models'],['pdssat'],['epic-boku'],['epic-iiasa'],['gepic'],['papsim'],['pegasus'],\
             ['lpj-guess'],['lpjml'],['cgms-wofost'],['epic-tamu'],['orchidee-crop'],['pepic']]
# List of the crops that are included
crop_list_main = [['mai'],['soy'],['ric'],['whe']]
# Scale of aggregation (here 573 FPUs):
aggregation_list_main = [['573']]
# Path to directory where the crop yield data is stored
input_path = r'D:\work\data\modified_crop_data'
# Path to parent directory, where files are saved
savepath = r'D:\work\research\crops_and_oscillations\results_v8'
# List of GGCMI model configurations included
runtype_list_main = [['fullharm']]
# Irrigation set-up for the data set
irrig_setup_list_main = [['combined']]
# List of climate inputs
climate_list_main = [['AgMERRA']]
# List of the oscillation indices and their settings:
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)

# Combined, harmnon (fully fertilzed scenario)
model_list_main = [['all_fertilizer_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['harmnon']]
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)


# firr, fullharm (fully irrigated scenario)
model_list_main = [['all_models']]
irrig_setup_list_main = [['firr']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)


# noirr, fullharm (rainfed scenario)
model_list_main = [['all_models']]
irrig_setup_list_main = [['noirr']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)


# firr, harmnon (fully irrigated and fertilized scenario)
model_list_main = [['all_fertilizer_models']]
irrig_setup_list_main = [['firr']]
runtype_list_main = [['harmnon']]
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)


# Combined, default (actual irrigation, otherwise default model settings)
model_list_main = [['all_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['default']]
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)


# Combined, fullharm (actual irrigation, otherwise default model settings)
climate_list_main = [['Princeton']]
model_list_main = [['all_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['iod_son_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'september_aggreg', input_path, savepath)
oscillation_list_main = [['enso_djf_adj'],['nao_djf_adj']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'december_aggreg', input_path, savepath)

# Combined, fullharm (actual irrigation, otherwise default model settings)
climate_list_main = [['AgMERRA']]
model_list_main = [['all_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['enso_djf_reg'],['enso_mam'],['enso_jja'],['enso_son'],['enso_djf_plus'],['iod_djf_reg'],['iod_mam'],['iod_jja'],['iod_son'],['iod_djf_plus'],['nao_djf_reg'],['nao_mam'],['nao_jja'],['nao_son'],['nao_djf_plus']]

isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'annual_aggreg', input_path, savepath)








