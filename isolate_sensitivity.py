import os
import numpy as np
from sklearn.linear_model import RidgeCV

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
#    if 'AgMERRA' in filename and 'orchid' in filename and 'soy' in filename:
#        data = data[:,1:]
# Create an empty array to store detrended crop yield data
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
        

def import_oscillation_index(oscillation_index,climate_input,aggreg_season,enso_idx_info):
# Imports each oscillation index (ENSO, IOD, NAO).
    os.chdir('D:\work\data\oscillation_indices')
        
# Imports the raw oscillation index data based on the oscillation in question
# defined by oscillation_index.

    if enso_idx_info == 'nino34':
        print('nino34')
        enso_raw = np.genfromtxt('nino34.long.anom.data',skip_header = 1,skip_footer = 7)
    elif enso_idx_info == 'jmasst':
        enso_raw = np.genfromtxt('jmasst1868-today.filter-5')

    iod_raw = np.genfromtxt('dmi.long.data',skip_header = 1,skip_footer = 6)
    nao_raw = np.genfromtxt('nao_hurrell_pc_monthly.txt',skip_header = 1)
        
# Select the appropriate years of oscillation index data based on the climate
# forcing of the crop yield data in question.
    if 'AgMERRA' in climate_input[0]:
        enso_years_to_select = np.all([enso_raw[:,0] >= 1980, enso_raw[:,0] <= 2010],axis = 0)
        iod_years_to_select = np.all([iod_raw[:,0] >= 1980, iod_raw[:,0] <= 2010],axis = 0)
        nao_years_to_select = np.all([nao_raw[:,0] >= 1980, nao_raw[:,0] <= 2010],axis = 0)
         
    elif 'Princeton' in climate_input[0]:
        enso_years_to_select = np.all([enso_raw[:,0] >= 1948, enso_raw[:,0] <= 2008],axis = 0)
        iod_years_to_select = np.all([iod_raw[:,0] >= 1948, iod_raw[:,0] <= 2008],axis = 0)
        nao_years_to_select = np.all([nao_raw[:,0] >= 1948, nao_raw[:,0] <= 2008],axis = 0)

    enso_raw = enso_raw[enso_years_to_select,:]     
    iod_raw = iod_raw[iod_years_to_select,:]     
    nao_raw = nao_raw[nao_years_to_select,:]
    
    if 'enso' in oscillation_index[0]:
        osc_raw = enso_raw
    elif 'iod' in oscillation_index[0]:
        osc_raw = iod_raw
    elif 'nao' in oscillation_index[0]:
        osc_raw = nao_raw

# Calculate the seasonal average of the index based on the parameters specified
# as the inputs of the function, i.e. aggreg_season and oscillation_index.         
    elif aggreg_season == 'may_sowing' and 'multiv' in oscillation_index[0]:
        enso_raw_months = np.vstack((enso_raw[2:-2,12],enso_raw[3:-1,1],enso_raw[3:-1,2])).T
        enso_idx = np.mean(enso_raw_months,axis = 1)
        
        iod_raw_months = iod_raw[2:-2,[9,10,11]]
        iod_idx = np.mean(iod_raw_months,axis = 1)
        
        nao_raw_months = np.vstack((nao_raw[2:-2,12],nao_raw[3:-1,1],nao_raw[3:-1,2])).T
        nao_idx = np.mean(nao_raw_months,axis = 1)
        
        osc_idx = np.vstack((enso_idx,iod_idx,nao_idx)).T
        osc_idx_standardized = (osc_idx - np.mean(osc_idx, axis = 0)) / np.std(osc_idx, axis = 0)
        osc_idx_standardized = np.hstack((enso_raw[2:-2,0][:,np.newaxis],osc_idx_standardized))
        
        
    elif aggreg_season == 'annual_harvest' and 'multiv' in oscillation_index[0]:
        
        if 'djf_reg' in oscillation_index[0]:
            enso_raw_months = np.vstack((enso_raw[1:-3,12],enso_raw[2:-2,1],enso_raw[2:-2,2])).T
            iod_raw_months = np.vstack((iod_raw[1:-3,12],iod_raw[2:-2,1],iod_raw[2:-2,2])).T
            nao_raw_months = np.vstack((nao_raw[1:-3,12],nao_raw[2:-2,1],nao_raw[2:-2,2])).T
        elif 'mam' in oscillation_index[0]:
            enso_raw_months = enso_raw[2:-2,[3,4,5]]
            iod_raw_months = iod_raw[2:-2,[3,4,5]]
            nao_raw_months = nao_raw[2:-2,[3,4,5]]
        elif 'jja' in oscillation_index[0]:
            enso_raw_months = enso_raw[2:-2,[6,7,8]]
            iod_raw_months = iod_raw[2:-2,[6,7,8]]
            nao_raw_months = nao_raw[2:-2,[6,7,8]]
        elif 'son' in oscillation_index[0]:
            enso_raw_months = enso_raw[2:-2,[9,10,11]]
            iod_raw_months = iod_raw[2:-2,[9,10,11]]
            nao_raw_months = nao_raw[2:-2,[9,10,11]]
        elif 'djf_plus' in oscillation_index[0]:
            enso_raw_months = np.vstack((enso_raw[2:-2,12],enso_raw[3:-1,1],enso_raw[3:-1,2])).T
            iod_raw_months = np.vstack((iod_raw[2:-2,12],iod_raw[3:-1,1],iod_raw[3:-1,2])).T
            nao_raw_months = np.vstack((nao_raw[2:-2,12],nao_raw[3:-1,1],nao_raw[3:-1,2])).T
        
        enso_idx = np.mean(enso_raw_months,axis = 1)
        iod_idx = np.mean(iod_raw_months,axis = 1)
        nao_idx = np.mean(nao_raw_months,axis = 1)
        
        osc_idx = np.vstack((enso_idx,iod_idx,nao_idx)).T
        osc_idx_standardized = (osc_idx - np.mean(osc_idx, axis = 0)) / np.std(osc_idx, axis = 0)
        osc_idx_standardized = np.hstack((enso_raw[2:-2,0][:,np.newaxis],osc_idx_standardized))

    else:
# Standardize the oscillation index.  
        osc_idx_standardized = (osc_idx - np.mean(osc_idx)) / np.std(osc_idx)
        osc_idx_standardized = np.vstack((osc_raw[2:-2,0],osc_idx_standardized)).T
    
    return osc_idx_standardized

    
def calculate_slopes(data,osc_idx,osc_idx_type,N):

# Initialization of an array where values are saved
    slope = np.zeros((data.shape[0],osc_idx.shape[0]))
    pval = np.zeros((data.shape[0],osc_idx.shape[0]))
    rsquared =  np.zeros((data.shape[0],1))

# If, crop yield data exists, calculate the slope coefficient for each oscillation and FPU
# using regularized ridge regression.
    for k in range(0,slope.shape[0]):
        y = data[k,:]
        if all(~np.isnan(y)):    
            if 'multiv' in osc_idx_type:
                X0 = np.hstack((osc_idx.T,np.ones((osc_idx[1,:][:,np.newaxis].shape))))        
                ridge_model = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]).fit(X0, y)
                coeffs = ridge_model.coef_
                rsquared_temp = ridge_model.score(X0,y)
                
                coeffs_sample = np.empty((0,4))
                for j in range(0,N):
                        rand_idx_data = np.random.choice(data.shape[1],data.shape[1],replace = True)
                        y_sample = y[rand_idx_data]
                        X0_sample = X0[rand_idx_data,:]
                        ridge_model_sample = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]).fit(X0_sample, y_sample)
                        coeffs_sample = np.vstack((coeffs_sample,ridge_model_sample.coef_[np.newaxis,:]))
      
# Calculate how many of the sampled slope values are 
# smaller or larger than zero. 
                bstrp_test = np.zeros((2,coeffs.shape[0])) 
                bstrp_test[0,:] = np.sum(coeffs_sample > 0, axis = 0)
                bstrp_test[1,:] = np.sum(coeffs_sample < 0, axis = 0)
                pvals = (1 - np.amax(bstrp_test,axis = 0) / N)*2.0
# Select the maximum of the array that stores the number of sampled correlations
# larger or smaller than zero, and divide that with the total number of samples.
# Then substract this value from 1, and multiply this value by two (two-sided test) to get the p-value.
                slope[k,:] = coeffs[0:3]
                pval[k,:] = pvals[0:3]
                rsquared[k,0] = rsquared_temp

        else:
            slope[k,0] = np.nan

    return slope, pval, rsquared

def isolate_osc_sensitivity_GGCM(model_list,runtype,irrig_setup,crop,aggregation,oscillation_type,climate_input,aggreg_season,enso_idx_info,input_path,savepath):

# Import information about the oscillations
    osc_idx = import_oscillation_index(oscillation_type,climate_input,aggreg_season,enso_idx_info)

# Change directory where crop yield data is imported from based on the irrigation setup:
    if irrig_setup[0] == 'combined':
        input_path = os.path.join(input_path, r'__GGCM_actual_cropland_review1_final\yield')
    else :
        input_path = os.path.join(input_path, r'__GGCM_full_cropland_review1_final\yield')

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
        osc_idx_temp = osc_idx[:,1:]
        
        if 'Princeton' in file and 'orchid' in file:
            osc_idx_temp = osc_idx[osc_idx[:,0] >= 1981,1:]

# Combine the data for detrended crop yield and oscillation index data for different models.
# SOMETHING WRONG HERE! :)
        if i == 0:
            crop_yield_data = detrended_data
            osc_idx_data = osc_idx_temp
            i = 1
        elif osc_idx_temp.shape[1] > 0:
            osc_idx_data = np.vstack((osc_idx_data,osc_idx_temp))
            crop_yield_data = np.hstack((crop_yield_data,detrended_data))
        else:
            osc_idx_data = np.vstack((osc_idx_data,osc_idx_temp[:,np.newaxis]))
            crop_yield_data = np.hstack((crop_yield_data,detrended_data))                
        
# Checks, if data exists for the specified parameters
    if len(filelist) > 0:
        osc_idx_data = osc_idx_data.T
        print(osc_idx_data.shape)
        print(crop_yield_data.shape)
        
# Update path, where the sensitivity results are stored for visualization
        os.chdir(os.path.join(savepath, irrig_setup[0]+'_'+runtype[0]+'\sensitivity'))
        
# Calculate slope, p-values and r-squared for each oscillation and FPU
        bstrp_N = 1000
        slopes, pvals, rsquared = calculate_slopes(crop_yield_data,osc_idx_data,oscillation_type[0],bstrp_N)
# Combine the results into a arrays
        slope_stack = np.hstack((ids[:,np.newaxis],slopes,pvals))            
        rsquared_stack = np.hstack((ids[:,np.newaxis], rsquared))
            
# Save results with the file name depending on the function parameters
        if runtype[0] == 'default':
            np.savetxt('sensitivity_enso_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,1,4]], delimiter=";")
            np.savetxt('sensitivity_iod_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,2,5]], delimiter=";")
            np.savetxt('sensitivity_nao_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,3,6]], delimiter=";")


        elif runtype[0] == 'fullharm':
            np.savetxt('sensitivity_enso_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'actfert'+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,1,4]], delimiter=";")
            np.savetxt('sensitivity_iod_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'actfert'+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,2,5]], delimiter=";")
            np.savetxt('sensitivity_nao_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'actfert'+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,3,6]], delimiter=";")
            np.savetxt('rsquared_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+runtype[0]+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',rsquared_stack , delimiter=";")


        elif runtype[0] == 'harmnon':
            np.savetxt('sensitivity_enso_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'fullfert'+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,1,4]], delimiter=";")
            np.savetxt('sensitivity_iod_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'fullfert'+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,2,5]], delimiter=";")
            np.savetxt('sensitivity_nao_'+oscillation_type[0]+'_'+crop[0]+'_'+irrig_setup[0]+'_'+'fullfert'+'_'+model_list[0]+'_'+climate_input[0]+'_'+enso_idx_info+'_GGCM_'+aggregation[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,3,6]], delimiter=";")


def isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, aggreg_season, enso_idx_info, input_path, savepath):
    for model_main in model_list_main:
        for runtype_main in runtype_list_main:
            for aggregation_main in aggregation_list_main:
                for irrig_setup_main in irrig_setup_list_main:
                    for crop_main in crop_list_main:
                        for oscillation_main in oscillation_list_main:
                            for climate_main in climate_list_main:
                                isolate_osc_sensitivity_GGCM(model_main,runtype_main,irrig_setup_main,crop_main,aggregation_main,oscillation_main,climate_main,aggreg_season,enso_idx_info,input_path,savepath)




# ISOLATE SENSITIVITY
# Combined, fullharm (Actual scenario).
# List of the crops that are included
crop_list_main = [['mai'],['ric'],['soy'],['whe']]
# Scale of aggregation (here 573 FPUs):
aggregation_list_main = [['573']]
# Path to directory where the crop yield data is stored
input_path = r'D:\work\data\modified_crop_data'
# Path to parent directory, where files are saved
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1'

# List of models that want to be included
model_list_main = [['all_models'],['all_fertilizer_models'],['pdssat'],['epic-boku'],['epic-iiasa'],['gepic'],['papsim'],['pegasus'],\
             ['lpj-guess'],['lpjml'],['cgms-wofost'],['epic-tamu'],['orchidee-crop'],['pepic']]
 # List of GGCMI model configurations included
runtype_list_main = [['fullharm']]
# Irrigation set-up for the data set
irrig_setup_list_main = [['combined']]
# List of climate inputs
climate_list_main = [['AgMERRA']]
# Specify method of analysis, only multivariate ridge regression here.
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)

# Combined, harmnon (fully fertilzed scenario)
model_list_main = [['all_fertilizer_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['harmnon']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)


# firr, fullharm (fully irrigated scenario)
model_list_main = [['all_models']]
irrig_setup_list_main = [['firr']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)


# noirr, fullharm (rainfed scenario)
model_list_main = [['all_models']]
irrig_setup_list_main = [['noirr']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)


# firr, harmnon (fully irrigated and fertilized scenario)
model_list_main = [['all_fertilizer_models']]
irrig_setup_list_main = [['firr']]
runtype_list_main = [['harmnon']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)


# Combined, default (actual irrigation, otherwise fullharm model settings)
model_list_main = [['all_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['default']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)


# Combined, fullharm (actual irrigation, otherwise fullharm model settings)
climate_list_main = [['Princeton']]
model_list_main = [['all_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'jmasst', input_path, savepath)

# Combined, fullharm (actual irrigation, otherwise fullharm model settings), growing season sensitivity
climate_list_main = [['AgMERRA']]
model_list_main = [['all_models']]
irrig_setup_list_main = [['combined']]
runtype_list_main = [['fullharm']]
oscillation_list_main = [['multiv_djf_reg'],['multiv_mam'],['multiv_jja'],['multiv_son'],['multiv_djf_plus']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'annual_harvest', 'jmasst', input_path, savepath)

# Combined, fullharm (actual irrigation, otherwise fullharm model settings), enso index with nino34
model_list_main = [['all_models']]
runtype_list_main = [['fullharm']]
irrig_setup_list_main = [['combined']]
climate_list_main = [['AgMERRA']]
oscillation_list_main = [['multiv']]
isolate_sensitivity_GGCM_main(model_list_main, runtype_list_main, aggregation_list_main, irrig_setup_list_main,crop_list_main, climate_list_main, oscillation_list_main, 'may_sowing', 'nino34', input_path, savepath)






