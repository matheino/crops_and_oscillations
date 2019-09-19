import os
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as dg
import scipy.stats as stat
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

        

def import_oscillation_index(oscillation_index,climate_input,aggreg_season):
# Imports each oscillation index (ENSO, IOD, NAO).
    os.chdir('D:\work\data\oscillation_indices')
        
# Imports the raw oscillation index data based on the oscillation in question
# defined by oscillation_index.

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
        
    if aggreg_season == 'may_sowing' and 'multiv' in oscillation_index[0]:
        enso_raw_months = np.vstack((enso_raw[2:-2,12],enso_raw[3:-1,1],enso_raw[3:-1,2])).T
        enso_idx = np.mean(enso_raw_months,axis = 1)
        iod_raw_months = iod_raw[2:-2,[9,10,11]]
        iod_idx = np.mean(iod_raw_months,axis = 1)
        nao_raw_months = np.vstack((nao_raw[2:-2,12],nao_raw[3:-1,1],nao_raw[3:-1,2])).T
        nao_idx = np.mean(nao_raw_months,axis = 1)
        
        osc_idx = np.vstack((enso_idx,iod_idx,nao_idx)).T
        osc_idx_standardized = (osc_idx - np.mean(osc_idx, axis = 0)) / np.std(osc_idx, axis = 0)
        osc_idx_standardized = np.hstack((enso_raw[2:-2,0][:,np.newaxis],osc_idx_standardized))

    else:
# Standardize the oscillation index.  
        osc_idx_standardized = (osc_idx - np.mean(osc_idx)) / np.std(osc_idx)
        osc_idx_standardized = np.vstack((osc_raw[2:-2,0],osc_idx_standardized)).T
    
    return osc_idx_standardized.T

    
def import_clim_data(input_path,crop,dtype):
    os.chdir(input_path)
        
    crop_list_temp = {'mai':'maize',
                      'ric':'rice',
                      'soy':'soybeans',
                      'whe':'wheat'}
    
    crop_temp = crop_list_temp[crop[0]]
    
    if dtype == 'Temperature':
        print('Temperature')
        clim_data = np.genfromtxt('fpu_573_temperature_'+crop_temp+'.csv', delimiter = ';')
    elif dtype == 'Soil_moisture':
        print('Soil_moisture')
        clim_data = np.genfromtxt('fpu_573_soil_moisture_'+crop_temp+'.csv', delimiter = ';')
        
    fpu_ids = clim_data[:,0]
    clim_data = clim_data[:,1:]
    
    
    detrended_clim_data = np.zeros((clim_data.shape[0],clim_data.shape[1]-2))*np.nan

# Loop goes through each year of data, starting from first year + 1,
# and calculates the proportional and absolute deviation from the 5-year 
# (3-year at both ends of the time series) mean.
    for i in range(0, clim_data.shape[1]-2):
        
        if i == 0 or i == range(0, clim_data.shape[1]-1)[-2]:
            yrly_indices = np.linspace(i,i+2,3).astype(int) # Indices for 3 years surrounding the year in question.
        else:
            yrly_indices = np.linspace(i-1,i+3,5).astype(int) # Indices for 5 years surrounding the year in question.

        baseline_mean = np.mean(clim_data[:,yrly_indices],axis=1)
        yrly_val = clim_data[:,i+1]
        
# To avoid dividing with zero, or having NaN values in the calculation
# all nan values are ignored in the calculations and zero values in 
# the baseline_mean
        check_for_nans = np.hstack((~np.isnan(baseline_mean[:,np.newaxis]),~np.isnan(yrly_val[:,np.newaxis]), baseline_mean[:,np.newaxis] != 0))
        not_nan_array = np.all(check_for_nans, axis = 1)                
        detrended_clim_data_temp = (yrly_val[not_nan_array] - baseline_mean[not_nan_array])
        detrended_clim_data[not_nan_array,i] = detrended_clim_data_temp
# If all data in an array is zero, change values into nan.
    all_data_zero = np.all(detrended_clim_data == 0, axis = 1)
    detrended_clim_data[all_data_zero,:] = np.nan
    
    return fpu_ids, detrended_clim_data

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


def isolate_clim_corr(crop,climate_input,oscillation_type,input_path,savepath, dtype, aggreg_season):

# Import information about the oscillations
    osc_idx = import_oscillation_index(oscillation_type,climate_input,aggreg_season)
    
    ids, clim_data = import_clim_data(input_path,crop,dtype[0])
    
    print(ids.shape)
    print(osc_idx.shape)
    print(clim_data.shape)
    
    
# Calculate p-value (H0: no linear relationship) for each FPU
    corr, pvals, rsquared = calculate_slopes(clim_data, osc_idx[1:,:], oscillation_type, 1000)
# Combine the results into a single array
    slope_stack = np.hstack((ids[:,np.newaxis],corr,pvals))

# Update path, where the sensitivity results are stored for visualization
    os.chdir(savepath)
# Save results with the file name depending on the function parameters
    np.savetxt('clim_corr_enso_'+oscillation_type[0]+'_'+dtype[0]+'_'+crop[0]+'_'+climate_input[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,1,4]], delimiter=";")
    np.savetxt('clim_corr_iod_'+oscillation_type[0]+'_'+dtype[0]+'_'+crop[0]+'_'+climate_input[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,2,5]], delimiter=";")
    np.savetxt('clim_corr_nao_'+oscillation_type[0]+'_'+dtype[0]+'_'+crop[0]+'_'+climate_input[0]+'_'+aggreg_season+'.csv',slope_stack[:,[0,3,6]], delimiter=";")



def isolate_clim_corr_main(crop_list_main, climate_list_main, oscillation_list_main, input_path, savepath, dtype_list_main, aggreg_season):
    for crop_main in crop_list_main:
        for oscillation_main in oscillation_list_main:
            for climate_main in climate_list_main:
                for dtype in dtype_list_main:
                    isolate_clim_corr(crop_main, climate_main, oscillation_main, input_path, savepath, dtype, aggreg_season)




# ISOLATE SENSITIVITY
# Combined, fullharm (Actual scenario).
# List of the crops that are included
crop_list_main = [['mai'],['soy'],['ric'],['whe']]
# Path to directory where the crop yield data is stored
input_path = r'D:\work\data\modified_clim_data'
# Path to parent directory, where files are saved
savepath = r'D:\work\research\crops_and_oscillations\results_review_v1\clim_corr'
# List of GGCMI model configurations included
climate_list_main = [['AgMERRA']]
# List of the oscillation indices and their settings:
oscillation_list_main = [['multiv']]


dtype_list_main = [['Soil_moisture'],['Temperature']]
aggreg_season = 'may_sowing'

isolate_clim_corr_main(crop_list_main, climate_list_main, oscillation_list_main, input_path, savepath, dtype_list_main, aggreg_season)

