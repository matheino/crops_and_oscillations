#!/usr/bin/env python
# -*- coding: utf-8 -*- import pylab
import os
import numpy as np
import matplotlib.pyplot as plt

def obtain_data(file_name, path, alpha):
    
    os.chdir(path)

    data = np.genfromtxt(file_name,delimiter = ';') 
    
    sens = data[:,1]
    pval = data[:,2]
#    sens[pval > alpha] = np.nan
    
    return sens,pval
    
def create_label(crop,pval,val):
# Create a label for the figure, based on the average differencen and the 
# result of its significance test.

    if pval < 0.001:
        pval_string = '***'
    elif pval < 0.01:
        pval_string = '**'
    elif pval < 0.1:
        pval_string = '*'
    else:
        pval_string = ''

    label = 'Avg: '+str(round(val,1))+'%'+pval_string

    return label

def bootstrap_difftest(data,N):
# Initialize a matrix with sampled values for each fpu/raster cell, N is bootstrap sample size
    bstrp_means = np.zeros((N))
    
    for j in range(0,N):
        rand_idx_data = np.random.choice(data.shape[0],data.shape[0],replace = True)
# Calculate median of bootsrap sample N times.
        sample = data[rand_idx_data]
        bstrp_means[j] = np.mean(sample)
# Calculate how many of the sampled mean values are 
# smaller or larger than zero,
    bstrp_test = np.zeros((bstrp_means.shape[0],2))
    bstrp_test[:,0] = np.sum(bstrp_means > 0)
    bstrp_test[:,1] = np.sum(bstrp_means < 0)
# and divide that with the total number of samples. Then substract this value from 1,
# and multiply this value by two (two-sided test) to get the p-value.
    bstrp_test_pval = (1 - np.amax(bstrp_test) / N)*2.0
    
    return bstrp_test_pval, bstrp_means

def plot_histograms_comparison(alpha,model_list,aggregation_list,crop_list,oscillation_list,climate_list,setup1,setup2,input_path_setup1,input_path_setup2,savepath,hist_xlim):
    
    file_list1 = os.listdir(input_path_setup1)
    file_list2 = []
    
    for file in file_list1:
        file_list2.append(file.replace(setup1,setup2))
    
    i = 0
    j = 0
    
    for model in model_list:
        for climate in climate_list:
            for aggregation in aggregation_list:              
                for crop in crop_list:
                    sens1 = []
                    sens2 = []
                    pval1 = []
                    pval2 = []
# Loop through the the oscillations, as the sensitivity results for all the oscillations
# are aggregated in this analysis.
                    for osc in oscillation_list:
# Loop through all the file names and check, that the input parameters match
# for the scanerios being compared.
                        for file1, file2 in zip(file_list1,file_list2):
                            if aggregation in file1 and crop in file1 and osc in file1 and model in file1 and climate in file1 and setup1 in file1:
                                print file1
                                sens, pval = obtain_data(file1,input_path_setup1,alpha)
# Save the sensitivity results for all the oscillations into sens1 and sens2 variables,
# and the corresponding p-values to the pval1 and pval2 variables.
                                if i == 0:
                                    sens1 = sens
                                    pval1 = pval
                                    i = 1
                                else:
                                    sens1 = np.hstack((sens1,sens))
                                    pval1 = np.hstack((pval1,pval))

                                                                    
                            if aggregation in file2 and crop in file2 and osc in file2 and model in file2 and climate in file2 and setup2 in file2:
                                print file2
                                sens, pval = obtain_data(file2,input_path_setup2,alpha)
                                if j == 0:
                                    sens2 = sens
                                    pval2 = pval
                                    j = 1
                                else:
                                    sens2 = np.hstack((sens2,sens))
                                    pval2 = np.hstack((pval2,pval))

# sens_data (pval_data) includes sensitivity (p-value) results for all the oscillations for both scenarios.             
                    sens_data = np.hstack((sens1[:,np.newaxis],sens2[:,np.newaxis]))
                    pval_data = np.hstack((pval1[:,np.newaxis],pval2[:,np.newaxis]))
# Remove thos FPUS that don't grow the crop in question or where both scenarios show non-significant sensitivity
# Remove also those FPUs where either scenario has a nan value for the sensitivity (can happen e.g. if a model has all zero for yield).
                    keep_these_indices_pval = np.any((pval_data <= alpha),axis = 1)
                    sens_data = sens_data[keep_these_indices_pval,:]
                    keep_these_indices_not_nan = np.all(~np.isnan(sens_data),axis = 1)                      
                    sens_data = sens_data[keep_these_indices_not_nan,:]
                    x = sens_data[:,0]
                    y = sens_data[:,1]
# Calculate the relative difference in absolute sensitivity between the scenarios
                    diff = (np.abs(y)-np.abs(x))/np.mean(np.abs(x))
# Relative value from fraction to percentage
                    diff = diff*100
# Calculate the mean relative difference in absolute sensitivity between the scenarios
                    mean_diff = np.mean(diff)
# Assess the significance with of the mean calculated above (H0: mean is zero)
# with bootstrapping the difference values
                    N = 1000
                    pval_diff, means_diff = bootstrap_difftest(diff,N)   
                    
# Create a histograms of the bootstrapped mean values and include the 
# mean difference as well as the related p-value to the figure label as well.
                    fig, ax = plt.subplots()
                    label  = create_label(crop,pval_diff,mean_diff)
                    weights = np.ones(means_diff.shape)/float(N)
                    hist_xlim = float(hist_xlim)
                    hist_bins = np.arange(-hist_xlim, hist_xlim+hist_xlim/50, hist_xlim/50)

                    ax.hist(means_diff,bins=hist_bins,alpha = 0.85,color='blue',weights=weights)
# Set histogram parameters.
                    labels= []
                    legend_location = 'upper left'                    
                    leg = ax.legend(labels,frameon = False,loc = legend_location,fontsize = 20)
                    leg._legend_box.align = "left"
                    leg.set_title(title = label,prop={'size':20})

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlim(left=-hist_xlim,right=hist_xlim)
                    ax.set_ylim(top = 0.37)
                    plt.yticks(np.arange(0.1,0.32,0.1))
                    ax.tick_params(axis='both', labelsize = 17)
                    
                    print pval_diff
# Set legend information.
                    if crop == 'whe' and setup2 == 'noirr':
                        ax.set_xlabel('Difference in yield deviation (%)\nper unit index change',fontsize = 17)
                        ax.set_ylabel('Frequency',fontsize = 17)
                    
                    if crop != 'whe':
                        ax.axes.xaxis.set_ticklabels([])
                        
                    if setup2 != 'noirr':
                        ax.axes.yaxis.set_ticklabels([])
# Save figure.
                    os.chdir(savepath)
#                    print crop
                    fig_name = 'histogram_'+crop+'_'+setup1+'_vs_'+setup2+'.png'
                    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
#                    plt.title(title, loc = 'center',fontsize=20)
                    plt.show()
                    

# HISTOGRAMS
climate_list = ['AgMERRA']
aggregation_list = ['573']
crop_list = ['mai','ric','soy','whe']
oscillation_list = ['enso_djf_adj','iod_son_adj','nao_djf_adj']
alpha = 0.1
save = 1
savepath = r'D:\work\research\crops_and_oscillations\results_v8\scenario_comparisons' 

model_list_main = ['all_models']
# combined vs noirr
input_path1 = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_v8\noirr_fullharm\sensitivity'            
plot_histograms_comparison(alpha,model_list_main,aggregation_list,crop_list,oscillation_list,climate_list,'combined','noirr',input_path1,input_path2,savepath,149)          

# combined vs firr
input_path1 = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_v8\firr_fullharm\sensitivity'
plot_histograms_comparison(alpha,model_list_main,aggregation_list,crop_list,oscillation_list,climate_list,'combined','firr',input_path1,input_path2,savepath,74)          


model_list_main = ['all_fertilizer_models']
# combined: actual vs. unlimited fertilizers
input_path1 = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_v8\combined_harmnon\sensitivity'
plot_histograms_comparison(alpha,model_list_main,aggregation_list,crop_list,oscillation_list,climate_list,'actfert','fullfert',input_path1,input_path2,savepath,34)          

# combined, actual vs unlimited irrigation+fertilizers
input_path1 = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_v8\firr_harmnon\sensitivity'
plot_histograms_comparison(alpha,model_list_main,aggregation_list,crop_list,oscillation_list,climate_list,'combined_actfert','firr_fullfert',input_path1,input_path2,savepath,74)          

model_list_main = ['all_models']
# combined : rainfed vs full irrigation
input_path1 = r'D:\work\research\crops_and_oscillations\results_v8\firr_fullharm\sensitivity'
input_path2 = r'D:\work\research\crops_and_oscillations\results_v8\noirr_fullharm\sensitivity'
plot_histograms_comparison(alpha,model_list_main,aggregation_list,crop_list,oscillation_list,climate_list,'firr_actfert','noirr_actfert',input_path1,input_path2,savepath,249)          


