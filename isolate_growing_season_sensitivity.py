import os
import numpy as np

def isolate_growing_season_sensitivity(alpha,model_list,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,season_list,runtype,temporal_aggreg,input_path,savepath,save):

# List of all the files located in the directory specified by input path.
    file_list = os.listdir(input_path)
    os.chdir(input_path)
    
# Loop through the different potential parameter settings
    for climate in climate_list:
        for model in model_list:    
            for aggregation in aggregation_list:
                for irrig in irrig_setup_list:
                    for crop in crop_list:
                        for osc in oscillation_list:
                            i = 0
                            for season in season_list:
                                for file in file_list:
# Loop through the list of files and select the files that include the information specified by the different parameters
                                    if model in file and aggregation in file and irrig in file and crop in file and osc in file and climate in file and file.endswith('.csv') and \
                                    season in file and temporal_aggreg in file:
# Import the data specified by the file variable
                                        print file
                                        os.chdir(input_path)
                                        data = np.genfromtxt(file,delimiter = ';')
# Create tables that include the sensitivity and related p-value information.
# In the tables each column represents the sensitivity calculated with the index
# for a specific season (order defined in the season_list variable).
                                        if aggregation == '573':
                                            ids = data[:,0]
                                            sens_temp = data[:,1]
                                            pval_temp = data[:,2]
                                            
                                        if i == 0:
                                            sens = sens_temp[:,np.newaxis]
                                            pval = pval_temp[:,np.newaxis]
                                            i = 1
                                        else:    
                                            sens = np.hstack((sens,sens_temp[:,np.newaxis]))
                                            pval = np.hstack((pval,pval_temp[:,np.newaxis]))
# Import tha tabulated data of harvest season
                            os.chdir(r'D:\work\data\crop_calendar_isi_mip')
                            gs_data = np.genfromtxt(crop+'_growing_season.csv',delimiter = ';')
# Put sensitivity as 0, if p-value > alpha
                            sens[pval > alpha] = 0.0
# Initialize arrays for storing the sensitivity values and related p-values for the
# harvest season sensitivity, as well as information about the season.
                            sens_gs = np.zeros((gs_data.shape[0],1))
                            pval_gs = np.zeros((gs_data.shape[0],1))+1
                            gs_info = np.zeros((gs_data.shape[0],1))
# Loop through the FPUs (as specified by row in gs_data).
                            for i in range(0,gs_data.shape[0]):
# Select the harvest season information for each FPU
                                gs_data_temp = np.copy(gs_data[i,1:]).astype(bool)
# Check that the FPU has data about harvest season.
                                if np.any(gs_data_temp):
# Select the sensitivity, pvalue and index number for for the harvest season
# that is most dominant in the FPU in question.
                                    sens_gs[i,0] = sens[i,gs_data_temp]
                                    pval_gs[i,0] = pval[i,gs_data_temp]
                                    gs_info[i,0] = np.argwhere(gs_data_temp)
                        
# Combine information about ids and the other variabls. Then, save the information.
                            if '573' in aggregation:
                                sens_stack = np.hstack((ids[:,np.newaxis],sens_gs,pval_gs))
                                gs_stack = np.hstack((ids[:,np.newaxis],gs_info,pval_gs))
                                                        
                            os.chdir(savepath)
                            np.savetxt('sensitivity_growing_season_'+osc+'_'+crop+'_'+irrig+'_'+runtype[0]+'_'+model+'_'+climate+'_GGCM_'+aggregation+'.csv',sens_stack, delimiter=";")
                            np.savetxt('sensitivity_gs_seasons_'+osc+'_'+crop+'_'+irrig+'_'+runtype[0]+'_'+model+'_'+climate+'_GGCM_'+aggregation+'.csv',gs_stack, delimiter=";") 
                            print ' '


model_list_main = ['all_models']
season_list = ['djf_reg','mam','jja','son','djf_plus']
aggregation_list = ['573']
irrig_setup_list = ['combined']
climate_list = ['AgMERRA']
crop_list = ['mai','ric','soy','whe']
oscillation_list = ['enso','iod','nao']
runtype_list = ['fullharm']
temporal_aggreg = 'annual_aggreg'
alpha = 0.1
save = 1
input_path = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity'
savepath = r'D:\work\research\crops_and_oscillations\results_v8\combined_fullharm\sensitivity_gs_figs'

isolate_growing_season_sensitivity(alpha,model_list_main,aggregation_list,irrig_setup_list,crop_list,oscillation_list,climate_list,season_list,runtype_list,temporal_aggreg,input_path,savepath,save)



