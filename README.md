# crops_and_oscillations
Scripts for analysing the relationship between climate oscillations and crop yields at global scale.

Each Python script file is stand alone. Brief descriptions of what the script files do:

- combine_irrigated_and_rainfed_yields.py: combines irrigated and rainfed yields for each FPU, by calculating a harvested area (based on MIRCA2000 data) weighted mean yield; exports the resulting yield information as a csv file.

- compare_managment_scenarios_aggreg.py: compares sensitivity between two management scenarios through all FPUs and oscillations; calculates average sensitivity difference and the corresponding statistical significance with a bootsrap method; exports the plotted data as a png file.

- compare_managment_scenarios_spatial.py: calculates sensitivity difference for each FPU and plots the result on a map; exports the plotted maps as a png file.

- import_AgMIP_GGCM_data.py: imports GGCMI data and aggregates it to FPU level; exports the data as a csv file.

- import_growing_seasons.py: imports GGCMI growing season data and calculates the dominant harvest season for each FPU; result exported as a csv file.

- isolate_anomalies.py: calculates the median crop yield anomaly during the strong oscillation phases, and calculates the related satistical significance; exports the information as a csv file.

- isolate_growing_season_sensitivity.py: assess crop yield sensitivity to ENSO, IOD and NAO during the growing season; exports the plotted maps as a png file.

- isolate_sensitivity.py: calculates the sensitivity of maize, rice, soybean and wheat yield to ENSO, IOD and NAO; exports the results as a csv file.

- sens_anom_models_combination.py: looks into how well the different GGCMs agree with the ensemble results; also studies whether the results from sensitivity analysis and anomaly analysis agree with eachother; exports the plotted maps as a png file.

- visualize_anomaly_results.py: plots anomaly results on a map, and calculates a global aggregate table; exports the plotted maps as a png file.

- visualize_harvest_season.py: plots the harvest season for different areas on a global map; exports the plotted maps as a png file.

- visualize_min_max_median_sensitivity.py: visualize the agreement of individual models to the ensemble results; also, looks into how the maximum, minimum and median sensitivity of the individual models; exports the plotted maps as a png file.

- visualize_sensitivity_results.py: plots sensitivity results for different areas on a global map; exports the plotted maps as a png file.
