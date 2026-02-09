import numpy as np
import os
import xarray as xr
import sys
import pickle as pkl

cmipTable = {
    "pr": "Amon",
    "psl": "Amon",
    "tas": "Amon",
    "zmta": "Amon",
    "tos": "Omon",
    "siconc": "OImon",
    "monmaxpr": "Aday",
    "monmaxtasmax": "Aday",
    "monmintasmin": "Aday",
    "mrso": "Lmon"
}
cmipVar = {
    "pr": "pr",
    "psl": "psl",
    "tas": "tas",
    "zmta": "ta",
    "tos": "tos",
    "siconc": "siconc",
    "monmaxpr": "pr",
    "monmaxtasmax": "tasmax",
    "monmintasmin": "tasmin",
    "mrso": "mrso"
}

def global_mean(data,lats):
    # area weighting coefficients
    sq=np.cos(np.pi*lats[:,None]/180.)
    return (data*sq).mean(axis=(-1,-2))/((data*0+1.)*sq).mean(axis=(-1,-2))

def custom_masked_array(da):
    masked_array=da.to_masked_array()
    #update mask to exclude nans and big numbers for sure
    good_mask=np.logical_and(np.logical_not(masked_array.mask),np.abs(masked_array.data)<=1e+16)
    masked_array=np.ma.masked_array(masked_array.data,mask=np.logical_not(good_mask))
    return masked_array


def single_func(ds,var_short):
    """
    Function to compute single-item metrics.
    This is a placeholder function; replace with actual logic as needed.
    """
    data=custom_masked_array(ds[var_short])
    M=data.shape[0]
    
    ### This is a shortcut to compute anomalies without using time variable in the dataset. Assuming the dimensions are (time, ...) and time in months
    data_anomalies=data-np.concatenate(np.ma.stack(np.split(data,M//12,axis=0),axis=0).mean(axis=0,keepdims=True).repeat(M//12,axis=0),axis=0)

    lats=ds['lat'].to_numpy()
    res={}
    res['global_mean']=global_mean(data,lats=lats)  
    res['global_mean_anomalies']=global_mean(data_anomalies,lats=lats)
    return res


def make_comparison(method_folders, method_names,variable, tier):
    evaluation_subfolder=f'Tier{tier}'
    id='ForceSMIP'
    single_items_estimated = {}

    for method_path, method in zip(method_folders,method_names):
        method_eval_path=os.path.join(method_path,evaluation_subfolder)
        print(f'Processing {method_eval_path}',flush=True)
        # Check if the directory exists
        if os.path.isdir(method_eval_path):               
            method_eval_files= sorted([f for f in os.listdir(method_eval_path) if f.startswith(variable + '_')])
            # Check if there are files inside
            if len(method_eval_files)>0:
                try:
                    tmp_dict_list_single = [] 
                    for file in method_eval_files:
                        forced_response_estimate_ds=xr.open_dataset(os.path.join(method_eval_path, file),decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
                        
                        # We have a dictionary of pairwise metrics for each member of the evaluation ensemble
                        if cmipVar[variable] in forced_response_estimate_ds.data_vars:
                            tmp_dict_list_single.append(single_func(forced_response_estimate_ds,cmipVar[variable]))
                        else:
                            tmp_dict_list_single.append(single_func(forced_response_estimate_ds,variable))


                        sys.stdout.flush()
                        #dataset.close()
                        forced_response_estimate_ds.close()
                    
                    # Transform the list of dictionaries into a dictionary of arrays
                    single_items_estimated[method]={}
                    single_items_estimated[method][id] = {}
                    for key in tmp_dict_list_single[0].keys():
                        single_items_estimated[method][id][key] = np.stack([d[key] for d in tmp_dict_list_single],axis=0)
                except KeyError:
                    print(f'CustomWarning: Wrong var names in {method_eval_path}',flush=True)
            else:
                print(f"CustomWarning: No files found in {method_eval_path}")
        else:
            print(f"CustomWarning: No directory {method_eval_path}")

    return single_items_estimated


def make_comparison_original(datasets_folder, variable, tier):
    evaluation_subfolder=f'Evaluation-Tier{tier}'
    id='ForceSMIP'

    single_items_original_data={}
        
    evaluation_path=os.path.join(datasets_folder,id,evaluation_subfolder,cmipTable[variable],variable)
    print(f'Processing {evaluation_path}',flush=True)
    evaluation_files= sorted(os.listdir(evaluation_path))
    tmp_dict_list_single = []
    for file in evaluation_files:
        evaluation_data_ds=xr.open_dataset(os.path.join(evaluation_path, file),decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
        #dataset=Dataset(os.path.join(evaluation_path, file))
        #evaluation_data=dataset.variables[cmipVar[variable]][:]
        tmp_dict_list_single.append(single_func(evaluation_data_ds,cmipVar[variable]))
        #dataset.close()
        evaluation_data_ds.close()
        
    # Transform the list of dictionaries into a dictionary of arrays
    single_items_original_data[id] = {}
    for key in tmp_dict_list_single[0].keys():
        single_items_original_data[id][key] = np.stack([d[key] for d in tmp_dict_list_single],axis=0)        


    return single_items_original_data

if __name__ == "__main__":
    datasets_folder='/media/disk/databases/'
    method_folders=sys.argv[1].split(',')
    method_names=sys.argv[2].split(',')
    variable=sys.argv[3]
    tier=sys.argv[4]

    # Call the function to make the comparison
    print(f"Comparing {variable} for tier {tier}...")
    print(f"Datasets folder: {datasets_folder}")
    print(f"Method folders: {method_folders}")
    single_items_estimated = make_comparison(method_folders, method_names, variable, tier)
    single_items_original_data = make_comparison_original(datasets_folder, variable, tier)
    ids=['ForceSMIP']

    comparison_folder='.'  #'evaluation_metrics'
    os.makedirs(f'{comparison_folder}/original_data', exist_ok=True)
    with open(f'{comparison_folder}/original_data/{variable}_tier{tier}.pkl3', 'wb') as f: pkl.dump((ids, single_items_original_data), f)
    
    for method in single_items_estimated.keys():
        os.makedirs(f'{comparison_folder}/{method}', exist_ok=True)
        with open(f'{comparison_folder}/{method}/{variable}_tier{tier}.pkl3', 'wb') as f: pkl.dump((ids, single_items_estimated[method]), f)

    print('Done!')
