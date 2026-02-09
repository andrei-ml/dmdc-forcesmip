import xarray as xr
import numpy as np
import os
import datetime


import pandas as pd


def load_forcings(tier_num: int, mean_center: bool = True, rescale: bool = False):
    # Load Forcings
    forcings_df = pd.read_csv('./interpolatedForcing.csv')
    if tier_num == 2:
        forcings  = np.array(forcings_df.iloc[228:]['totalforcing'])
    elif tier_num == 3:
        forcings  = np.array(forcings_df.iloc[1176:]['totalforcing'])
    
    if rescale and mean_center:
        forcings  = (forcings-np.mean(forcings))/np.std(forcings)
    elif mean_center:
        forcings  = forcings-np.mean(forcings)
    elif rescale:
        forcings  = forcings/np.std(forcings)
    
    return forcings

def load_data(data_path: str = './data/Evaluation-Tier1', 
               dir_names: dict = {'Omon': ['tos']},
               group_name: str = 'UV-ISP',
               method: str = 'DMDc',
               tier: int=2):
    datasets = {}
    nan_masks = {}
    for dir_name in dir_names.keys():

        dir_vars = dir_names[dir_name]

        for var in dir_vars:
            
            # Loop through each file
            directory = os.path.join(data_path, dir_name, var)
            # Define the file path
            listdir = os.listdir(directory)

            for i, file in enumerate(listdir, start=1):

                print('File {}/{}'.format(i,len(listdir)), end='\r')
                # Reading temperature file
                ## Open the NetCDF file using xarray
                file_path = os.path.join(directory, file)
                ds = xr.open_dataset(file_path)

                if var == 'monmaxtasmax':
                    var_short = 'tasmax'
                else:
                    var_short = var

                # Extract numpy arrays for data and lats
                x=ds[var_short].to_masked_array()
                lat=ds['lat'].to_numpy()

                # Update data mask to exclude nans and big numbers for sure
                good_mask=np.logical_and(np.logical_not(x.mask),np.abs(x.data)<=1e+16)
                x=np.ma.masked_array(x.data,mask=np.logical_not(good_mask))

                # Computing anomalies
                t, nlat, nlon = x.data.shape
                climatology=np.concatenate(np.ma.stack(np.split(x,t//12,axis=0),axis=0).mean(axis=0,keepdims=True).repeat(t//12,axis=0),axis=0)
                anomalies=x-climatology

                # Remove pixels without data
                # nan_mask = np.all(anomalies.mask, axis=0)
                nan_mask = anomalies.mask
                data = anomalies[:,np.logical_not(np.all(nan_mask, axis=0))]


                split_fpath = file_path.split('/')
                name_memID = split_fpath[-1].split('.')[0]
                data_name = f'{name_memID}_{tier}_{method}_{group_name}'

                datasets[data_name] = data
                nan_masks[data_name] = nan_mask

    return datasets, nan_masks, t, nlat, nlon, ds, var_short

def reformat_and_save(pred_forced_resp, 
                      nan_mask,
                      t, nlat, nlon, 
                      ds, 
                      method,
                      out_path,
                      var_name,
                      data_name):

    ################################
    ################################

    #add the mask back in
    if pred_forced_resp.shape != (t,nlat*nlon):
        dmdc_pred_forced_resp = np.empty((t,nlat,nlon))
        dmdc_pred_forced_resp[~nan_mask] = pred_forced_resp.ravel()
        dmdc_pred_forced_resp[nan_mask] = np.nan
    else:
        dmdc_pred_forced_resp = pred_forced_resp.reshape((t,nlat,nlon))


    #make a xarray object
    forced_resp_xr = ds.copy()
    forced_resp_xr = forced_resp_xr.rename({list(ds.data_vars)[0]:var_name})
    forced_resp_xr[var_name] = ({'time':ds['time'].values,'lat': ds['lat'].values, 'lon': ds['lon'].values},dmdc_pred_forced_resp)
    forced_resp_xr.attrs = {'creation_date': str(datetime.datetime.now()), 'method': method}

    #save the result
    f_name = os.path.join(out_path, f'{data_name}.nc')
    forced_resp_xr.to_netcdf(f_name)
    
    print(f'{f_name} done!')
    
