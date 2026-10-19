# To run the code, the following packages need to be installed:
# conda create -c conda-forge -n dmdcp python=3 numpy pandas eofs xcdat
# pip install nipals scikit-learn

import numpy as np
import pandas as pd
from eofs.standard import Eof
import os
import datetime
from nipals import nipals
import xcdat
import sys
from subprocess import Popen
import sklearn

import matplotlib.pyplot as plt

cmipVar = {
    "pr": "pr",
    "psl": "psl",
    "tas": "tas",
    "tos": "tos",
    "siconc": "siconc",
    "monmaxtasmax": "tasmax",
    "mrso": "mrso"
}
cmipTable = {
    "pr": "Amon",
    "psl": "Amon",
    "tas": "Amon",
    "tos": "Omon",
    "siconc": "OImon",
    "monmaxtasmax": "Aday",
    "mrso": "Lmon"
}

class DMDcp:
    def __init__(self,lag):
        self.lag=lag

    def fit(self,pcs,forcings):
        self.forcings_constant=forcings.mean(axis=0)
        self.t,self.npcs=pcs.shape
        forcings=forcings-self.forcings_constant
        inputs = np.concatenate([pcs[:-self.lag,:], forcings[:-self.lag,:]],axis=-1)
        targets=pcs[self.lag:,:]
        self.G=np.linalg.lstsq(inputs,targets)[0] 

    def predict(self,pcs_start,forcings):
        t,nforcings=forcings.shape
        forcings=forcings-self.forcings_constant
        
        pcs=np.zeros((t,self.npcs))
        pcs[:self.lag,:]=pcs_start
        for n in range(self.lag,t):
            pcs[n,:]=self.G.T@np.concatenate([pcs[n-self.lag,:],forcings[n-self.lag,:]],axis=-1)
        return pcs
    
    def get_pullback_attractor(self,long_forcings,transition_time):
        pcs=self.predict(pcs_start=np.zeros((self.lag,self.npcs)),forcings=long_forcings)
        pcs=pcs[transition_time:,:]
        return pcs


def compute_lagged_LR(data,forcings,lag,long_forcings):
    # Check the remaining pixels for nans
    nan_flag = np.sum(data.mask)
    print(f'nan_flag={nan_flag}')
    t_flat,dim_flat=data.shape
    t_flat_long=long_forcings.shape[0]
    long_forcings=long_forcings[t_flat_long-t_flat-lag:t_flat_long-lag,:]

    # Core method part: train the model and compute the forced response
    if nan_flag:
        # Perform the regression gridpoint-wise
        pred_forced_resp_squeezed=np.empty(data.shape)
        for k in range(dim_flat):
            # Define valid inputs and outptus
            good_mask_k=np.logical_not(data.mask[:,k])
            data_k=data.data[lag:,k][good_mask_k[lag:],None]
            forcings_k=forcings[lag:,:][good_mask_k[lag:],:]
            long_forcings_k=long_forcings[good_mask_k,:]
            # Predict the regression
            model=sklearn.linear_model.LinearRegression(fit_intercept=False)
            model.fit(forcings_k, data_k)
            pred_k=model.predict(long_forcings_k)
            # Insert predictions in the proper places of the output array
            pred_forced_resp_squeezed[good_mask_k,k]=pred_k[:,0]
            pred_forced_resp_squeezed[np.logical_not(good_mask_k),k]=np.nan
    else:
        # Perform the regression for all targets at once
        model=sklearn.linear_model.LinearRegression()
        model.fit(forcings[:t_flat-lag,:], data.data[lag:,:])
        pred_forced_resp_squeezed=model.predict(long_forcings)
    
    return pred_forced_resp_squeezed

def get_forcings(forcings_df,tier_num,transition_time=1200):
    totalforcing  = np.array(forcings_df['totalforcing'])
    forcings_before = np.ones(transition_time)*totalforcing[0]
    
    long_forcings=np.concatenate([forcings_before,totalforcing],axis=0)
    if tier_num == 2:
        long_forcings  = long_forcings[228:]
        short_forcings  = np.array(forcings_df.iloc[228:]['totalforcing'])
    elif tier_num == 3:
        long_forcings  = long_forcings[1176:]
        short_forcings  = np.array(forcings_df.iloc[1176:]['totalforcing'])
        
    return short_forcings[:,None], long_forcings[:,None]

def get_forced_response(ds,var_short,tier_num,forcings_df):
    # Custom hyperparameters of the DMDc-p method
    truncation=20
    lag=12
    if var_short=='psl' and tier_num==2: truncation=100 
    if var_short=='psl' and tier_num==3: truncation=40
    if var_short=='mrso' and tier_num==3: lag=2
    transition_time=1200

    # Extract numpy arrays for data and lats
    x=ds[var_short].to_masked_array()
    lat=ds['lat'].to_numpy()

    # Update data mask to exclude nans and big numbers for sure
    good_mask=np.logical_and(np.logical_not(x.mask),np.abs(x.data)<=1e+16)
    x=np.ma.masked_array(x.data,mask=np.logical_not(good_mask))

    # Geographical rescaling coefficient
    resc_coef=np.sqrt(np.cos(lat[:,None]/180.*np.pi))
    x=x*resc_coef

    # Computing anomalies
    t, nlat, nlon = x.data.shape
    climatology=np.concatenate(np.ma.stack(np.split(x,t//12,axis=0),axis=0).mean(axis=0,keepdims=True).repeat(t//12,axis=0),axis=0)
    anomalies=x-climatology


    # Remove pixels without data
    nan_mask = np.all(anomalies.mask, axis=0)
    data = anomalies[:,np.logical_not(nan_mask)]
    
    # Check the remaining pixels for nans
    nan_flag = np.sum(data.mask)
    print(f'nan_flag={nan_flag}')

    # Dimensionality reduction
    data_mean = data.mean(axis = 0)
    data_centered = data - data_mean
    data_centered=data_centered.data
    data_centered[data.mask]=np.nan  # a 2D np.array with nans in the missing places (for nipals algorithm) 
    if nan_flag:
        pca = nipals.Nipals(data_centered)
        pca.fit(ncomp=truncation, center = False, scale = False, maxiter=10000, tol = 0.0001)
        eofs = pca.loadings.to_numpy().T
        pcs = pca.scores.to_numpy()
    else:
        eofs_xr = Eof(data_centered, center=False, ddof=1)
        eofs = eofs_xr.eofs(neofs=truncation)
        pcs = eofs_xr.pcs(npcs=truncation)

    # Prepare forcing time series for the model: normal and extended in the past
    forcings, long_forcings=get_forcings(forcings_df,tier_num,transition_time)

    # Core method part: train the model and compute the forced response
    model=DMDcp(lag)
    model.fit(pcs, forcings)
    pcs_forced=model.get_pullback_attractor(long_forcings,transition_time)

    # The model with zero DMD part for the PCA residual (grid-wise regression to a lagged forcing)
    data_centered_residual = np.ma.MaskedArray(data_centered - np.matmul(pcs, eofs), mask=data.mask)
    data_centered_residual_forced=compute_lagged_LR(data_centered_residual,forcings,lag,long_forcings)    

    # Inverse transform
    pred_forced_resp_squeezed = np.matmul(pcs_forced,eofs)+data_centered_residual_forced

    # Optional plot to check
    #plt.plot(data_centered.mean(axis=-1))
    #plt.plot(pred_forced_resp_squeezed.mean(axis=-1))
    #plt.show()

    # Go back to normal grid, make masked_array, add climatology and unscale
    pred_forced_resp = np.empty((t,nlat,nlon))
    pred_forced_resp[:, np.logical_not(nan_mask)] = pred_forced_resp_squeezed+data_mean
    pred_forced_resp[anomalies.mask]=np.nan
    pred_forced_resp=np.ma.masked_array(pred_forced_resp,mask=anomalies.mask)
    pred_forced_resp=(pred_forced_resp+climatology)/resc_coef

    # Physical postprocessing
    if var_short in ['pr','mrso']:
        pred_forced_resp[np.logical_not(anomalies.mask)]=np.maximum(pred_forced_resp[np.logical_not(anomalies.mask)],0)  
    if var_short in ['siconc']:
        pred_forced_resp[np.logical_not(anomalies.mask)]=np.clip(pred_forced_resp[np.logical_not(anomalies.mask)],0.,100.)

    # Save to the xr dataset object by copying from the input dataset with update
    new_dict={key: ds[key] for key in ds.data_vars if var_short!=key}
    new_dict[var_short]=pred_forced_resp
    forced_resp_ds=ds.copy(data=new_dict)

    return forced_resp_ds
                

if __name__ == '__main__':
    method = 'DMDc-p'
    group_name = 'UV-ISP'    
    print('Command line arguments 1-4:')
    print('sys.argv[1] -- ForceSMIP (input) folder name; subsequent directory structure will be assumed')
    input_path=sys.argv[1]
    print('sys.argv[2] -- folder name for predictions; it must exist')
    output_path=sys.argv[2]
    print('sys.argv[3] -- variable name')
    variable=sys.argv[3]
    print('sys.argv[4] -- tier number (2 or 3)')
    tier_num=int(sys.argv[4])    

    # Load forcing time series
    forcings_df = pd.read_csv('./interpolatedForcing.csv')
    
    # Define evaluation folder
    data_path = os.path.join(input_path,f'Evaluation-Tier{tier_num}',cmipTable[variable],variable)            
    
    # Loop over data files in evaluation folder
    data_files=sorted(os.listdir(data_path))
    for data_file in data_files:

        # Open the dataset
        ds = xcdat.open_dataset(os.path.join(data_path,data_file))


        # Compute the corresponding forced response dataset
        forced_resp_ds=get_forced_response(ds,cmipVar[variable],tier_num,forcings_df)
        #forced_resp_ds.attrs = {'creation_date': str(datetime.datetime.now()), 'method': method}
        forced_resp_ds.attrs['creation_date']=str(datetime.datetime.now())
        forced_resp_ds.attrs['method']=method

        #Save to the file  
        if 'evaluation_id' in ds.attrs:
            evaluation_id=ds.attrs['evaluation_id']
            f_name=os.path.join(output_path,f"{variable}_{evaluation_id}_{tier_num}_{method}_{group_name}.nc")
        else:
            f_name = os.path.join(output_path, data_file)
        ds.close()
        forced_resp_ds.to_netcdf(f_name)

        print(f_name + ' done!',flush=True)

