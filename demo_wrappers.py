"""
Wrappers that run DMDc, DMDc-opt and PullbackDMDc on a single dataset in memory.

All three functions have the signature of PullbackDMDc's get_forced_response:
    get_forced_response_*(ds, var_short, tier_num, forcings_df) -> forced response dataset
and are used in demo_evaluation_results.ipynb.

Running this file is a test of the wrappers: it estimates the forced response of one
member of a ForceSMIP training large ensemble, compares the global means with the
ensemble mean (ground truth), saves the plotted time series to a .pkl3 file and plots them.

Expected training data layout: <training_path>/<Table>/<variable>/<model>/<one file per member>
"""

import os
import sys
import pickle as pkl

import numpy as np
import pandas as pd
import xcdat
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
for folder in ['DMDc', 'DMDc-opt', 'PullbackDMDc', 'evaluation_metrics']:
    sys.path.insert(0, os.path.join(REPO, folder))

if not hasattr(np, 'asfarray'):  # removed in NumPy 2; only used as a type hint in dmdc.py and dmdc_opt.py
    np.asfarray = np.asarray

from dmdc import DMDc
from dmdc_opt import DMDc_opt
from dmdc_p import get_forced_response as get_forced_response_pullbackdmdc
from dmdc_p import cmipTable, cmipVar
from evaluation_metrics import single_func, custom_masked_array


def load_forcings(forcings_df, tier_num, mean_center=True, rescale=False):
    """Analogue of DMDc.data_utils.load_forcings that takes the forcing DataFrame instead of reading the csv."""
    if tier_num == 2:
        forcings = np.array(forcings_df.iloc[228:]['totalforcing'])
    elif tier_num == 3:
        forcings = np.array(forcings_df.iloc[1176:]['totalforcing'])

    if rescale and mean_center:
        forcings = (forcings - np.mean(forcings)) / np.std(forcings)
    elif mean_center:
        forcings = forcings - np.mean(forcings)
    elif rescale:
        forcings = forcings / np.std(forcings)

    return forcings


def load_data(ds, var_short):
    """Analogue of data_utils.load_data for one dataset already in memory."""
    # Masked array without nans and big numbers (same as in data_utils.load_data)
    x = custom_masked_array(ds[var_short])

    # Computing anomalies
    t = x.shape[0]
    climatology = np.concatenate(np.ma.stack(np.split(x, t // 12, axis=0), axis=0).mean(axis=0, keepdims=True).repeat(t // 12, axis=0), axis=0)
    anomalies = x - climatology

    # Remove pixels without data
    nan_mask = np.ma.getmaskarray(anomalies)
    data = anomalies[:, np.logical_not(np.all(nan_mask, axis=0))]
    return data, nan_mask


def reformat(pred_forced_resp, nan_mask, ds, var_short):
    """Analogue of data_utils.reformat_and_save that returns the dataset instead of saving it."""
    # Add the mask back in
    grid = np.full(nan_mask.shape, np.nan)
    grid[:, np.logical_not(np.all(nan_mask, axis=0))] = pred_forced_resp
    grid[nan_mask] = np.nan

    forced_resp_ds = ds.copy()
    forced_resp_ds[var_short] = (ds[var_short].dims, grid)
    return forced_resp_ds


def get_forced_response_dmdc(ds, var_short, tier_num, forcings_df):
    """As in run_dmdc.py."""
    truncation = 10 if (tier_num == 2 and var_short == 'tas') else 20
    forcings_centered_scaled = load_forcings(forcings_df, tier_num, mean_center=True, rescale=True)

    data, nan_mask = load_data(ds, var_short)
    my_dmdc = DMDc(truncation=truncation, lag=3)
    my_dmdc.fit(data.data, forcings_centered_scaled)
    centered_fr_dmdc, data_mean_dmdc = my_dmdc.predict()
    preds = centered_fr_dmdc + data_mean_dmdc
    return reformat(preds, nan_mask, ds, var_short)


def get_forced_response_dmdc_opt(ds, var_short, tier_num, forcings_df):
    """As in run_dmdc_opt.py."""
    truncation = 10 if (tier_num == 2 and var_short == 'tas') else 20
    forcings_centered_scaled = load_forcings(forcings_df, tier_num, mean_center=True, rescale=True)

    data, nan_mask = load_data(ds, var_short)
    my_dmdc = DMDc_opt(truncation=truncation, lag=3, optlag=60)
    my_dmdc.fit(data.data, forcings_centered_scaled)
    centered_fr_dmdcopt, data_mean_dmdcopt = my_dmdc.predict()
    preds = centered_fr_dmdcopt + data_mean_dmdcopt
    return reformat(preds, nan_mask, ds, var_short)


if __name__ == '__main__':
    # ---- Settings ----
    tier = 2                    # 2: 1900-2023, 3: 1979-2023 (as in get_forcings of dmdc_p.py)
    training_path = '/media/disk/databases/ForceSMIP/Training-Ext'
    model = 'CESM2'
    variable = 'tas'            # tos, tas, pr, psl, monmaxtasmax, mrso, siconc
    member = 0                  # index in the sorted file list, or a substring, e.g. 'r10i1301p1f1'
    output_folder = '.'
    methods = {'DMDc': get_forced_response_dmdc,
               'DMDc-opt': get_forced_response_dmdc_opt,
               'PullbackDMDc': get_forced_response_pullbackdmdc}
    colors = {'DMDc': 'tab:blue', 'DMDc-opt': 'tab:orange', 'PullbackDMDc': 'tab:red'}

    # ---- Inputs ----
    y0, y1 = {2: (1900, 2023), 3: (1979, 2023)}[tier]
    period = slice(f'{y0}-01-01', f'{y1}-12-31')
    var_short = cmipVar[variable]
    forcings_df = pd.read_csv(os.path.join(REPO, 'PullbackDMDc', 'interpolatedForcing.csv'))

    data_path = os.path.join(training_path, cmipTable[variable], variable, model)
    files = sorted(os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.nc'))
    member_file = files[member] if isinstance(member, int) else next(f for f in files if member in os.path.basename(f))
    member_name = os.path.basename(member_file).split('.')[0]
    print(f'{member_name}: tier {tier}, {len(files)} members in the ensemble', flush=True)

    ds = xcdat.open_dataset(member_file).sel(time=period)
    assert ds.sizes['time'] == (y1 - y0 + 1) * 12, f'{member_file} does not cover {y0}-{y1}'

    # ---- Ground truth: ensemble mean over all members ----
    total = 0.
    for i, f in enumerate(files, start=1):
        print(f'Ensemble mean: member {i}/{len(files)}', flush=True)
        with xcdat.open_dataset(f) as ds_m:
            total = total + ds_m[var_short].sel(time=period).values
    ensemble_mean_ds = ds.copy()
    ensemble_mean_ds[var_short] = (ds[var_short].dims, total / len(files))

    # ---- Forced response estimates and global-mean anomalies (as in evaluation_metrics.py) ----
    gm = {'member': single_func(ds, var_short)['global_mean_anomalies'],
          'ensemble_mean': single_func(ensemble_mean_ds, var_short)['global_mean_anomalies']}
    for method_name, get_forced_response in methods.items():
        print(f'Running {method_name}...', flush=True)
        gm[method_name] = single_func(get_forced_response(ds, var_short, tier, forcings_df), var_short)['global_mean_anomalies']

    results = {'model': model, 'variable': variable, 'tier': tier, 'member_file': member_file,
               'n_ensemble_members': len(files), 'time': y0 + (np.arange(ds.sizes['time']) + 0.5) / 12.,
               'global_mean_anomalies': gm}
    ds.close()

    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, f'{member_name}_tier{tier}.pkl3')
    with open(out_file, 'wb') as f:
        pkl.dump(results, f)
    print(f'Saved {out_file}')

    # ---- Plot (as in demo_evaluation_results.ipynb) ----
    t = results['time']
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, gm['member'], '-', color='gray', linewidth=2, label='Original signal')
    for method_name in methods:
        rmse = np.sqrt(np.mean((gm[method_name] - gm['ensemble_mean']) ** 2))
        ax.plot(t, gm[method_name], '-', color=colors[method_name], linewidth=2, label=f'{method_name} (RMSE: {rmse:.3g})')
    ax.plot(t, gm['ensemble_mean'], '-', color='black', linewidth=2, label=f'Ensemble mean ({len(files)} members)')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_title(f'global_mean_anomalies: {variable}, tier{tier}, {model}, {member_name}')
    fig.tight_layout()
    plt.savefig('tmp.png')
    plt.show()