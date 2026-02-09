import numpy as np
import os
import datetime
from data_utils import load_data, load_forcings, reformat_and_save
from dmdc import DMDc


if __name__ == '__main__':
    method_name='DMDc'
    for tier_num in [2,3]:

        forcings_centered_scaled = load_forcings(tier_num, mean_center = True, rescale = True)


        # data_path = f'/data/databases/ForceSMIP/Evaluation-Tier{tier_num}'  #for testing
        data_path = f'/data/databases/ForceSMIP/ForceSMIP_update/Evaluation-Tier{tier_num}'  #for testing

        group_name = 'UV-ISP'

        #evaluaton variables
        
        # if tier_num == 2:
        #     dir_names = {'Amon': ['pr', 'psl'],
        #                 'Omon': ['tos']
        #                 }
        if tier_num == 2:
            dir_names = {'Amon': ['tas']
                        }
        # elif tier_num == 3:
        #     dir_names = {'Aday': ['monmaxtasmax'],
        #                 'Amon': ['pr', 'psl','tas'],
        #                 'Lmon': ['mrso'],
        #                 'OImon': ['siconc'],
        #                 'Omon': ['tos']
        #                 }
        elif tier_num == 3:
            dir_names = {'Aday': ['monmaxtasmax'],
                        'OImon': ['siconc']
                        }
        #############

        datasets, nan_masks, t, nlat, nlon, ds, var_short = load_data(data_path, 
                                                                      dir_names, 
                                                                      group_name, 
                                                                      method=method_name,
                                                                      tier=tier_num)

        for data_name in datasets.keys():
            print(data_name)
            var_name = data_name.split('_')[0]

            #remove nans
            data  = datasets[data_name]
            nan_mask = nan_masks[data_name]
            

            method_name = 'DMDc'

            if tier_num == 2 and var_name == 'tas':
                truncation = 10
            else:
                truncation = 20

            my_dmdc = DMDc(truncation=truncation, lag=3)
            my_dmdc.fit(data.data, forcings_centered_scaled)
            centered_fr_dmdc, data_mean_dmdc = my_dmdc.predict()
            preds = centered_fr_dmdc + data_mean_dmdc

            out_path = f'./DMDc/Tier{tier_num}' #directory to save results


            reformat_and_save(preds, 
                            nan_mask,
                            t, nlat, nlon, 
                            ds, 
                            method_name,
                            out_path,
                            var_name,
                            data_name)










            
            
