"""
Transform the groundtruth HR data from the original dataset into standard format.
"""

# Author: Shuo Li
# Date: 2025/02/12

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
import sys
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pyVHR
import util_analysis


def main_gen_gtHR(name_dataset):
    """Main function for generating ground truth BPM data for UBFC-Phys dataset.
    Parameters
    ----------
    name_dataset: Name of the selected rPPG dataset. 
                  ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    
    Returns
    -------

    """

    # Get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # Parameter class initialization.
    Params = util_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)
    # Groundtruth class initialization.
    GT = util_analysis.GroundTruth(dir_dataset=Params.dir_dataset, name_dataset=name_dataset)

    # Structure for different datasets.
    if name_dataset == 'UBFC-rPPG':
        # Sequnce num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 11)) + list(range(12, 18)) + list(range(22, 24)) + \
                         list(range(25, 27)) + list(range(30, 32)) + list(range(33, 50))
        # Create dataframe for saving standard groundtruth data.
        df_gt = pd.DataFrame(columns=['Time', 'BVP', 'BPM'])
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            print([name_dataset, num_attendant])
            # Load ground truth.
            gtTime, gtTrace, gtHR = GT.get_GT(specification=['realistic', num_attendant], 
                                              num_frame_interp=None, 
                                              slice=[0, 1])
            data_gt = {'Time': gtTime, 'BVP': gtTrace, 'BPM': gtHR}
            df_gt = pd.DataFrame(data_gt)
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'gt', str(num_attendant)+'.csv')
            df_gt.to_csv(dir_save_data, index=None)



    elif name_dataset == 'LGI-PPGI':
        # Name of attendants.
        list_attendant = ['angelo', 'alex', 'cpi', 'david', 'felix', 'harun']
        # Motion types.
        list_motion = ['gym', 'talk', 'resting', 'rotation']
        # Loop over all attendants.
        for name_attendant in tqdm(list_attendant):
            for motion in list_motion:
                print([name_dataset, name_attendant, motion])
                # Load ground truth.
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[name_attendant, motion], 
                                                  num_frame_interp=None, 
                                                  slice=[0, 1])
                data_gt = {'Time': gtTime, 'BVP': gtTrace, 'BPM': gtHR}
                df_gt = pd.DataFrame(data_gt)
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'gt', name_attendant+'_'+motion+'.csv')
                df_gt.to_csv(dir_save_data, index=None)
 

    elif name_dataset == 'BUAA-MIHR':
        # Sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # Illumination levels.
        list_lux = ['lux 1.0', 'lux 1.6', 'lux 2.5', 'lux 4.0', 'lux 6.3', 'lux 10.0', 'lux 15.8', 'lux 25.1']
        # Attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # Loop over all illumination levels.
            for lux in list_lux:
                print([name_dataset, num_attendant, lux])
                # Load ground truth.
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[num_attendant, lux, list_name[num_attendant-1]], 
                                                  num_frame_interp=None, 
                                                  slice=[0, 1])
                data_gt = {'Time': gtTime, 'BVP': gtTrace, 'BPM': gtHR}
                df_gt = pd.DataFrame(data_gt)
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'gt', str(num_attendant).zfill(2)+'_lux'+lux[4:]+'.csv')
                df_gt.to_csv(dir_save_data, index=None)


if __name__ == "__main__":
    # Generate standard ground truth HR for included datasets.
    list_dataset = ['UBFC-rPPG', 'LGI-PPGI', 'BUAA-MIHR']  # ['UBFC-rPPG', 'LGI-PPGI', 'BUAA-MIHR'].
    # loop over all selected rPPG datasets.
    for name_dataset in list_dataset:
        print([name_dataset])
        main_gen_gtHR(name_dataset=name_dataset)