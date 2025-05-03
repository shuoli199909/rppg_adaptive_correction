"""
Extract raw RGB traces from facial videos.
"""

# Author: Shuo Li
# Date: 2025/04/12

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations.
import sys
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_analysis
import util_pyVHR


def main_vid2rgb(name_dataset):
    """Main function for face detection of videos.
    Parameters
    ----------
    name_dataset: Name of the selected dataset.
                  [UBFC-rPPG, UBFC-Phys, LGI-PPGI, BUAA-MIHR].
    
    Returns
    -------

    """
    # Get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # Parameter class initialization.
    Params = util_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)


    # Video -> RGB signal.
    if name_dataset == 'UBFC-rPPG':
        # Sequnce num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 11)) + list(range(12, 18)) + list(range(22, 24)) + \
                         list(range(25, 27)) + list(range(30, 32)) + list(range(33, 50))
        for num_attendant in tqdm(list_attendant):
            print([name_dataset, num_attendant])
            # Video directory.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # Video fps.
            Params.fps = util_pyVHR.get_fps(dir_vid)
            # RGB signal extraction.
            df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
            # Save RGB signals.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'.csv')
            df_rgb.to_csv(dir_save_data, index=False)


    elif name_dataset == 'LGI-PPGI':
        # Name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # Motion types.
        list_motion = ['talk', 'gym', 'resting', 'rotation']
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                print([name_dataset, attendant, motion])
                # Video directory.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # Video fps.
                Params.fps = util_pyVHR.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # Save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'.csv')
                df_rgb.to_csv(dir_save_data, index=False)
 

    elif name_dataset == 'BUAA-MIHR':
        # Sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # Illumination levels.
        list_lux = ['lux 6.3', 'lux 10.0', 'lux 15.8', 'lux 25.1']
        # Attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        for num_attendant in tqdm(list_attendant):
            for lux in list_lux:
                name = list_name[num_attendant-1]
                print([num_attendant, name, lux])
                # Video directory.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, lux.replace(' ', '')+'_'+name+'.avi')
                # Video fps.
                Params.fps = util_pyVHR.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # Save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+str(lux).replace(' ', '')+'.csv')
                df_rgb.to_csv(dir_save_data, index=False)


if __name__ == "__main__":
    # Available datasets.
    list_dataset = ['UBFC-rPPG', 'LGI-PPGI', 'BUAA-MIHR']  # ['UBFC-rPPG', 'LGI-PPGI', 'BUAA-MIHR'].
    # Extract RGB signal
    for name_dataset in list_dataset:
        main_vid2rgb(name_dataset=name_dataset)