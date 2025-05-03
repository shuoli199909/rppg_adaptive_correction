"""
Transform raw RGB traces to BVP signals.
"""

# Author: Shuo Li
# Date: 2025/04/12

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations.
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_analysis


def main_rgb2bvpwin(name_dataset, algorithm):
    """Main function for transforming RGB traces to HR-related signals.
    Parameters
    ----------
    name_dataset: Name of the selected dataset.
                  ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    algorithm: Selected rPPG algorithm. ['CHROM', 'LGI', 'OMIT', 'POS'].
    
    Returns
    -------

    """
    # Get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # Parameter class initialization.
    Params = util_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)

    # RGB signal -> bvp signal.
    if name_dataset == 'UBFC-rPPG':
        # Sequence num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 11)) + list(range(12, 18)) + list(range(22, 24)) + \
                         list(range(25, 27)) + list(range(30, 32)) + list(range(33, 50))
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
            dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'.csv')
            df_rgb = pd.read_csv(dir_sig_rgb)
            # Fill NA.
            df_rgb["R"] = df_rgb["R"].fillna(method = 'ffill').fillna(method = 'bfill')
            df_rgb["G"] = df_rgb["G"].fillna(method = 'ffill').fillna(method = 'bfill')
            df_rgb["B"] = df_rgb["B"].fillna(method = 'ffill').fillna(method = 'bfill')
            # RGB signal initialization.
            sig_rgb = df_rgb[['R', 'G', 'B']].values
            sig_rgb = sig_rgb[:, np.newaxis, :]
            # RGB video information.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # Get video fps.
            capture = cv2.VideoCapture(dir_vid)
            Params.fps = capture.get(cv2.CAP_PROP_FPS)
            # RGB signal -> windowed bvp signals.
            sig_bvp_win, time_win = util_analysis.rgb2bvpwin_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
            # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
            df_bvp_win = pd.DataFrame(columns=['Time', 'Window Number', 'BVP'])
            # Loop over all windows.
            for i_window in range(0, len(sig_bvp_win)):
                data_bvp_win_tmp = {'Time': time_win[i_window, :], 'Window Number': i_window, 
                                    'BVP': sig_bvp_win[i_window][0, :]}
                df_bvp_win_tmp = pd.DataFrame(data=data_bvp_win_tmp)
                df_bvp_win = pd.concat([df_bvp_win, df_bvp_win_tmp])
            # Data saving.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'bvp_win', str(num_attendant)+'_'+algorithm+'.csv')
            df_bvp_win.to_csv(dir_save_data, index=False)


    elif name_dataset == 'LGI-PPGI':
        # name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # motion types.
        list_motion = ['gym', 'talk', 'rotation']
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                # parse the RGB signal from the RGB dataframe. size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # Fill NA.
                df_rgb["R"] = df_rgb["R"].fillna(method = 'ffill').fillna(method = 'bfill')
                df_rgb["G"] = df_rgb["G"].fillna(method = 'ffill').fillna(method = 'bfill')
                df_rgb["B"] = df_rgb["B"].fillna(method = 'ffill').fillna(method = 'bfill')
                # RGB signal initialization.
                sig_rgb = df_rgb[['R', 'G', 'B']].values
                sig_rgb = sig_rgb[:, np.newaxis, :]
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> windowed bvp signals.
                sig_bvp_win, time_win = util_analysis.rgb2bvpwin_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_bvp_win = pd.DataFrame(columns=['Time', 'Window Number', 'BVP'])
                # Loop over all windows.
                for i_window in range(0, len(sig_bvp_win)):
                    data_bvp_win_tmp = {'Time': time_win[i_window, :], 'Window Number': i_window, 
                                        'BVP': sig_bvp_win[i_window][0, :]}
                    df_bvp_win_tmp = pd.DataFrame(data=data_bvp_win_tmp)
                    df_bvp_win = pd.concat([df_bvp_win, df_bvp_win_tmp])
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'bvp_win', attendant+'_'+motion+'_'+algorithm+'.csv')
                df_bvp_win.to_csv(dir_save_data, index=False)


    elif name_dataset == 'BUAA-MIHR':
        # sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # lux levels.
        list_lux = ['lux 6.3', 'lux 10.0', 'lux 15.8', 'lux 25.1']
        # attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # loop over all illumination levels.
            for lux in list_lux:
                # parse the RGB signal from the RGB dataframe. size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+lux.replace(' ', '')+'.csv')
                df_rgb = pd.read_csv(dir_sig_rgb, index_col=None)
                # Fill NA.
                df_rgb["R"] = df_rgb["R"].fillna(method = 'ffill').fillna(method = 'bfill')
                df_rgb["G"] = df_rgb["G"].fillna(method = 'ffill').fillna(method = 'bfill')
                df_rgb["B"] = df_rgb["B"].fillna(method = 'ffill').fillna(method = 'bfill')
                # RGB signal initialization.
                sig_rgb = df_rgb[['R', 'G', 'B']].values
                sig_rgb = sig_rgb[:, np.newaxis, :]
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, \
                                       lux.replace(' ', '') + '_' + list_name[num_attendant-1]+'.avi')
                # get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> windowed bvp signals.
                sig_bvp_win, time_win = util_analysis.rgb2bvpwin_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_bvp_win = pd.DataFrame(columns=['Time', 'Window Number', 'BVP'])
                # Loop over all windows.
                for i_window in range(0, len(sig_bvp_win)):
                    data_bvp_win_tmp = {'Time': time_win[i_window, :], 'Window Number': i_window, 
                                        'BVP': sig_bvp_win[i_window][0, :]}
                    df_bvp_win_tmp = pd.DataFrame(data=data_bvp_win_tmp)
                    df_bvp_win = pd.concat([df_bvp_win, df_bvp_win_tmp])
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'bvp_win', str(num_attendant).zfill(2) + \
                                             '_' + str(lux).replace(' ', '') + '_' + algorithm+'.csv')
                df_bvp_win.to_csv(dir_save_data, index=False)


if __name__ == "__main__":
    # available datasets.
    list_dataset = ['UBFC-rPPG', 'LGI-PPGI', 'BUAA-MIHR']  #, 'UBFC-rPPG', 'LGI-PPGI']   # ['UBFC-rPPG', 'LGI-PPGI', 'BUAA-MIHR'].
    # selected rPPG algorithms.
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']   # ['LGI', 'OMIT', 'CHROM', 'POS', 'ICA', 'PBV'].
    for name_dataset in list_dataset:
        for algorithm in list_algorithm:
            print([name_dataset, algorithm])
            main_rgb2bvpwin(name_dataset=name_dataset, algorithm=algorithm)