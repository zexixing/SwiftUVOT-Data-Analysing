from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from scipy.ndimage import rotate
from os import listdir
from tools import *
from _mypath import thisdir
import pandas as pd
import numpy as np
import tarfile
import os

def untar(folder_path, file_name):
    """untar zip file"""
    file_path = folder_path + file_name
    tar = tarfile.open(file_path)
    names = tar.getnames()
    for name in names:
        tar.extract(name, folder_path)
    tar.close()

def untar_obs(folder_name):
    folder_path = get_path('../data/'+folder_name+'/')
    obs_list = listdir(folder_path)
    for obs in obs_list:
        if obs[0] != '.':
            if (os.path.isfile(folder_path+obs) == True and 
                os.path.exists(folder_path+obs[:-4]) == False):
                untar(folder_path, obs)
                os.remove(folder_path+obs)

def make_obs_log(horizon_id,
                 folder_name, 
                 map_type, 
                 output_name):
    """creat an observing log file before all work,
    to provide non-data info for every extension.
    
    Inputs: 1. Horizon ID of the object, string;
            2. name of the folder containing dirs of every observation, string;
            3. 'sk' or 'rw' or 'ex', expected to be 'sk', string;
            4. ouput name of the newly generated obs log, string
            
    Outputs: No, only generate an obs log file
    """
    # initiate quantity names. w: write; r: read; v: value
    # make the code easier to be modified or appended
    obs     = {"w": "OBS_ID"}
    ext     = {"w": "EXTENSION"}
    start_t = {"w": "START",
               "r": "DATE-OBS"}
    end_t   = {"w": "END",
               "r": "DATE-END"}
    mid_t   = {"w": "MIDTIME"}
    exp     = {"w": "EXP_TIME",
               "r": "EXPOSURE"}
    fil     = {"w": "FILTER",
               "r": "FILTER"}
    pa      = {"w": "PA",
               "r": "PA_PNT"}    
    helio   = {"w": "HELIO",
               "r": "r"}
    helio_v = {"w": "HELIO_V",
               "r": "r_rate"}
    obs_dis = {"w": "OBS_DIS",
               "r": "delta"}
    phase   = {"w": "PHASE",
               "r": "alpha"}
    ra      = {"w": "RA",
               "r": "RA"}
    dec     = {"w": "DEC",
               "r": "DEC"}
    px      = {"w": "PX"}
    py      = {"w": "PY"}
    
    # initiate a log file
    output_path = get_path('../docs/'+output_name)
    f = open(output_path, 'w')
    f.write(  obs["w"]     + ' ' 
            + ext["w"]     + ' ' 
            + start_t["w"] + ' ' 
            + end_t["w"]   + ' ' 
            + mid_t["w"]   + ' ' 
            + exp["w"]     + ' ' 
            + fil["w"]     + ' '
            + pa["w"]      + ' ' 
            + helio["w"]   + ' '
            + helio_v["w"] + ' ' 
            + obs_dis["w"] + ' ' 
            + phase["w"]   + ' '
            + ra["w"]      + ' '
            + dec["w"]     + ' '
            + px["w"]      + ' '
            + py["w"]      + '\n'
            )
    
    # start a loop iterating over observations
    input_path = get_path('../data/'+folder_name+'/')
    obs_list = os.listdir(input_path)
    obs_list = [obs for obs in obs_list if obs[0] != '.']
    obs_list.sort()
    for obs_id in obs_list:
        obs["v"] = obs_id
        # find the only *sk* file for every obs
        map_dir_path = os.path.join(input_path, 
                                    obs_id+'/uvot/image/')
        map_file_list = os.listdir(map_dir_path)
        map_files = [x for x in map_file_list 
                    if (map_type in x)]
        if len(map_files) == 0:
            break
        else:
            for map_file in map_files:
                map_file_path = os.path.join(map_dir_path, 
                                            map_file)
                # get the num of extensions to
                # start a loop interating over extensions
                hdul = fits.open(map_file_path)
                ext_num = len(hdul) - 1
                for ext_id in range(1, 1 + ext_num):
                    # read observing info from header of every extension
                    # and put the info into the log file
                    ext["v"]     = ext_id
                    ext_header   = hdul[ext_id].header
                    start_t["v"] = Time(ext_header[start_t["r"]])
                    end_t["v"]   = Time(ext_header[end_t["r"]])
                    exp["v"]     = ext_header[exp["r"]]
                    dt           = end_t["v"] - start_t["v"]
                    mid_t["v"]   = start_t["v"] + 1/2*dt
                    fil["v"]     = ext_header[fil["r"]]
                    pa["v"]      = ext_header[pa["r"]]
                    f.write(  obs["v"]           + ' '
                            + f'{int(ext["v"])}' + ' '
                            + f'{start_t["v"]}'  + ' '
                            + f'{end_t["v"]}'    + ' '
                            + f'{mid_t["v"]}'    + ' '
                            + f'{exp["v"]}'      + ' '
                            + fil["v"]           + ' '
                            + f'{pa["v"]}'       + ' ')
                    
                    # read ephmerides info from Horizon API
                    # and put the info into the log file
                    obj = Horizons(id=horizon_id, 
                                location='@swift',
                                epochs=mid_t["v"].jd)
                    eph = obj.ephemerides()[0]
                    helio["v"] = eph[helio["r"]]
                    helio_v["v"] = eph[helio_v["r"]]
                    obs_dis["v"] = eph[obs_dis["r"]]
                    phase["v"] = eph[phase["r"]]
                    ra["v"] = eph[ra["r"]]
                    dec["v"] = eph[dec["r"]]
                    f.write(  f'{helio["v"]}'    + ' '
                            + f'{helio_v["v"]}'  + ' '
                            + f'{obs_dis["v"]}'  + ' '
                            + f'{phase["v"]}'    + ' '
                            + f'{ra["v"]}'       + ' '
                            + f'{dec["v"]}'      + ' ')
                    w = WCS(ext_header)
                    px["v"], py["v"] = w.wcs_world2pix( ra["v"], dec["v"], 1)
                    f.write(  f'{px["v"]}'       + ' '
                            + f'{py["v"]}'       + '\n')
    f.close()
            
def set_coord(image_array, target_index,
              size):
    """To shift a target on an image 
    into the center of a new image;
    
    The size of the new image can be given 
    but have to ensure the whole original
    image is included in the new one.
    
    Inputs: array of an original image, 2D array
            original coordinate values of the target, array shape in [r, c]
            output size, tuple of 2 elements
    Outputs: array of the shifted image in the new coordinate, 2D array
    """
    # interpret the size and create new image
    try:
        half_row, half_col = size
    except:
        print("Check the given image size!")
    new_coord = np.zeros((2*half_row - 1, 
                          2*half_col - 1))
    # shift the image, [target] -> [center]
    def shift_r(r):
        return int(r+(half_row-1- target_index[0]))
    def shift_c(c):
        return int(c+(half_col-1- target_index[1]))
    for r in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            new_coord[shift_r(r), 
                      shift_c(c)] = image_array[r, c]
    # reture new image
    return new_coord

def stack_image(obs_log_name, filt, size, output_name):
    '''sum obs images according to 'FILTER'
    
    Inputs:
    obs_log_name: the name of an obs log in docs/
    filt: 'uvv' or 'uw1' or 'uw2'
    size: a tuple
    output_name: string, to be saved in docs/
    
    Outputs:
    1) a txt data file saved in docs/
    2) a fits file saved in docs/
    '''
    # load obs_log in DataFrame according to filt
    obs_log_path = get_path('../docs/'+obs_log_name)
    img_set = pd.read_csv(obs_log_path, sep=' ',
                          index_col=['FILTER'])
    img_set = img_set[['OBS_ID', 'EXTENSION',
                       'PX', 'PY', 'PA', 'EXP_TIME',
                       'END', 'START']]
    if filt == 'uvv':
        img_set = img_set.loc['V']
    elif filt == 'uw1':
        img_set = img_set.loc['UVW1']
    elif filt == 'uw2':
        img_set = img_set.loc['UVW2']
    #---transfer OBS_ID from int to string---
    img_set['OBS_ID']=img_set['OBS_ID'].astype(str)
    img_set['OBS_ID']='000'+img_set['OBS_ID']
    # create a blank canvas in new coordinate
    stacked_img = np.zeros((2*size[0] -1,
                            2*size[1] -1))
    # loop among the data set, for every image, shift it to center the target, rotate and add to the blank canvas
    exp = 0
    for i in range(len(img_set)):
        #---get data from .img.gz---
        if img_set.index.name == 'FILTER':
            img_now = img_set.iloc[i]
        else:
            img_now = img_set
        img_path = get_path(img_now['OBS_ID'], 
                            filt, to_file=True)
        img_hdu =  fits.open(img_path)[img_now['EXTENSION']]
        img_data = img_hdu.data.T # .T! or else hdul PXY != DS9 PXY
        #---shift the image to center the target---
        new_img = set_coord(img_data, 
                            np.array([img_now['PX']-1,
                                      img_now['PY']-1]),
                            size)
        #---rotate the image according to PA to
        #---eliminate changes of pointing
        #---this rotating step may be skipped---
        #new_img = rotate(new_img, 
        #                 angle=img_now['PA'],
        #                 reshape=False,
        #                 order=1)
        #---sum modified images to the blank canvas---
        stacked_img = stacked_img + new_img
        exp += img_now['EXP_TIME']
    # get the summed results and save in fits file
    output_path = get_path('../docs/'+output_name+
                          '_'+filt+'.fits')
    hdu = fits.PrimaryHDU(stacked_img)
    if img_set.index.name == 'FILTER':
        dt = Time(img_set.iloc[-1]['END']) - Time(img_set.iloc[0]['START'])
        mid_t = Time(img_set.iloc[0]['START']) + 1/2*dt
    else:
        dt = Time(img_set['END']) - Time(img_set['START'])
        mid_t = Time(img_set['START']) + 1/2*dt
    hdr = hdu.header
    hdr['TELESCOP'] = img_hdu.header['TELESCOP']
    hdr['INSTRUME'] = img_hdu.header['INSTRUME']
    hdr['FILTER'] = img_hdu.header['FILTER']
    hdr['COMET'] = obs_log_name.split('_')[0]+' '+obs_log_name.split('_')[-1][:-4]
    hdr['PLATESCL'] = ('1','arcsec/pixel')
    hdr['XPOS'] = f'{size[0]}'
    hdr['YPOS'] = f'{size[1]}'
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdr['MID_TIME'] = f'{mid_t}'
    hdu.writeto(output_path)