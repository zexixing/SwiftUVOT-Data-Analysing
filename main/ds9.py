#!/usr/bin/python

from tools import get_path
import itertools
import numpy as np
import pandas as pd
import os

def ds9_reg_files():
    name = input('target name[Borisov] or no: ')
    if not name:
        name = 'Borisov'
    no_reg_list = input('image files loaded without regions: ')
    no_reg_list = no_reg_list.split()
    reg_list = input('image files loaded with regions: ')
    reg_list = reg_list.split()
    month = input('input the month(sep, nov) of your observations: ')
    if name != 'no':
        if_ext = input('only the first extension [n]? ')
        if not if_ext:
            if_ext = 'y'
        log_path = get_path('../docs/'+month+'_obs-log_'+name+'.txt')
        img_set = pd.read_csv(log_path, sep=' ')
        img_set = img_set[['OBS_ID', 'EXTENSION', 'FILTER', 'PX', 'PY']]
        img_set['OBS_ID']=img_set['OBS_ID'].astype(str)
        img_set['OBS_ID']=[i[:5] for i in img_set['OBS_ID'].to_list()]
        img_set = img_set.set_index(['OBS_ID'])
        front_path = get_path('../data/'+name+'_raw_'+month+'/')
        no_reg_path = []
        reg_path = []
        for f in no_reg_list:
            if isinstance(img_set.loc[f]['EXTENSION'],np.int64):
                no_reg_path.append('"'+front_path+'000'+f+'/'+
                                   'uvot/image/sw000'+f+
                                   'uw2_sk.img.gz" -cmap aips0 -zscale -smooth ')
            else:
                if if_ext == 'n':
                    loop_num = len(img_set.loc[f])
                else:
                    loop_num = 1
                for e in range(loop_num):
                    ext = img_set.loc[f].iloc[e]["EXTENSION"]
                    filt = img_set.loc[f].iloc[e]["FILTER"]
                    if filt == 'V':
                        filt = 'uvv'
                    elif filt == 'UVW1':
                        filt = 'uw1'
                    no_reg_path.append('"'+front_path+'000'+f+'/'+
                                       'uvot/image/sw000'+f+
                                       filt+'_sk.img.gz['+f'{int(ext)}'+']"'+
                                       ' -cmap aips0 -zscale -smooth')
        for f in reg_list:
            if isinstance(img_set.loc[f]['EXTENSION'],np.int64):
                reg_path.append('"'+front_path+'000'+f+'/'+
                                'uvot/image/sw000'+f+
                                'uw2_sk.img.gz[1]"'+
                                ' -regions command "circle '+
                                str(img_set.loc[f]['PX'])+' '+
                                str(img_set.loc[f]['PY'])+' 20"'+
                                ' -cmap aips0 -zscale -smooth')
            else:
                if if_ext == 'n':
                    loop_num = len(img_set.loc[f])
                else:
                    loop_num = 1
                for e in range(loop_num):
                    ext = img_set.loc[f].iloc[e]["EXTENSION"]
                    filt = img_set.loc[f].iloc[e]["FILTER"]
                    if filt == 'V':
                        filt = 'uvv'
                    elif filt == 'UVW1':
                        filt = 'uw1'
                    px = img_set.loc[f].iloc[e]["PX"]
                    py = img_set.loc[f].iloc[e]["PY"]
                    reg_path.append('"'+front_path+'000'+f+'/'+
                                    'uvot/image/sw000'+f+
                                    filt+'_sk.img.gz['+f'{int(ext)}'+']"'+
                                    ' -regions command "circle '+
                                    f'{px}'+' '+f'{py}'+' 20"'+
                                    ' -cmap aips0 -zscale -smooth')
        command_line = '/Applications/SAOImageDS9.app/Contents/MacOS/ds9 ' + \
                       ' '.join(no_reg_path) + ' '+\
                       ' '.join(reg_path) + ' &'
    else:
        front_path = get_path('../docs/')
        no_reg_list = [front_path+f+' -cmap aips0 -zscale -smooth' for f in no_reg_list]
        if reg_list:
            reg_name_list = input("regions' names: ")
            reg_name_list = reg_name_list.split()
            if len(reg_list) != len(reg_name_list):
                raise Exception('different numbers of fits and region files!')
            reg_list = [front_path+f+' -cmap aips0 -zscale -smooth' for f in reg_list]
            reg_name_list = ['-regions '+front_path+r for r in reg_name_list]
            reg_list = list(itertools.chain.from_iterable(zip(reg_list,reg_name_list)))
        command_line = '/Applications/SAOImageDS9.app/Contents/MacOS/ds9 ' + \
                       ' '.join(no_reg_list) + ' '+\
                       ' '.join(reg_list) + ' &'
    return command_line

def ds9_reg_filters():
    name = input('target name [Borisov]: ')
    month = input('input the month(sep, nov) of your observations: ')
    if not name:
        name = 'Borisov'
    front_path = get_path('../data/'+name+'_raw_'+month+'/')
    log_path = get_path('../docs/'+month+'_obs-log_'+name+'.txt')
    img_set = pd.read_csv(log_path, sep=' ', index_col='FILTER')
    img_set = img_set[['OBS_ID', 'EXTENSION', 'PX', 'PY']]
    filt = input('uvv or uw1 or uw2: ')
    if filt == 'uvv':
        img_set = img_set.loc['V']
    elif filt == 'uw1':
        img_set = img_set.loc['UVW1']
    elif filt == 'uw2':
        img_set = img_set.loc['UVW2']
    else:
        raise Exception('Check the input filter name!')
    if_ext = input('only the first ext [y]? ')
    if not if_ext:
        if_ext = 'y'
    if_reg = input('with or without regions [y]? ')
    if not if_reg:
        if_reg = 'y'
    img_set['OBS_ID']=img_set['OBS_ID'].astype(str)
    img_set["OBS_ID"]='000'+img_set["OBS_ID"]
    img_set = img_set.set_index('OBS_ID')
    obs_id_list = list(set(img_set.index.to_list()))
    command_list = []
    if if_reg == 'n':
        for f in obs_id_list:
            if isinstance(img_set.loc[f]['EXTENSION'],np.int64):
                command_list.append('"'+front_path+f+
                                    '/uvot/image/sw'+f+filt+'_sk.img.gz[1]"'+
                                    ' -cmap aips0 -zscale -smooth')
            else:
                if if_ext == 'n':
                    loop_num = len(img_set.loc[f])
                else:
                    loop_num = 1
                for e in range(loop_num):
                    ext = img_set.loc[f].iloc[e]["EXTENSION"]
                    command_list.append('"'+front_path+f+
                                        '/uvot/image/sw'+f+filt+
                                        '_sk.img.gz['+f'{int(ext)}'+']"'+
                                        ' -cmap aips0 -zscale -smooth')
    elif if_reg == 'y':
        for f in obs_id_list:
            if isinstance(img_set.loc[f]['EXTENSION'],np.int64):
                command_list.append('"'+front_path+f+
                                    '/uvot/image/sw'+f+'uw2_sk.img.gz[1]"'+
                                    ' -regions command "circle '+
                                    str(img_set.loc[f]['PX'])+' '+
                                    str(img_set.loc[f]['PY'])+' 20"'+
                                    ' -cmap aips0 -zscale -smooth')
            else:
                if if_ext == 'n':
                    loop_num = len(img_set.loc[f])
                else:
                    loop_num = 1
                for e in range(loop_num):
                    ext = img_set.loc[f].iloc[e]["EXTENSION"]
                    px = img_set.loc[f].iloc[e]["PX"]
                    py = img_set.loc[f].iloc[e]["PY"]
                    command_list.append('"'+front_path+f+
                                        '/uvot/image/sw'+f+
                                        filt+'_sk.img.gz['+f'{int(ext)}'+']"'+
                                        ' -regions command "circle '+
                                        f'{px}'+' '+f'{py}'+' 20"'+
                                        ' -cmap aips0 -zscale -smooth')
    command_list = '/Applications/SAOImageDS9.app/Contents/MacOS/ds9 ' + \
                   ' '.join(command_list)+' &'
    return command_list

def run_ds9():
    mode = input('files or filters: ')
    if mode == 'files':
        command_line = ds9_reg_files()
    elif mode == 'filters':
        command_line = ds9_reg_filters()
    else:
        raise Exception('Please check the mode!')
    os.system(command_line)

run_ds9()