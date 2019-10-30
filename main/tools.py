from _mypath import thisdir
import os

def get_path(*rela_path, to_file=False):
    """get an absolute path
    if to_file=True, the path will link to a .img.gz file according to inputted OBS_ID and FILTER
    if to_file=False, the path will link to any file/dir according to the rela_path
    
    Inputs: 
    1) 'OBS_ID', 'FILTER' or 2) 'rela_path', to_file=F
    Outpus: 
    absolute path, string
    """
    if to_file == True:
        print(rela_path)
        data_path = '../data/Borisov_raw/'
        map_path = './uvot/image/'
        file_name = 'sw' + rela_path[0] + rela_path[1] + '_sk.img.gz'
        path = os.path.join(thisdir, data_path, rela_path[0], map_path, file_name)
        print(path)
        return path
    else:
        return os.path.join(thisdir, rela_path[0])