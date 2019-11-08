import numpy as np
from astropy.io import fits
from itertools import islice
from tools import get_path
from cr2others import *

def load_img(img_name):
    img_path = get_path('../docs/'+img_name)
    img_data = fits.open(img_path)[0].data
    return img_data

def load_header(img_name):
    img_path = get_path('../docs/'+img_name)
    img_header = fits.open(img_path)[0].header
    return img_header

def get_dist(point_1, point_2):
    i2 = (point_1[0]-point_2[0])**2
    j2 = (point_1[1]-point_2[1])**2
    dist = np.sqrt(i2+j2)
    return dist

def limit_loop(img_name, center, r):
    img_data = load_img(img_name)
    i_range = np.arange(len(img_data))
    j_range = np.arange(len(img_data[0]))
    cen_pix = [center[1]-1, center[0]-1]
    i_range = i_range[i_range > cen_pix[0]-r]
    i_range = i_range[i_range < cen_pix[0]+r]
    j_range = j_range[j_range > cen_pix[1]-r]
    j_range = j_range[j_range < cen_pix[1]+r]
    return img_data, cen_pix, i_range, j_range

def circle_ct(img_name, center, r):
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, center, r)
    pixel = 0
    count = 0
    for i in i_range:
        for j in j_range:
            pos_pix = [i, j]
            pos_cen_dist = get_dist(cen_pix, pos_pix)
            if pos_cen_dist < r:
                pixel += 1
                count += img_data[i, j]
    return count, pixel

def donut_ct(img_name, center, r):
    r1 = r[0]
    r2 = r[1]
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, center, r2)
    pixel = 0
    count = 0
    for i in i_range:
        for j in j_range:
            pos_pix = [i, j]
            pos_cen_dist = get_dist(cen_pix, pos_pix)
            if pos_cen_dist > r1 and pos_cen_dist < r2:
                pixel += 1
                count += img_data[i, j]
    return count, pixel

def multi_circle_ct(img_name, center_list, r_list):
    count_list = []
    pixel_list = []
    for n in range(len(r_list)):
        count, pixel = circle_ct(img_name, center_list[n], r_list[n])
        count_list.append(count)
        pixel_list.append(pixel)
    return count_list, pixel_list

def aper_phot(src_count, src_pixel, 
              bg_bri, exposure, 
              src_count_err, bg_bri_err):
    """aperture photometry
    """
    bg_count = bg_bri*src_pixel
    bg_count_err = bg_bri_err*src_pixel
    net_count = src_count - bg_count
    net_count_err = np.sqrt(bg_count_err**2
                            +src_count_err**2)
    net_cr = net_count/exposure
    net_cr_err = net_count_err/exposure
    snr = net_cr/net_cr_err
    return net_cr, net_cr_err, snr

def load_reg_list(reg_name):
    reg_path = get_path('../docs/'+reg_name)
    reg_file = open(reg_path)
    center_i_list = []
    center_j_list = []
    radiu_list = []
    for line in islice(reg_file, 3, None):
        reg_data = line.split('(')[1]
        reg_data = reg_data.split(')')[0]
        reg_data = reg_data.split(',')
        center_j_list.append(float(reg_data[0]))
        center_i_list.append(float(reg_data[1]))
        if len(reg_data[2:]) == 1:
            radiu_list.append(float(reg_data[2:][0]))
        else:
            radiu_list.append([float(k) for k in reg_data[2:]])
    return center_i_list, center_j_list, radiu_list

def reg2bg_bri(img_name, bg_method, bg_center, bg_r, n=3):
    if bg_method == 'single':
        bg_count, bg_pixel = circle_ct(img_name, bg_center, bg_r)
    elif bg_method == 'donut':
        bg_count, bg_pixel = dount_ct(img_name, bg_center, bg_r)
    elif bg_method == 'multi':
        bg_count, bg_pixel = multi_circle_ct(img_name, bg_center, bg_r)
        bg_count = np.array(bg_count)
        bg_pixel = np.array(bg_pixel)
    else:
        raise Exception('check the input of method')
    if bg_method == 'multi':
        bg_bri = bg_count/bg_pixel
        bg_bri = np.mean(bg_bri)
        bg_bri_err = (1/n)*np.sqrt(np.sum(bg_count/np.power(bg_pixel, 2)))
    else:
        bg_bri = bg_count/bg_pixel
        bg_bri_err = np.sqrt(bg_count)/bg_pixel
    return bg_bri, bg_bri_err

def aper_phot_multi(img_name, filt, src_center, src_r, bg_center, bg_r):
    exposure = float(load_header(img_name)['EXPTIME'])
    src_count, src_pixel = circle_ct(img_name, src_center, src_r)
    bg_bri, bg_bri_err = reg2bg_bri(img_name, 'multi', bg_center, bg_r)
    cr, cr_err, snr = aper_phot(src_count, src_pixel, \
                                bg_bri, exposure,
                                np.sqrt(src_count), bg_bri_err)
    mag, mag_err = cr2mag(cr, cr_err, filt)
    return cr, snr, mag