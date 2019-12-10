import numpy as np
from astropy.io import fits
from itertools import islice
from tools import get_path
from conversion import *

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

def mask(img_name, lowlim, uplim):
    # 1: unmasked, 0: masked
    img_data = load_img(img_name)
    mask_img = np.ones(img_data.shape)
    i_range = np.arange(len(img_data))
    j_range = np.arange(len(img_data[0]))
    for i in i_range:
        for j in j_range:
            if img_data[i,j]>uplim or img_data[i,j]<lowlim:
                mask_img[i,j] = 0
    return mask_img

def limit_loop(img_name, center, r):
    if isinstance(img_name, str) == True:
        img_data = load_img(img_name)
    else:
        img_data = img_name
    i_range = np.arange(len(img_data))
    j_range = np.arange(len(img_data[0]))
    cen_pix = [center[1]-1, center[0]-1]
    i_range = i_range[i_range > cen_pix[0]-r]
    i_range = i_range[i_range < cen_pix[0]+r]
    j_range = j_range[j_range > cen_pix[1]-r]
    j_range = j_range[j_range < cen_pix[1]+r]
    return img_data, cen_pix, i_range, j_range

def mask_region(mask_img, mask_name):
    if isinstance(mask_img, str):
        img_data = load_img(mask_img)
        mask_img = np.ones(img_data.shape)
    regions = load_reg_list(mask_name)
    x_list = regions[1]
    y_list = regions[0]
    r_list = regions[2]
    center_list = list(zip(x_list, y_list))
    for k in range(len(r_list)):
        img_data, cen_pix, i_range, j_range = limit_loop(mask_img, center_list[k], r_list[k])
        for i in i_range:
            for j in j_range:
                pos_pix = [i, j]
                pos_cen_dist = get_dist(cen_pix, pos_pix)
                if pos_cen_dist < r_list[k]:
                    mask_img[i,j] = 0
    return mask_img

def circle_ct(img_name, center, r, method='mean', mask_img = False):
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, center, r)
    if isinstance(mask_img, bool):
        mask_img = np.ones(img_data.shape)
    pixel = 0
    count = 0
    pixel_unmask = 0
    count_list = []
    for i in i_range:
        for j in j_range:
            pos_pix = [i, j]
            pos_cen_dist = get_dist(cen_pix, pos_pix)
            if pos_cen_dist < r:
                pixel_unmask += 1
                pixel += mask_img[i, j]
                count += img_data[i, j]*mask_img[i, j]
                if mask_img[i, j] == 1:
                    count_list.append(img_data[i, j])
    if method == 'mean':
        count_unmask = (count/pixel)*pixel_unmask
        return count_unmask, pixel_unmask
    elif method == 'median':
        count_unmask = np.median(np.array(count_list))*pixel_unmask
        return count_unmask, pixel_unmask, np.std(np.array(count_list))*pixel_unmask
    else:
        pass

def donut_ct(img_name, center, r, method='mean', mask_img = False):
    r1 = r[0]
    r2 = r[1]
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, center, r2)
    if isinstance(mask_img, bool):
        mask_img = np.ones(img_data.shape)
    pixel = 0
    pixel_unmask = 0
    count = 0
    count_list = []
    for i in i_range:
        for j in j_range:
            pos_pix = [i, j]
            pos_cen_dist = get_dist(cen_pix, pos_pix)
            if pos_cen_dist > r1 and pos_cen_dist < r2:
                pixel_unmask += 1
                pixel += mask_img[i, j]
                count += img_data[i, j]*mask_img[i, j]
                if mask_img[i, j] == 1:
                    count_list.append(img_data[i,j])
    if method == 'mean':
        return count, pixel
    elif method == 'median':
        count_unmask = np.median(np.array(count_list))*pixel_unmask
        count_err = np.std(np.array(count_list))*pixel_unmask
        return count_unmask, pixel_unmask, count_err
    else:
        pass

def azimuthal_ct(img_name,
                 center, aperture, 
                 step_num, 
                 method,
                 mask_img = False,
                 start = False):
    if not start:
        start = 0
    step = (aperture-start)/step_num
    count = 0
    pixel = 0
    err2 = 0
    for i in range(0, step_num):
        r_i = (i*step+start,(i+1)*step+start)
        result = donut_ct(img_name, 
                          center, r_i, 
                          method, 
                          mask_img)
        count += result[0]
        pixel += result[1]
        if method == 'median':
            err2 += np.power(result[2],2)
    if method == 'mean':
        return count, pixel
    elif method == 'median':
        return count, pixel, np.sqrt(err2)
    else:
        pass

def multi_circle_ct(img_name, center_list, r_list, method='mean', mask_img = False):
    count_list = []
    pixel_list = []
    for n in range(len(r_list)):
        result = circle_ct(img_name, center_list[n], r_list[n], method, mask_img)
        count_list.append(result[0])
        pixel_list.append(result[1])
    return count_list, pixel_list

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

def reg2bg_bri(img_name, bg_method, bg_center, bg_r, count_method, mask_img = False):
    if bg_method == 'single':
        result = circle_ct(img_name, bg_center, bg_r, count_method, mask_img)
    elif bg_method == 'donut':
        result = donut_ct(img_name, bg_center, bg_r, count_method, mask_img)
    elif bg_method == 'multi':
        result = multi_circle_ct(img_name, bg_center, bg_r, count_method, mask_img)[:2]
        bg_count = np.array(result[0])
        bg_pixel = np.array(result[1])
    else:
        raise Exception('check the input of method (single/dount/multi)')
    if count_method == 'mean':
        if bg_method == 'multi':
            bg_bri = bg_count/bg_pixel
            bg_bri = np.mean(bg_bri)
            n = len(bg_center)
            bg_bri_err = (1/n)*np.sqrt(np.sum(abs(bg_count)/np.power(bg_pixel, 2)))
        else:
            bg_bri = result[0]/result[1]
            bg_bri_err = np.sqrt(abs(result[0]))/result[1]
    elif count_method == 'median':
        if bg_method == 'multi':
            bg_bri = bg_count/bg_pixel
            bg_bri = np.median(bg_bri)
            bg_bri_err = np.std(bg_bri)
        else:
            bg_bri = result[0]/result[1]
            bg_bri_err = result[2]/result[1]
    else:
        raise Exception('check the input of method (mean/median)')
    return bg_bri, bg_bri_err

def aper_phot_cr(src_count, src_pixel, 
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
    snr = abs(net_cr/net_cr_err)
    return net_cr, net_cr_err, snr

def aper_phot(img_name, filt,
              src_center, src_r,
              bg_center, bg_r,
              src_method, bg_method,
              step_num=5, mask_img = False, start = False):
    '''
    src_method = 'total_mean' or 'total_median'
                 or 'azim_mean' or 'azim_median'
    bg_method = 'single_mean' or 'single_median'
                or 'donut_mean' or 'donut_median'
                or 'multi_mean'
    
    return cr, cr_err, snr, mag, mag_err, bg_cr, bg_cr_err
    '''
    # src photometry: get src_count, src_pixel
    src_shape = src_method.split('_')[0]
    src_stat = src_method.split('_')[1]
    if src_shape == 'total':
        if not start:
            result = circle_ct(img_name, 
                               src_center, src_r, 
                               src_stat, 
                               mask_img)
        else:
            result = donut_ct(img_name,
                              src_center, (start, src_r),
                              src_stat,
                              mask_img)

    elif src_shape == 'azim':
        result = azimuthal_ct(img_name,
                              src_center, src_r, 
                              step_num, 
                              src_stat,
                              mask_img, start)
    else:
        RaiseExcept('Please check your source photometry method!')
    src_count = result[0]
    src_pixel = result[1]
    # bg photometry: get bg_bri
    bg_shape = bg_method.split('_')[0]
    bg_stat = bg_method.split('_')[1]
    bg_bri, bg_bri_err = reg2bg_bri(img_name, bg_shape, 
                                    bg_center, bg_r, 
                                    bg_stat, False)
    # photometry
    if filt:
        exposure = float(load_header(img_name)['EXPTIME'])
    else:
        exposure = 1.
    if src_stat == 'mean':
        src_count_err = np.sqrt(abs(src_count))
    elif src_stat == 'median':
        src_count_err = result[2]
    cr, cr_err, snr = aper_phot_cr(src_count, src_pixel,
                                   bg_bri, exposure,
                                   src_count_err, bg_bri_err)
    if filt:
        mag, mag_err = cr2mag(cr, cr_err, filt)
    else:
        mag = float('NaN')
        mag_err = float('NaN')
    bg_cr = bg_bri/exposure
    bg_cr_err = bg_bri_err/exposure
    return (cr, cr_err), snr, (mag, mag_err), (bg_cr, bg_cr_err)

