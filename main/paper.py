import numpy as ny
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import matplotlib.pyplot as plt
import os
import re
from itertools import combinations
from tools import *
from before_stack import *
from conversion import *
from aper_phot import *
from cr_to_flux import *
from profile_q import *

def obs_log(obs_log_name, horizon_id):
    obs_log_path = get_path('../docs/'+obs_log_name)
    obs_log = pd.read_csv(obs_log_path, sep=' ', index_col=['FILTER'])
    obs_log = obs_log[['START', 'END', 'EXP_TIME']]
    if obs_log.index.name == 'FILTER':
        start = obs_log['START'].iloc[0]
        end = obs_log['END'].iloc[-1]
    else:
        start = obs_log['START']
        end = obs_log['END']
    dt = Time(end)-Time(start)
    mid_time = Time(start)+1/2*dt
    obj = Horizons(id=horizon_id,
                   location='@swift',
                   epochs=mid_time.jd)
    eph = obj.ephemerides()[0]
    r = eph['r']
    rv = eph['r_rate']
    delta = eph['delta']
    phase = eph['alpha']
    exp_v = 0
    exp_uw1 = 0
    # exp time
    obs_log_v = obs_log.loc['V']
    exp_v_list = list(obs_log_v['EXP_TIME'])
    exp_v = sum(exp_v_list)
    obs_log_uw1 = obs_log.loc['UVW1']
    exp_uw1_list = list(obs_log_uw1['EXP_TIME'])
    exp_uw1 = sum(exp_uw1_list)
    tp = 2458826.0535289953
    dtp = mid_time.jd - tp
    obs_table = {'start':start, 'end':end, 
                 'midtime':mid_time, 'tp': dtp,
                 'r':r, 'rv':rv, 'delta':delta,
                 'phase':phase,'exp_v':exp_v,'exp_uw1':exp_uw1}
    return obs_table

def para(mon):
    global img_name_uw1, img_name_v, spec_name_sun, spec_name_OH, img_name, err2_name
    global bg_reg_name, mask_name, mask_img, obs_log_name, wvm_name, phase_name
    global src_r, red, z
    global bg_r, bg_center, src_center
    img_name_uw1 = mon+'_stack_uw1.fits'
    img_name_v = mon+'_stack_uvv.fits'
    spec_name_sun = 'sun_1A.txt'
    spec_name_OH = '2019-07-15_emission_models_OH.txt'
    src_center = (1000, 1000)
    bg_reg_name = mon+'_sub_bg.reg'
    mask_name = mon+'_mask.reg'
    obs_log_name = mon+'_obs-log_Borisov_part.txt'
    wvm_name = mon+'_wvm.txt'
    phase_name = 'phase_correction.txt'
    if mon == 'sep':
        src_r = 45
        red = 0
        z = 2.859E+16
    elif mon == 'nov':
        src_r = 57
        red = 19
        z = 5.025E+16
    elif mon == 'dec':
        src_r = 67
        red = 14
        z = 6.278E+16
    elif mon == 'latedec':
        src_r = 71
        red = 13
        z = 6.146E+16
    else:
        pass
    img_name = mon+'_sub_red'+str(int(red))+'.fits'
    err2_name = mon+'_sub_red'+str(int(red))+'_err.fits'
    mask_img = mask_region(img_name, mask_name)
    bg = load_reg_list(bg_reg_name)
    bg_x = bg[1]
    bg_y = bg[0]
    bg_r = bg[2]
    bg_center = list(zip(bg_x, bg_y))

def oh(mon):
    para(mon)
    img_path = get_path('../docs/'+img_name)
    err2_path = get_path('../docs/'+err2_name)
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(err2_path):
        os.remove(err2_path)
    fits_sub(img_name_uw1, img_name_v, mon, red)
    mask_img = mask_region(img_name, mask_name)
    result = aper_phot(img_name, False,
                       src_center, src_r,
                       bg_center, bg_r,
                       'azim_median', 'multi_mean',
                       int(src_r/2.), mask_img, 0.)
    cr = result[0][0]
    bg_bri = result[3][0]
    # err
    src_err, pixel = circle_ct(err2_name, src_center, src_r, 'mean', mask_img)
    pixel_unmask = circle_ct(err2_name, src_center, src_r, 'median', mask_img)[1]
    src_err = (np.sqrt(src_err)*1.253/pixel)*pixel_unmask
    bg_bri_err, pixel = multi_circle_ct(err2_name, bg_center, bg_r, 'mean', mask_img)
    bg_bri_err = np.array(bg_bri_err)
    pixel = np.array(pixel)
    bg_bri_err = np.sqrt(np.sum(bg_bri_err/np.power(pixel,2)))/len(pixel)
    bg_err = bg_bri_err*pixel_unmask
    cr_err = np.sqrt(np.power(src_err,2)+np.power(bg_err,2))
    flux = 1.2750906353215913e-12*cr
    flux_err = 1.2750906353215913e-12*cr_err
    mean_delta = obs_log(mon+'_obs-log_Borisov_part.txt', 90004424)['delta']
    lumi = flux*4*np.pi*np.power(au2km(mean_delta)*1000*100, 2)
    num, num_err = flux2num(flux, flux_err,
                            'fluorescenceOH.txt',
                            obs_log_name,
                            method='both_ends',
                            horizon_id=90004424,
                            if_show=False)
    g_factor = lumi/num
    q, q_err = num2q(num, num_err,
                     wvm_name, src_r,
                     if_show=False)
    active_area = q/z
    active_area_err = q_err/z
    r = np.sqrt(active_area/(4*np.pi))
    r_err = active_area_err/(4*np.sqrt(np.pi*active_area))
    src_r_km = au2km(as2au(src_r,obs_log(obs_log_name, 90004424)['delta']))
    return (src_r, src_r_km), (cr, cr_err), (bg_bri, bg_bri_err), g_factor, (num, num_err), (q, q_err), (active_area/1e10, active_area_err/1e10), (r/1e5, r_err/1e5)

def aperture_phot(mon):
    para(mon)
    result_uw1 = aper_phot(img_name_uw1, 'uw1', 
                           src_center, src_r,
                           bg_center, bg_r,
                           'azim_median', 'multi_mean',
                           int(src_r/2.), mask_img, 0.)

    cr_uw1, cr_uw1_err = result_uw1[0]
    snr_uw1 = result_uw1[1]
    mag_uw1, mag_uw1_err = result_uw1[2]
    bg_cr_uw1, bg_cr_uw1_err = result_uw1[3]

    result_v = aper_phot(img_name_v, 'v', 
                         src_center, src_r,
                         bg_center, bg_r,
                         'azim_median', 'multi_mean',
                         int(src_r/2.), mask_img, 0.)

    cr_v, cr_v_err = result_v[0]
    snr_v = result_v[1]
    mag_v, mag_v_err = result_v[2]
    bg_cr_v, bg_cr_v_err = result_v[3]
    
    flux_uw1, flux_uw1_err = flux_ref_uw1(spec_name_sun,
                                          spec_name_OH,
                                          cr_uw1, cr_uw1_err,
                                          cr_v, cr_v_err, red)
    flux_v, flux_v_err = flux_ref_v(spec_name_sun, 
                                    spec_name_OH, 
                                    cr_uw1, cr_uw1_err, 
                                    cr_v, cr_v_err, red)
    
    return (cr_uw1, cr_uw1_err), snr_uw1, (mag_uw1, mag_uw1_err), (bg_cr_uw1, bg_cr_uw1_err), (flux_uw1, flux_uw1_err), (cr_v, cr_v_err), snr_v, (mag_v, mag_v_err), (bg_cr_v, bg_cr_v_err), (flux_v, flux_v_err)

def two_afr(mon):
    para(mon)
    if mon == 'sep':
        aper = 5.3
    elif mon == 'nov':
        aper = 5.7
    elif mon == 'dec':
        aper = 6.7
    elif mon == 'latedec':
        aper = 7.1
    result_v = aper_phot(img_name_v, 'v', 
                         src_center, aper,
                         bg_center, bg_r,
                         'azim_median', 'multi_mean',
                         int(aper/2), mask_img, 0.)
    mag_v, mag_v_err = result_v[2]
    afr, afr_err = mag2afr(90004424, 
                           mag_v, mag_v_err, 
                           aper, obs_log_name, 
                           phase_name, False)
    afr_corr, afr_corr_err = mag2afr(90004424, 
                                     mag_v, mag_v_err, 
                                     aper, obs_log_name, 
                                     phase_name, True)
    return (afr, afr_err), (afr_corr, afr_corr_err)

def latex_obg_log_table():
    print('\\begin{tabular}{c{4cm}c{4cm}c{2cm}c{2cm}c{2cm}c{2cm}c{2cm}c{3cm}c{3cm}}'+'\n'
          +'    \hline'+'\n'
          +'    \hline'+'\n'
          +'    Start Time& End Time& $r_h$& $dr_h$& $\Delta$& S-T-O& UVW1 $T_{exp}$& V $T_{exp}$ \\\\'+'\n'
          +'    && (AU)& (km/s)& (AU)& (\degree)& (s)& (s) \\\\'+'\n'
          +'    \hline')
    for mon in ['sep', 'nov', 'dec', 'latedec']:
        obs = obs_log(mon+'_obs-log_Borisov.txt', 90004424)
        obs_part = obs_log(mon+'_obs-log_Borisov_part.txt', 90004424)
        print('    '+obs['start']+'& '+obs['end']+'& '+str(round(obs['r'],2))+'& '+str(round(obs['rv'],2))+'& '+str(round(obs['delta'],2))+'& '+str(round(obs['phase'],2))+'& '+str(round(obs['exp_uw1'],2))+' ('+str(round(obs_part['exp_uw1'],2))+')'+'& '+str(round(obs['exp_v'],2))+' ('+str(round(obs_part['exp_v'],2))+')'+' \\\\')
    print('    \hline')

def format_trans(value, err):
    s=re.compile('E')
    f = '.1E'
    pm = '\\textpm'
    if abs(value) > abs(err):
        a = value
        b = err
    else:
        a = err
        b = value
    str_a = format(a,f)
    order_a = float(s.split(str_a)[1])
    value_b = float(s.split(format(b,f))[0])
    order_b = float(s.split(format(b,f))[1])
    order_diff = order_b - order_a
    value_b = value_b*np.power(10, order_diff)
    value_b_str = str(round(value_b,1))
    value_a_str = s.split(str_a)[0]
    order_str = s.split(str_a)[1]
    if abs(value) > abs(err):
        if float(order_str) == 0:
            return value_a_str+pm+value_b_str
        else:
            return '('+value_a_str+pm+value_b_str+')E'+order_str
    else:
        if float(order_str) == 0:
            value_b_str+pm+value_a_str
        else:
            return '('+value_b_str+pm+value_a_str+')E'+order_str
    
def latex_result_table():
    a = '& '
    for mon in ['sep', 'nov', 'dec', 'latedec']:
        if mon == 'sep':
            red = 0
            afr_r = 5.3
            afr_r_km = 0.1
        elif mon == 'nov':
            red = 19
            afr_r = 5.7
            afr_r_km = 0.1
        elif mon == 'dec':
            red = 14
            afr_r = 6.7
            afr_r_km = 0.1
        elif mon == 'latedec':
            red = 13
            afr_r = 7.1
            afr_r_km = 0.1
        else:
            pass
        obs_part = obs_log(mon+'_obs-log_Borisov_part.txt', 90004424)
        midtime = obs_part['midtime']
        tp = obs_part['tp']
        
        result = oh(mon)
        src_r, src_r_km = result[0]
        cr, cr_err = result[1]
        flux_oh = 1.2750906353215913e-12*cr
        flux_oh_err = 1.2750906353215913e-12*cr_err
        bg_bri, bg_bri_err = result[2]
        g_factor = result[3]
        num, num_err = result[4]
        q, q_err = result[5]
        active_area, active_area_err = result[6]
        r, r_err = result[7]
        
        result = aperture_phot(mon)
        cr_uw1, cr_uw1_err = result[0]
        snr_uw1 = result[1]
        mag_uw1, mag_uw1_err = result[2]
        bg_cr_uw1, bg_cr_uw1_err = result[3]
        flux_uw1, flux_uw1_err = result[4]
        cr_v, cr_v_err = result[5]
        snr_v = result[6]
        mag_v, mag_v_err = result[7]
        bg_cr_v, bg_cr_v_err = result[8]
        flux_v, flux_v_err = result[9]
        
        result = two_afr(mon)
        afr, afr_err = result[0]
        afr_corr, afr_corr_err = result[1]

        tp = str(round(tp,1))
        rfov = str(int(src_r))+'/'+str(round(src_r_km/1e5,1))+' ($Q$_\\mathrm{H_2O})'
        rfov_afr = str(round(afr_r,1))+'/'+str(round(afr_r_km,1))+' ($Af\\rho$)'
        cr_v = str(round(cr_v,1))+'\\textpm'+str(round(cr_v_err,1))
        m_v = str(round(mag_v,1))+'\\textpm'+str(round(mag_v_err,1))
        flux_v = str(round(flux_v/1e-12,1))+'\\textpm'+str(round(flux_v_err/1e-12,1))
        cr_uw1 = str(round(cr_uw1,1))+'\\textpm'+str(round(cr_uw1_err,1))
        m_uw1 = str(round(mag_uw1,1))+'\\textpm'+str(round(mag_uw1_err,1))
        flux_uw1 = str(round(flux_uw1/1e-12,1))+'\\textpm'+str(round(flux_uw1_err/1e-12,1))
        cr_oh = str(round(cr,1))+'\\textpm'+str(round(cr_err,1))
        flux_oh = str(round(flux_oh/1e-12,1))+'\\textpm'+str(round(flux_oh_err/1e-12,1))
        red = str(int(red))
        g_factor = str(round(g_factor/1e-16,1))
        num = str(round(num/1e31,1))+'\\textpm'+str(round(num_err/1e31,1))
        q = str(round(q/1e26,1))+'\\textpm'+str(round(q_err/1e26,1))
        active_area = str(round(active_area,1))+'\\textpm'+str(round(active_area_err,1))
        r = str(round(r,2))+'\\textpm'+str(round(r_err,2))
        afr_str = str(round(afr*100,1))+'\\textpm'+str(round(afr_err*100,1))
        phase_corr = str(round(afr/afr_corr,2))
        afr_corr = str(round(afr_corr*100,1))+'\\textpm'+str(round(afr_corr_err*100,1))
        print (str(midtime)+a+ tp+a+ rfov+a+ 'V'+a+ cr_v+a+ m_v+a+ flux_v+a+ cr_oh+a+ flux_oh+a+ red+a+ g_factor+a+ num+a+ q+a+ active_area+a+ r+a+ afr_str+a+ phase_corr+a+ afr_corr+' \\\\')
        print (a+a+ rfov_afr+a+ 'UVW1'+a+ cr_uw1+a+ m_uw1+a+ flux_uw1+a+a+a+a+a+a+a+a+a+a+a+' \\\\')
