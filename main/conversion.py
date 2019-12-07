import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from tools import *

def filt_para(filt):
    para = {'v': {'fwhm': 769, 'zp': 17.89, 'zp_err': 0.013, 'cf': 2.61e-16, 'cf_err': 2.4e-18},
            'b': {'fwhm': 975, 'zp': 19.11, 'zp_err': 0.016, 'cf': 1.32e-16, 'cf_err': 9.2e-18},
            'u': {'fwhm': 785, 'zp': 18.34, 'zp_err': 0.020, 'cf': 1.5e-16, 'cf_err': 1.4e-17},
            'uw1': {'fwhm': 693, 'zp': 17.49, 'zp_err': 0.03, 'cf': 4.3e-16, 'cf_err': 2.1e-17, 'rf': 0.1375},
            'um2': {'fwhm': 498, 'zp': 16.82, 'zp_err': 0.03, 'cf': 7.5e-16, 'cf_err': 1.1e-17},
            'uw2': {'fwhm': 657, 'zp': 17.35, 'zp_err': 0.04, 'cf': 6.0e-16, 'cf_err': 6.4e-17}
            }
    return para[filt]

def cr2mag(cr, cr_err, filt):
    """use zero points to transfer counts to mag
    """
    zp = filt_para(filt)['zp']
    mag = zp - 2.5*np.log10(cr)
    mag_err_1 = 2.5*cr_err/(np.log(10)*cr)
    mag_err_2 = filt_para(filt)['zp_err']
    mag_err = np.sqrt(mag_err_1**2 + mag_err_2**2)
    return mag, mag_err

def cr2equi_flux(cr, cr_err, filt):
    # read ea
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/20. # A to nm
    ea_area = ea_data['SPECRESP']
    delta_wave = ea_wave[1] - ea_wave[0]
    factor = 0
    for i in range(len(ea_wave)):
        factor += ea_area[i]*ea_wave[i]*delta_wave*1e8*5.034116651114543
    equi_flux = cr/(factor*10) # 10: nm to A
    equi_flux_err = cr_err/(factor*10)
    snr = equi_flux/equi_flux_err
    return equi_flux, equi_flux_err, snr

def cr2sb(cr, cr_err, filt, solid_angle):
    """convert count rate to surface brightness
    unit: W m-2 sr-1

    cr to flux density: Poole et al. 2008
    flux den to flux: flux den*FWHM
    flux to sf: flux*factors/solid_angle
    factors: 1 arcsec2 = 2.35e-11 sr
             1 erg s-1 = 1e-7W
             1 cm2 = 1e-4m2
    """
    fwhm = filt_para(filt)['fwhm']
    cf = filt_para(filt)['cf']
    cf_err = filt_para(filt)['cf_err']
    factors = 1. #TODO:
    sb = cr*cf*fwhm*factors/solid_angle
    sb_err_1 = cr_err*cf
    sb_err_2 = cf_err*cr
    sb_err = (fwhm*factors/solid_angle)*np.sqrt(sb_err_1**2 + sb_err_2**2)
    return sb, sb_err

def cr2flux(cr, cr_err, filt):
    fwhm = filt_para(filt)['fwhm']
    cf = filt_para(filt)['cf']
    cf_err = filt_para(filt)['cf_err']
    flux = cr*cf*fwhm*(4/2.35)
    flux_err = fwhm*np.sqrt(np.power(cr_err*cf,2)+np.power(cf_err*cr,2))*(4/2.35)
    return flux, flux_err

def mag_sb_flux_from_spec(spec_name, filt):
    """use effective area and theoretical spectra
    to calculate apparent magnitude
    """
    # read spectra
    spec_path = get_path('../data/auxil/'+spec_name)
    spec_wave = np.loadtxt(spec_path)[:, 0]
    spec_flux = np.loadtxt(spec_path)[:, 1]#*2.720E-4 # flux moment to irradiance
    if np.min(spec_wave)<1000:
        spec_wave = spec_wave*10
    # read ea
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/2#0. # A to nm
    ea_area = ea_data['SPECRESP']
    # interpolate ea to cater for spec
    ea = interpolate.interp1d(ea_wave, ea_area, fill_value='extrapolate')
    spec_ea = ea(spec_wave)
    wave_min = max([np.min(spec_wave), np.min(ea_wave)])
    wave_max = min([np.max(spec_wave), np.max(ea_wave)])
    spec = np.c_[np.c_[spec_wave, spec_flux.T], spec_ea.T]
    spec_reduce = spec[spec[:,0]>wave_min, :]
    spec_reduce = spec_reduce[spec_reduce[:,0]<wave_max, :]
    spec = spec_reduce[spec_reduce[:,2]>0, :]
    # integral
    delta_wave = spec[1, 0] - spec[0, 0]
    cr = 0
    for i in range(len(spec)):
        cr += spec[i, 0]*spec[i, 1]*spec[i, 2]*delta_wave*1e7*5.034116651114543 #10^8 for Kurucz
    # cr to mag
    return cr, cr2mag(cr, 0, filt), cr2sb(cr, 0, filt, 1.), cr2flux(cr, 0, filt)

def flux2num(flux, flux_err, g_name, obs_log_name, 
             method, horizon_id):
    # load g factor file
    g_path = get_path('../docs/'+g_name)
    g_file = np.loadtxt(g_path, skiprows=3)
    helio_v_list = g_file[:, 0]
    g_1au_list = (g_file[:,1]+g_file[:,2]+g_file[:,3])*1e-16
    # interpolate
    g_1au = interpolate.interp1d(helio_v_list, g_1au_list, 
                                 kind='cubic', fill_value='extrapolate')
    # load obs log
    obs_log_path = get_path('../docs/'+obs_log_name)
    obs_log = pd.read_csv(obs_log_path, sep=' ', 
                          index_col=['FILTER'])
    obs_log = obs_log[['HELIO', 'HELIO_V', 'START', 'END', 'OBS_DIS']]
    obs_log = obs_log.loc['UVW1']
    if obs_log.index.name == 'FILTER':
        r_list = obs_log['HELIO']
        rv_list = obs_log['HELIO_V']
        delta_list = obs_log['OBS_DIS']
        start = obs_log['START'].iloc[0]
        end = obs_log['END'].iloc[-1]
        #!print('start time: '+start)
        #!print('end time: '+end)
        # calculate mean helio velocity
        if method == 'both_ends':
            mean_r = (r_list.iloc[0]+r_list.iloc[-1])/2.
            mean_rv = (rv_list.iloc[0]+rv_list.iloc[-1])/2.
            mean_delta = (delta_list.iloc[0]+delta_list.iloc[-1])/2.
        elif method == 'mid_time':
            dt = Time(end)-Time(start)
            mid_time = Time(start)+1/2*dt
            obj = Horizons(id=horizon_id,
                        location='@swift',
                        epochs=mid_time.jd)
            eph = obj.ephemerides()[0]
            mean_r = eph['r']
            mean_rv = eph['r_rate']
            mean_delta = eph['delta']
        elif method == 'all':
            mean_r = np.mean(r_list)
            mean_rv = np.mean(rv_list)
            mean_delta = np.mean(delta_list)
    else:
        start = obs_log['START']
        end = obs_log['END']
        mean_r = obs_log['HELIO']
        mean_rv = obs_log['HELIO_V']
        mean_delta = obs_log['OBS_DIS']
    # return num
    lumi = flux*4*np.pi*np.power(au2km(mean_delta)*1000*100, 2)
    lumi_err = flux_err*4*np.pi*np.power(au2km(mean_delta)*1000*100, 2)
    mean_g = g_1au(mean_rv)/np.power(mean_r, 2)
    num = lumi/mean_g
    #!print('flux to num: '+str(4*np.pi*np.power(au2km(mean_delta)*1000*100, 2)/mean_g))
    num_err = lumi_err/mean_g
    #!print('mid-time r: '+str(mean_r)+' (AU)'+'\n'
    #      +'mid-time rv: '+str(mean_rv)+' (km/s)'+'\n'
    #      +'mid-time delta: '+str(mean_delta)+' (AU)'+'\n'
    #      +'mid-time g factor (for 3 total lines): '+str(mean_g)+' (erg s-1 mol-1)')
    return num, num_err

def num_assu(wvm_name, aperture):
    # load col density from wvm file
    dis = []
    col_den = []
    wvm_path = get_path('../docs/'+wvm_name)
    wvm_file = open(wvm_path)
    wvm_file_lines = wvm_file.readlines()
    wvm_file.close()
    delta = wvm_file_lines[2].split()
    delta = float(delta[13])
    for line in wvm_file_lines[52:70]:
        line = line[:-1]
        line = line.split()
        line = [float(i) for i in line]
        dis.append(line[0]*1000*100)
        dis.append(line[2]*1000*100)
        dis.append(line[4]*1000*100)
        dis.append(line[6]*1000*100)
        col_den.append(line[1])
        col_den.append(line[3])
        col_den.append(line[5])
        col_den.append(line[7])
    # interpolate
    dis2col = interpolate.interp1d(dis, col_den, 
                                   kind='quadratic', fill_value='extrapolate') 
    if not aperture:
        aperture = wvm_file_lines[72].split()
        aperture = float(aperture[3])
    step_num = int(aperture*100)#10000.
    aperture = au2km(as2au(aperture, delta))*1000*100. # arcsec to cm
    dis_list = np.linspace(0, aperture, step_num)
    dis_list = (dis_list[1:]+dis_list[:-1])/2.
    step = dis_list[1]-dis_list[0]
    col_list = dis2col(dis_list)
    num_assu = 0
    # integral
    for i in range(int(step_num)-1):
        area = 2*np.pi*dis_list[i]*step
        num_assu = num_assu + area*col_list[i]
    return num_assu

def num2q(num, num_err, wvm_name, aperture=False):
    """ covert number to production rate
        from web vectoria model
    """
    # get assumed num of the model within the aperture
    num_model = num_assu(wvm_name, aperture)
    # readin the assumed Q_H2O
    wvm_path = get_path('../docs/'+wvm_name)
    wvm_file = open(wvm_path)
    wvm_file_lines = wvm_file.readlines()
    wvm_file.close()
    q_assu = wvm_file_lines[6].split()
    q_assu = float(q_assu[4])
    # ratio -> actual Q_H2O
    q = (q_assu/num_model)*num
    #! print('num to q: '+str(q_assu/num_model))
    q_err = (q_assu/num_model)*num_err
    return q, q_err