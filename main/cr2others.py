import numpy as np
from astropy.io import fits
from tools import get_path
from scipy import interpolate
from aper_phot import *

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
    flux = cr*cf*fwhm
    flux_err = fwhm*np.sqrt(np.power(cr_err*cf,2)+np.power(cf_err*cr,2))
    return flux, flux_err

def mag_sb_flux_from_spec(spec_name, filt):
    """use effective area and theoretical spectra
    to calculate apparent magnitude
    """
    # read spectra
    spec_path = get_path('../data/auxil/'+spec_name)
    spec_wave = np.loadtxt(spec_path)[:, 0]
    spec_flux = np.loadtxt(spec_path)[:, 1]#*2.720E-4 # flux moment to irradiance
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

def cr2OH(method, 
            sun_spec_name, img_name, 
            src_center_uw1, src_r_uw1, 
            bg_center_uw1, bg_r_uw1,
            src_center_v, src_r_v,
            bg_center_v, bg_r_v):
    """OH = alpha*(UW1 - beta*V)
    """
    # alpha
    alpha = 0.5
    #beta
    sun_cr_uw1, sun_mag_uw1, sun_sb_uw1, sun_flux_uw1 = mag_sb_flux_from_spec(sun_spec_name, 'uw1')
    sun_cr_v, sun_mag_v, sun_sb_v, sun_flux_v = mag_sb_flux_from_spec(sun_spec_name, 'v')
    if method == 'flux':
        beta = sun_flux_uw1[0]/sun_flux_v[0]
        beta_err = error_prop('div', 
                                sun_flux_uw1[0], sun_flux_uw1[1],
                                sun_flux_v[0], sun_flux_v[1])
    elif method == 'cr':
        beta = sun_cr_uw1/sun_cr_v
        beta_err = 0.
    # comet photometry
    phot_uw1 = aper_phot_multi(img_name_uw1, 'uw1', 
                                src_center_uw1, src_r_uw1, 
                                bg_center_uw1, bg_r_uw1)
    phot_v = aper_phot_multi(img_name_uvv, 'v', 
                                src_center_uvv, src_r_uvv, 
                                bg_center_uvv, bg_r_uvv)
    if method == 'flux':
        flux_uw1, flux_err_uw1 = cr2flux(phot_uw1[0][0], phot_uw1[0][1], 'uw1')
        flux_v, flux_err_v = cr2flux(phot_v[0][0], phot_v[0][1], 'v')
    # calculate OH
    if method == 'cr':
        cr = alpha*(phot_uw1[0][0] - beta*phot_v[0][0])
        b_c_err = error_prop('mul',
                                beta, beta_err,
                                phot_v[0][0], phot_v[0][1])
        cr_err = alpha*error_prop('sub',
                                    phot_uw1[0][0], phot_uw1[0][1],
                                    beta*phot_v[0][0], b_c_err)
        return cr, cr_err
    if method == 'flux':
        flux = alpha*(flux_uw1 - beta*flux_v)
        e_f_err = error_prop('mul', beta, beta_err, flux_v, flux_err_v)
        flux_err = alpha*error_prop('sub', 
                                    flux_uw1, flux_err_uw1,
                                    beta*flux_v, b_f_err)
        return flux, flux_err
            
