import numpy as np
from astropy.io import fits
from tools import get_path
from scipy import interpolate

def filt_para(filt):
    para = {'v': {'fwhm': 769, 'zp': 17.89, 'cf': 2.61e-16},
            'b': {'fwhm': 975, 'zp': 19.11, 'cf': 1.32e-16},
            'u': {'fwhm': 785, 'zp': 18.34, 'cf': 1.5e-16},
            'uw1': {'fwhm': 693, 'zp': 17.49, 'cf': 4.3e-16, 'rf': 0.138},
            'um2': {'fwhm': 498, 'zp': 16.82, 'cf': 7.5e-16},
            'uw2': {'fwhm': 657, 'zp': 17.35, 'cf': 6.0e-16}
            }
    return para[filt]

def cr2mag(cr, cr_err, filt):
    """use zero points to transfer counts to mag
    """
    zp = filt_para(filt)['zp']
    mag = zp - 2.5*np.log10(cr)
    if cr_err == False:
        return mag
    else:
        mag_err = 2.5*cr_err/(np.log(10)*cr)
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
    factors = 1. #TODO:
    sb = cr*cf*fwhm*factors/solid_angle
    if cr_err == False: 
        return sb
    else:
        sb_err = cr_err*cf*fwhm*factors/solid_angle
        return sb, sb_err

def cr2flux(cr, cr_err, filt):
    fwhm = filt_para(filt)['fwhm']
    cf = filt_para(filt)['cf']
    flux = cr*cf*fwhm
    if cr_err == False:
        return flux
    else:
        flux_err = cr_err*cf*fwhm
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
    return cr, cr2mag(cr, False, filt), cr2sb(cr, False, filt, 1.), cr2flux(cr, False, filt)