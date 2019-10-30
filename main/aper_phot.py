import numpy as np
from astropy.io import fits
from tools import get_path
from scipy import interpolate

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

def cr2mag(cr, cr_err, filt):
    """use zero points to transfer counts to mag
    """
    if filt == 'uw1':
        zp = 17.44
    elif filt == 'v':
        zp = 17.89
    mag = zp - 2.5*np.log10(cr)
    if cr_err == False:
        return mag
    else:
        mag_err = 2.5*cr_err/(np.log(10)*cr)
        return mag, mag_err

def cr2mean_flux(cr, cr_err, filt):
    # read ea
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/20. # A to nm
    ea_area = ea_data['SPECRESP']
    delta_wave = ea_wave[1] - ea_wave[0]
    factor = 0
    for i in range(len(ea_wave)):
        factor += ea_area[i]*ea_wave[i]*delta_wave*1e8*5.034116651114543
    mean_flux = cr/(factor*10) # 10: nm to A
    mean_flux_err = cr_err/(factor*10)
    snr = mean_flux/mean_flux_err
    return mean_flux, mean_flux_err, snr

def mag_flux_from_spec(spec_name, filt):
    """use effective area and theoretical spectra
    to calculate apparent magnitude
    """
    # read spectra
    spec_path = get_path('../data/auxil/'+spec_name)
    spec_wave = np.loadtxt(spec_path)[:, 0]
    spec_flux = np.loadtxt(spec_path)[:, 1]*2.720E-4 # flux moment to irradiance
    # read ea
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/20. # A to nm
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
    flux = 0
    A = 0
    for i in range(len(spec)):
        cr += spec[i, 0]*spec[i, 1]*spec[i, 2]*delta_wave*1e8*5.034116651114543
        flux += spec[i, 1]*spec[i, 2]*delta_wave
        A += spec[i, 2]*delta_wave
    mean_flux = flux/A
    mean_flux = mean_flux/10. # nm to A
    # cr to mag
    return cr2mag(cr, False, filt), mean_flux