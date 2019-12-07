import numpy as np
from astropy.io import fits
from scipy import interpolate
from tools import *
from conversion import *
from aper_phot import load_img, load_header, donut_ct

def read_spec(spec_name):
    spec_path = get_path('../data/auxil/'+spec_name)
    spec_wave = np.loadtxt(spec_path)[:, 0]
    spec_flux = np.loadtxt(spec_path)[:, 1]
    if np.min(spec_wave)<1000:
        spec_wave = spec_wave*10
    return spec_wave, spec_flux

def read_ea(filt):
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/2
    ea_area = ea_data['SPECRESP']
    return ea_wave, ea_area

def sum_spec(emis_name, factor_emis, factor_emis_err,
             cont_name, factor_cont, factor_cont_err):
    emis_wave, emis_flux = read_spec(emis_name)
    cont_wave, cont_flux = read_spec(cont_name)
    emis_func = interpolate.interp1d(emis_wave, emis_flux, fill_value='extrapolate')
    cont_func = interpolate.interp1d(cont_wave, cont_flux, fill_value='extrapolate')
    wave = np.arange(np.min(cont_wave), np.max(cont_wave), 1.)
    emis = emis_func(wave)
    emis_err = np.array(emis*factor_emis_err)
    emis = np.array(emis*factor_emis)
    cont = cont_func(wave)
    cont_err = np.array(cont*factor_cont_err)
    cont = np.array(cont*factor_cont)
    flux_sum = cont+emis
    flux_sum_err = np.sqrt(np.power(emis_err, 2)+
                           np.power(cont_err, 2))
    return wave, flux_sum, flux_sum_err
    
def eff_wave(spec_name, filt):
    # filt: v or uw1 or b
    # refer to mag_sb_flux_from_spec in conversion.py
    # readin file
    spec_wave, spec_flux = read_spec(spec_name)
    ea_wave, ea_area = read_ea(filt)
    # interpolate
    if (ea_wave[1]-ea_wave[0]) > (spec_wave[1]-spec_wave[0]) :
        ea_func = interpolate.interp1d(ea_wave, ea_area, fill_value='extrapolate')
        ea_area = ea_func(spec_wave)
        wave = spec_wave
    else:
        spec_func = interpolate.interp1d(spec_wave, spec_flux, fill_value='extrapolate')
        spec_flux = spec_func(ea_wave)
        wave = ea_wave
    ea_area = ea_area[wave != 0]
    spec_flux = spec_flux[wave != 0]
    wave = wave[wave != 0]
    # integral
    nume = 0.
    deno = 0.
    delta_wave = wave[1]-wave[0]
    for i in range(len(wave)):
        nume += wave[i]*ea_area[i]*spec_flux[i]*delta_wave
        deno += ea_area[i]*spec_flux[i]*delta_wave
    return nume/deno
#    return 2634 # Vega

def bin_spec(spec_name, bin_min, bin_max):
    '''prepare for smoothing when (eff wavelength -> flux density)
    '''
    spec_wave, spec_flux = read_spec(spec_name)
    delta_wave = spec_wave[1]-spec_wave[0]
    wave = spec_wave[spec_wave>bin_min]
    flux = spec_flux[spec_wave>bin_min]
    flux = flux[wave<bin_max]
    wave = wave[wave<bin_max]
    bin_sum = (flux[0]+flux[-1])*delta_wave*(1/2)
    for i in range(len(wave[1:-1])):
        bin_sum += flux[i+1]*delta_wave
    if (wave[0]-bin_min)>delta_wave/2:
        bin_sum += spec_flux[spec_wave<bin_min][-1]*\
            (delta_wave/2-(bin_min-spec_wave[spec_wave<bin_min][-1]))
        bin_sum += flux[0]*delta_wave/2
    else: 
        bin_sum += flux[0]*(wave[0]-bin_min)
    if (bin_max-wave[-1])>delta_wave/2:
        bin_sum += spec_flux[spec_wave>bin_max][0]*\
            (delta_wave/2-(spec_wave[spec_wave>bin_max][0]-bin_max))
        bin_sum += flux[-1]*delta_wave/2
    else:
        bin_sum += flux[-1]*(bin_max-wave[-1])
    return (bin_max+bin_min)/2, bin_sum/(bin_max-bin_min)

def conv_factor(spec_name, filt):
    '''derive theoretical conversion factor (not eff wavelength of Vega!)
    '''
    # calculate eff_wave
    eff_w = eff_wave(spec_name, filt)
    # flux density at eff_wave
    step = 10
    point_1 = bin_spec(spec_name, eff_w-2*step, eff_w-step)
    point_2 = bin_spec(spec_name, eff_w-step, eff_w)
    point_3 = bin_spec(spec_name, eff_w, eff_w+step)
    point_4 = bin_spec(spec_name, eff_w+step, eff_w+2*step)
    wave = [point_1[0], point_2[0], point_3[0], point_4[0]]
    flux = [point_1[1], point_2[1], point_3[1], point_4[1]]
    func = interpolate.interp1d(wave, flux, fill_value='extrapolate')
    eff_f = func(eff_w)
    # predicted cr
    cr = mag_sb_flux_from_spec(spec_name, filt)[0]
    # conversion factor
    cf = eff_f/cr
    return cf

def cr2flux_def(cr, cr_err, spec_name, filt):
    ''' get flux from cr and theoretical cf
    '''
    # refer to cr2flux in conversion.py
    fwhm = filt_para(filt)['fwhm']
    cf = conv_factor(spec_name, filt)
    cf_err = 0. # TODO:
    flux = cr*cf*fwhm*(4/2.35482)
    flux_err = fwhm*np.sqrt(np.power(cr_err*cf,2)+np.power(cf_err*cr,2))*(4/2.35)
    return flux, flux_err

def reddening_correct(r):
    '''get the correction factor of beta
    r: %/100nm
    '''
    # read in ea
    ea_wave_uw1, ea_area_uw1 = read_ea('uw1')
    ea_wave_v, ea_area_v = read_ea('v')
    # calculate wave_uw1, wave_v
    wave_uw1 = 0
    ea_uw1 = 0
    wave_v = 0
    ea_v = 0
    delta_wave_uw1 = ea_wave_uw1[1]-ea_wave_uw1[0]
    delta_wave_v = ea_wave_v[1]-ea_wave_v[0]
    for i in range(len(ea_wave_uw1)):
        wave_uw1 += ea_wave_uw1[i]*ea_area_uw1[i]*delta_wave_uw1
        ea_uw1 += ea_area_uw1[i]*delta_wave_uw1
    wave_uw1 = wave_uw1/ea_uw1
    for i in range(len(ea_wave_v)):
        wave_v += ea_wave_v[i]*ea_area_v[i]*delta_wave_v
        ea_v += ea_area_v[i]*delta_wave_v
    wave_v = wave_v/ea_v
    # get reddening correction factor
    middle_factor = (wave_v - wave_uw1)*r/200000
    return (1-middle_factor)/(1+middle_factor)

def flux_my_cf_OH(spec_name_sun, spec_name_sum, cr_uw1, cr_uw1_err, cr_v, cr_v_err, r=0):
    '''get OH flux from theoretical cf and cr(uw1, v)
    '''
    alpha = 2
    cr_sun_uw1 = mag_sb_flux_from_spec(spec_name_sun, 'uw1')[0]
    cr_sun_v = mag_sb_flux_from_spec(spec_name_sun, 'v')[0]
    flux_sun_uw1 = cr2flux_def(cr_sun_uw1, 0, spec_name_sun,'uw1')[0]
    flux_sun_v = cr2flux_def(cr_sun_v, 0, spec_name_sun,'v')[0]
    beta = flux_sun_uw1*filt_para('uw1')['fwhm']/(flux_sun_v*filt_para('v')['fwhm'])
    beta = reddening_correct(r)*beta
    flux_v,  flux_v_err = cr2flux_def(cr_v, cr_v_err, spec_name_sum,'v')
    flux_uw1, flux_uw1_err = cr2flux_def(cr_uw1, cr_uw1_err, spec_name_sum,'uw1')
    flux_OH = alpha*(flux_uw1*filt_para('uw1')['fwhm']*4/2.35
                     -beta*flux_v*filt_para('v')['fwhm']*4/2.35)
    flux_OH_err = error_prop('sub',
                             alpha*flux_uw1*filt_para('uw1')['fwhm']*4/2.35,
                             alpha*flux_uw1_err*filt_para('uw1')['fwhm']*4/2.35,
                             alpha*beta*flux_v*filt_para('v')['fwhm']*4/2.35,
                             alpha*beta*flux_v_err*filt_para('v')['fwhm']*4/2.35)
    print('flux of uw1: '+str(flux_uw1)+' +/- '+str(flux_uw1_err))
    print('flux of v: '+str(flux_v)+' +/- '+str(flux_v_err))
    return flux_OH, flux_OH_err


def flux_theo(spec_name, filt):
    '''flux = integrate of flux density over wavelength
    (no effective area!)
    '''
    spec_wave, spec_flux = read_spec(spec_name)
    ea_wave, ea_area = read_ea(filt)
    ea_func = interpolate.interp1d(ea_wave, ea_area, fill_value='extrapolate')
    ea_area = ea_func(spec_wave)
    spec_wave = spec_wave[ea_area != 0]
    flux_theo = 0
    delta_wave = spec_wave[1]-spec_wave[0]
    for i in range(len(spec_wave)):
        flux_theo += spec_flux[i]*delta_wave
    return flux_theo

def flux_ref_uw1(spec_name_sun, spec_name_OH, 
                 cr_uw1, cr_uw1_err, cr_v, cr_v_err, r=0):
    '''get flux of uw1 reflection from OH cr
    '''
    cr_sun_v = mag_sb_flux_from_spec(spec_name_sun, 'v')[0]
    cr_sun_uw1 = mag_sb_flux_from_spec(spec_name_sun, 'uw1')[0]
    beta = cr_sun_uw1/cr_sun_v
    beta = reddening_correct(r)*beta
    cr_ref_uw1 = beta*cr_v
    cr_ref_uw1_err = beta*cr_v_err
    flux_sun_uw1 = flux_theo(spec_name_sun, 'uw1')
    flux_ref_uw1 = (flux_sun_uw1/cr_sun_uw1)*cr_ref_uw1
    flux_ref_uw1_err = (flux_sun_uw1/cr_sun_uw1)*cr_ref_uw1_err
    return flux_ref_uw1, flux_ref_uw1_err

def flux_ref_v(spec_name_sun, spec_name_OH, 
               cr_uw1, cr_uw1_err, cr_v, cr_v_err, r=0):
    '''get flux of v reflection from OH cr
    '''
    cr_sun_v = mag_sb_flux_from_spec(spec_name_sun, 'v')[0]
    flux_sun_v = flux_theo(spec_name_sun, 'v')
    flux_ref_v = (flux_sun_v/cr_sun_v)*cr_v
    flux_ref_v_err = (flux_sun_v/cr_sun_v)*cr_v_err
    return flux_ref_v, flux_ref_v_err

def flux_cr_OH(spec_name_sun, spec_name_OH, 
               cr_uw1, cr_uw1_err, cr_v, cr_v_err, r=0):
    '''get OH flux from OH cr
    '''
    cr_sun_v = mag_sb_flux_from_spec(spec_name_sun, 'v')[0]
    cr_sun_uw1 = mag_sb_flux_from_spec(spec_name_sun, 'uw1')[0]
    beta = cr_sun_uw1/cr_sun_v
    beta = reddening_correct(r)*beta
    cr_ref_uw1 = beta*cr_v
    #!print('beta: '+str(beta))
    cr_ref_uw1_err = beta*cr_v_err
    cr_OH = cr_uw1 - cr_ref_uw1
    cr_OH_err = error_prop('sub', 
                           cr_uw1, cr_uw1_err, 
                           cr_ref_uw1, cr_ref_uw1_err)
    flux_OH_model = flux_theo(spec_name_OH, 'uw1')
    cr_OH_model = mag_sb_flux_from_spec(spec_name_OH, 'uw1')[0]
    flux_OH = cr_OH/(cr_OH_model/flux_OH_model)
    #!print('cr to flux: '+str(1./(cr_OH_model/flux_OH_model)))
    flux_OH_err = cr_OH_err/(cr_OH_model/flux_OH_model)
    flux_uw1, flux_uw1_err = flux_ref_uw1(spec_name_sun, 
                                          spec_name_OH, 
                                          cr_uw1, cr_uw1_err, 
                                          cr_v, cr_v_err, r)
    flux_v, flux_v_err = flux_ref_v(spec_name_sun, 
                                    spec_name_OH, 
                                    cr_uw1, cr_uw1_err, 
                                    cr_v, cr_v_err, r)
    #!print('flux of uw1 (reflection): '+str(flux_uw1)+' +/- '+str(flux_uw1_err))
    #!print('flux of v: '+str(flux_v)+' +/- '+str(flux_v_err))
    return flux_OH, flux_OH_err

def flux_cr_spec(spec_name_sun, spec_name_OH, 
                 cr_uw1, cr_uw1_err, cr_v, cr_v_err,
                 r=0):
    cr_sun_v = mag_sb_flux_from_spec(spec_name_sun, 'v')[0]
    cr_sun_uw1 = mag_sb_flux_from_spec(spec_name_sun, 'uw1')[0]
    beta = cr_sun_uw1/cr_sun_v
    beta = reddening_correct(r)*beta
    cr_ref_uw1 = beta*cr_v
    cr_ref_uw1_err = beta*cr_v_err
    factor_sun_uw1 = cr_ref_uw1/cr_sun_uw1
    factor_sun_uw1_err = cr_ref_uw1_err/cr_sun_uw1
    factor_sun_v = cr_v/cr_sun_v
    factor_sun_v_err = cr_v_err/cr_sun_v
    cr_OH = cr_uw1 - cr_ref_uw1
    cr_OH_err = error_prop('sub',
                           cr_uw1, cr_uw1_err,
                           cr_ref_uw1, cr_ref_uw1_err)
    cr_OH_model = mag_sb_flux_from_spec(spec_name_OH, 'uw1')[0]
    factor_OH = cr_OH/cr_OH_model
    factor_OH_err = cr_OH_err/cr_OH_model
    return (factor_sun_uw1, factor_sun_uw1_err, 
            factor_sun_v, factor_sun_v_err,
            factor_OH, factor_OH_err)

def cr_OH(spec_name_sun,
          cr_uw1, cr_uw1_err,
          cr_v, cr_v_err,
          r):
    alpha = 2.
    sun_cr_uw1 = mag_sb_flux_from_spec(spec_name_sun, 'uw1')[0]
    sun_cr_v = mag_sb_flux_from_spec(spec_name_sun, 'v')[0]
    beta = sun_cr_uw1/sun_cr_v
    beta = reddening_correct(r)*beta
    beta_err = 0.
    cr = alpha*(cr_uw1 - beta*cr_v)
    b_c_err = error_prop('mul',
                         beta, beta_err,
                         cr_v, cr_v_err)
    cr_err = alpha*error_prop('sub',
                              cr_uw1, cr_uw1_err,
                              beta*cr_v, b_c_err)
    return cr, cr_err

def flux_cf_OH(spec_sun_name,
               cr_uw1, cr_uw1_err,
               cr_v, cr_v_err,
               r):
    alpha = 2.
    sun_flux_uw1 = mag_sb_flux_from_spec(spec_sun_name, 'uw1')[3]
    sun_flux_v = mag_sb_flux_from_spec(spec_sun_name, 'v')[3]
    beta = sun_flux_uw1[0]/sun_flux_v[0]
    beta = reddening_correct(r)*beta
    beta_err = error_prop('div',
                          sun_flux_uw1[0], sun_flux_uw1[1],
                          sun_flux_v[0], sun_flux_v[1])
    flux_uw1, flux_err_uw1 = cr2flux(cr_uw1, cr_uw1_err, 'uw1')
    flux_v, flux_err_v = cr2flux(cr_v, cr_v_err, 'v')
    flux = alpha*(flux_uw1 - beta*flux_v)
    b_f_err = error_prop('mul', 
                         beta, beta_err, 
                         flux_v, flux_err_v)
    flux_err = alpha*error_prop('sub', 
                                flux_uw1, flux_err_uw1,
                                beta*flux_v, b_f_err)
    print('flux of uw1: '+str(flux_uw1)+' +/- '+str(flux_err_uw1))
    print('flux of v: '+str(flux_v)+' +/- '+str(flux_err_v))
    return flux, flux_err

def get_OH(method, 
           spec_name_sun, spec_name_OH, spec_name_sum, 
           cr_uw1, cr_uw1_err,
           cr_v, cr_v_err,
           r = 0):
    '''
    method = 'cr', 'flux_cf', 'flux_my_cf', 'flux_cr'
    '''
    if method == 'cr':
        flux, flux_err = cr_OH(spec_name_sun, 
                               cr_uw1, cr_uw1_err,
                               cr_v, cr_v_err, r)
    elif method == 'flux_cf':
        flux, flux_err = flux_cf_OH(spec_name_sun,
                                    cr_uw1, cr_uw1_err,
                                    cr_v, cr_v_err, r)
    elif method == 'flux_my_cf':
        flux, flux_err = flux_my_cf_OH(spec_name_sun, spec_name_sum,
                                       cr_uw1, cr_uw1_err,
                                       cr_v, cr_v_err, r)
    elif method == 'flux_cr':
        flux, flux_err = flux_cr_OH(spec_name_sun, spec_name_OH, 
                                    cr_uw1, cr_uw1_err,
                                    cr_v, cr_v_err, r)
    else:
        RaiseExcept('Please check the method!')
    return flux, flux_err