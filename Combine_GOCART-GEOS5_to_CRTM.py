# -------------------------------------------------------------------------------
# Processing GOCART aerosol LUTs follow CRTM aerosol coefficients LUT conventions
# Input: GOCART aerosol LUTs
#         optics_DU.v15_6.nc
#         optics_NI.v2_6.nc
#         optics_OC.v1_6.nc
#         optics_SS.v3_6.nc
#         optics_BC.v1_6.nc              
#         optics_SU.v1_6.nc
# Output: one netCDF file that contains all aerosol coefficient
#           Dust: optics_DU.v15_5.nc                                                                            ",
#       Sea Salt: optics_SS.v3_5.nc                                                                                             
# Organic Carbon: optics_OC.v1_5.nc 
#   Black carbon: optics_BC.v1_5.nc                                                                 
#        Sulfate: optics_SU.v2_5.nc
#        Nitrate: optics_NI.v2_6.nc
#   Brown Carbon: optics_BRC.v2_5.nc
# -----------------------------------------------------------------------------
# Reference:
# 1. GOCART Phase function: phase_function.pro [Peter Colarco, November 2018]
# 2. Delta fit: dfit_Hu.py [Patrick Stegmann]
# 3. Delta fit: Fortran module provided by Mark [Hu 2000]
# -----------------------------------------------------------------------------
# Cheng Dang (dangch@ucar.edu) 2021-01


# =========================
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.special import legendre
from scipy.interpolate import griddata
import bfit_python



def phs_fun(N_Scattering_Angles, n_Stream, Mom_Coef):
    Phase_Func = np.empty((N_Scattering_Angles))   
    x = np.flip(np.arange(-1,1,2./N_Scattering_Angles))
    angles = 180./np.pi*np.arccos(x)
    #nx = 2.*angles/angles[-1] - 1.
    Phase_Func = np.polynomial.legendre.legval(x, Mom_Coef[0:n_Stream])
#     for ii in range(N_Scattering_Angles):
#         # Reconstruction of the phase function:
#         Phase_Func[ii] = np.polynomial.legendre.legval(x[ii],Mom_Coef[0:n_Stream])
    return [angles, Phase_Func]

def read_ncvar(infile, varname):
    nc_fid = Dataset(infile,'r')
    out =  np.array(nc_fid.variables[varname][:])
    return out

def read_GOCART_dim(infile):
    ncfid =  Dataset(infile, 'r')
    # Dimensions
    n_rh     = ncfid.dimensions['rh'].size
    n_lambda = ncfid.dimensions['lambda'].size
    n_radius = ncfid.dimensions['radius'].size
    n_Mom   = ncfid.dimensions['nMom'].size
    n_Pol   = ncfid.dimensions['nPol'].size
    ncfid.close()
    #print(n_rh, n_lambda, n_radius, n_Mom, n_Pol)
    return[n_rh, n_lambda, n_radius, n_Mom, n_Pol]

def read_GOCART_var(infile):
    ncfid =  Dataset(infile, 'r')
    rh    = ncfid.variables['rh'][:]
    gf    = ncfid.variables['growth_factor'][:]
    wvl   = ncfid.variables['lambda'][:]
    rds   = ncfid.variables['radius'][:]
    bext  = ncfid.variables['bext'][:]
    bsca  = ncfid.variables['bsca'][:]
    g     = ncfid.variables['g'][:]
    pmom  = ncfid.variables['pmom'][:]
    rEff  = ncfid.variables['rEff'][:]
    ncfid.close()
    return[rh, gf, wvl, rds, bext, bsca, g, pmom, rEff]

def weight(theta, cutoff = 3.0):
    if theta < cutoff:
        return 0.
    else:
        return 1.

def delta_fit_Hu(theta, Phase_Func, n_Stream):
    nx = 2.*theta/theta[-1] - 1.
    error = 0.0
    tt = 0
    legvals = np.zeros((theta.size,n_Stream))
    A = np.zeros((theta.size,n_Stream))
    for jj in range(n_Stream):
        Pn = legendre(jj)
        for ii in range(theta.size):
            legvals[ii,jj] = Pn(nx[ii])

    for ii in range(theta.size):
        for jj in range(n_Stream):
            A[ii,jj] = legvals[ii,jj]/Phase_Func[ii]*weight(theta[ii])

    b = np.ones(theta.size)

    sol = np.linalg.lstsq(A,b,rcond=None)
    coef = sol[0]

    dfit = np.polynomial.legendre.legval(nx, coef)
#     plt.figure()
#     plt.semilogy(theta, Phase_Func, 'o')
#     plt.semilogy(theta, dfit, '--', label=str(n_Stream))

    for ii in range(10,theta.size):
        error += (dfit[ii]-P11[ii])**2
    tt += 1
    return[coef, dfit, error]

def delta_fit_error(P11, dfit):
    error = 0.0
    for ii in range(10,P11.size):
        error += (dfit[ii]-P11[ii])**2
    return[error]

def extra_dust_wvl(infile_target, infile_interp, output_interp_file):
    # Proecess Dust data
    # The number of wavelength in dust LUT is different from n_Lamda of the other LUTs
    # Do interpolations for dust LUT first
    # ... Get the target number of wavelength
    print('read from: ' + infile_target)
    [n_rh, n_lambda, n_radius, n_Mom, n_Pol] = read_GOCART_dim(infile_target)
    [rh, gf, wvl, rds, bext, bsca, g, pmom, rEff]    = read_GOCART_var(infile_target)
    n_lambda_tgt = n_lambda
    wvl_tgt      = wvl

    # ... Get dust optical properties
    print('read from: ' + infile_interp)
    [n_rh, n_lambda, n_radius, n_Mom, n_Pol] = read_GOCART_dim(infile_interp)
    [rh, gf, wvl, rds, bext, bsca, g, pmom, rEff]  = read_GOCART_var(infile_interp)


    # ... interpolate results based for target wavelength
    bext_dust = np.zeros([n_radius, n_rh, n_lambda_tgt])
    bsca_dust = np.zeros([n_radius, n_rh, n_lambda_tgt])
    g_dust    = np.zeros([n_radius, n_rh, n_lambda_tgt]) 
    pmom_dust = np.zeros([n_Pol, n_Mom, n_radius, n_rh, n_lambda_tgt])

    for ir in range(n_radius):
        for irh in range(n_rh):        
            for i in range(n_lambda_tgt):
                bext_dust[ir,irh,i] = np.interp(wvl_tgt[i], wvl, bext[ir,irh,:])
                bsca_dust[ir,irh,i] = np.interp(wvl_tgt[i], wvl, bsca[ir,irh,:])
                g_dust[ir,irh,i]    = np.interp(wvl_tgt[i], wvl,    g[ir,irh,:])
            for ip in range(n_Pol):
                for im in range(n_Mom):
                    for i in range(n_lambda_tgt):
                        pmom_dust[ip,im,ir,irh,i] = np.interp(wvl_tgt[i], wvl, pmom[ip,im,ir,irh,:])

    # # check if the interpolation is correct
    # ir = 1
    # irh = 10
    # plt.figure()
    # plt.semilogx(wvl,bsca[ir, irh,:], 'o')
    # plt.semilogx(wvl_tgt, bsca_dust[ir, irh,:], '+')

    # ip = 0
    # im = 2
    # plt.figure()
    # plt.semilogx(wvl,pmom[ip,im,ir,irh,:], 'o')
    # plt.semilogx(wvl_tgt, pmom_dust[ip,im,ir,irh,:], '+')


    # OUTPUT interpolated data to a nc4 file
    print('write to: ' + output_interp_file)
    w_nc_fid = Dataset(output_interp_file, 'w', format='NETCDF4')
    # write dimensions
    w_nc_fid.createDimension('rh'    , n_rh)
    w_nc_fid.createDimension('lambda', n_lambda_tgt)
    w_nc_fid.createDimension('radius', n_radius)
    w_nc_fid.createDimension('nMom'  , n_Mom)
    w_nc_fid.createDimension('nPol'  , n_Pol)
    # create varaibles
    w_nc_fid.createVariable('rh', 'f8', ('rh'))
    w_nc_fid.createVariable('lambda',    'f8', ('lambda'))
    w_nc_fid.createVariable('radius', 'f8', ('radius'))
    w_nc_fid.createVariable('bext', 'f8', ('radius', 'rh', 'lambda'))
    w_nc_fid.createVariable('bsca', 'f8', ('radius', 'rh', 'lambda'))
    w_nc_fid.createVariable('g',    'f8', ('radius', 'rh', 'lambda'))
    w_nc_fid.createVariable('pmom', 'f8', ('nPol', 'nMom', 'radius', 'rh', 'lambda'))
    w_nc_fid.createVariable('rEff', 'f8', ('radius', 'rh'))
    w_nc_fid.createVariable('growth_factor', 'f8', ('radius', 'rh'))
    # write variables
    w_nc_fid.variables['rh'][:]     = rh
    w_nc_fid.variables['lambda'][:] = wvl_tgt
    w_nc_fid.variables['radius'][:] = rds
    w_nc_fid.variables['bext'][:]   = bext_dust
    w_nc_fid.variables['bsca'][:]   = bsca_dust
    w_nc_fid.variables['g'][:]      = g_dust
    w_nc_fid.variables['pmom'][:]   = pmom_dust
    w_nc_fid.variables['rEff'][:]   = rEff
    w_nc_fid.variables['growth_factor'][:]   = rEff
    w_nc_fid.close()

    
def write_CRTM_GOCART(dimensions, variables, output_gocart, Release = 3, Version = 1):
    [n_Wavelengths, n_Radii, n_Sigma, 
     n_Types, n_RH, n_Legendre_Terms, 
     n_Phase_Elements, tnsl] = dimensions

    [Aerosol_Type, Aerosol_Type_Name, Wavelength, 
     Reff, Reff_RH, Rsig, RH, ke, w, g, pcoeff, growthf] = variables
    
    # Unit conversion
    Reff = Reff *1e6
    Reff_RH = Reff_RH * 1e6
    Wavelength = Wavelength * 1e6

    # OUTPUT interpolated data to a nc4 file
    print('write to: ' + output_gocart)
    
    w_nc_fid = Dataset(output_gocart, 'w', format='NETCDF4')
    # write dimensions
    w_nc_fid.createDimension('n_Wavelengths', n_Wavelengths)
    w_nc_fid.createDimension('n_Radii', n_Radii)
    w_nc_fid.createDimension('n_Sigma', n_Sigma)
    w_nc_fid.createDimension('n_Types', n_Types)
    w_nc_fid.createDimension('n_RH'   , n_RH)
    w_nc_fid.createDimension('n_Legendre_Terms', n_Legendre_Terms)
    w_nc_fid.createDimension('n_Phase_Elements', n_Phase_Elements)
    w_nc_fid.createDimension('tnsl'   , tnsl)
    
    # create varaibles
    Aerosol_Type_nc      = w_nc_fid.createVariable('Aerosol_Type',      'i4',  ('n_Types',))
    Aerosol_Type_Name_nc = w_nc_fid.createVariable('Aerosol_Type_Name', 'S1',  ('n_Types', 'tnsl'))
    Wavelength_nc        = w_nc_fid.createVariable('Wavelength',        'f8',  ('n_Wavelengths',))
    Reff_nc              = w_nc_fid.createVariable('Reff',              'f8',  ('n_Types', 'n_Radii'), fill_value=0.0)
    Rsig_nc              = w_nc_fid.createVariable('Rsig',              'f8',  ('n_Types', 'n_Sigma'))
    RH_nc = w_nc_fid.createVariable('RH',                'f8',  ('n_RH',))
    ke_nc = w_nc_fid.createVariable('ke',                'f8',  ('n_Types', 'n_Sigma', 'n_Radii', 'n_RH', 'n_Wavelengths'))
    w_nc  = w_nc_fid.createVariable('w',                 'f8',  ('n_Types', 'n_Sigma', 'n_Radii', 'n_RH', 'n_Wavelengths'))
    g_nc  = w_nc_fid.createVariable('g',                 'f8',  ('n_Types', 'n_Sigma', 'n_Radii', 'n_RH', 'n_Wavelengths'))
    pcoeff_nc        = w_nc_fid.createVariable('pcoeff',            'f8',  ('n_Phase_Elements', 'n_Legendre_Terms', 'n_Types', 'n_Sigma', 'n_Radii', 'n_RH', 'n_Wavelengths'))
    Reff_RH_nc       = w_nc_fid.createVariable('Reff_RH',           'f8',  ('n_Types', 'n_Radii', 'n_RH'), fill_value=0.0)
    growth_factor_nc = w_nc_fid.createVariable('growth_factor',     'f8',  ('n_Types', 'n_Radii', 'n_RH'), fill_value=0.0)

    # write attributes
    nc_fid = Dataset('AerosolCoeff.nc4', 'r')
    for ncattr in nc_fid.variables['Aerosol_Type'].ncattrs():
        Aerosol_Type_nc.setncattr(ncattr, nc_fid.variables['Aerosol_Type'].getncattr(ncattr))
    for ncattr in nc_fid.variables['Aerosol_Type_Name'].ncattrs():
        Aerosol_Type_Name_nc.setncattr(ncattr, nc_fid.variables['Aerosol_Type_Name'].getncattr(ncattr))
    for ncattr in nc_fid.variables['Wavelength'].ncattrs():
        Wavelength_nc.setncattr(ncattr, nc_fid.variables['Wavelength'].getncattr(ncattr))
    for ncattr in nc_fid.variables['Reff'].ncattrs():
        Reff_nc.setncattr(ncattr, nc_fid.variables['Reff'].getncattr(ncattr))
    for ncattr in nc_fid.variables['Rsig'].ncattrs():
        Rsig_nc.setncattr(ncattr, nc_fid.variables['Rsig'].getncattr(ncattr))
    for ncattr in nc_fid.variables['RH'].ncattrs():
        RH_nc.setncattr(ncattr, nc_fid.variables['RH'].getncattr(ncattr))
    for ncattr in nc_fid.variables['ke'].ncattrs():
        ke_nc.setncattr(ncattr, nc_fid.variables['ke'].getncattr(ncattr))
    for ncattr in nc_fid.variables['w'].ncattrs():
        w_nc.setncattr(ncattr, nc_fid.variables['w'].getncattr(ncattr))
    for ncattr in nc_fid.variables['g'].ncattrs():
        g_nc.setncattr(ncattr, nc_fid.variables['g'].getncattr(ncattr))
    for ncattr in nc_fid.variables['pcoeff'].ncattrs():
        pcoeff_nc.setncattr(ncattr, nc_fid.variables['pcoeff'].getncattr(ncattr))        
    w_nc_fid.variables['Reff'].long_name           = 'dry particle effective radius of sizebin' 
    w_nc_fid.variables['Reff'].units               = 'Microns (um)' 
    w_nc_fid.variables['Reff_RH'].long_name        = 'effective radius of size bin'
    w_nc_fid.variables['Reff_RH'].units            = 'Microns (um)' 
    w_nc_fid.variables['growth_factor'].long_name  = 'growth factor = ratio of wet to dry particle radius'
    w_nc_fid.variables['growth_factor'].units      = "fraction"
    nc_fid.close()
    
    # write variables
    w_nc_fid.variables['Aerosol_Type'][:]      = Aerosol_Type
    w_nc_fid.variables['Aerosol_Type_Name'][:] = Aerosol_Type_Name
    w_nc_fid.variables['Wavelength'][:]        = Wavelength
    w_nc_fid.variables['Reff'][:]              = Reff
    w_nc_fid.variables['Reff_RH'][:]           = Reff_RH
    w_nc_fid.variables['Rsig'][:]              = Rsig
    w_nc_fid.variables['RH'][:]  = RH
    w_nc_fid.variables['ke'][:]  = ke
    w_nc_fid.variables['w'][:]   = w
    w_nc_fid.variables['ke'][:]  = ke
    w_nc_fid.variables['g'][:]   = g
    w_nc_fid.variables['pcoeff'][:]          = pcoeff
    w_nc_fid.variables['growth_factor'][:]   = growthf

    # write global variables 
    w_nc_fid.Release = np.int32(Release)
    w_nc_fid.Version = np.int32(Version)
    w_nc_fid.Data_Source = "GOCART-GEOS5"
    w_nc_fid.Title = "Aerosol Optical Properties in the infrared and visible spectral region." 
    w_nc_fid.History = "Combine GOCART LUTs (Cheng DAND, 202011）"
    w_nc_fid.Comment = "Based on GOCART LUTs: optics_DU.v15_6.nc, optics_DU.v15_6.nc, optics_NI.v2_6.nc,                         optics_OC.v1_6.nc, optics_SS.v3_6.nc, optics_SU.v1_6.nc, with delta fit applied"

    w_nc_fid.close()
    
def write_phase_function(outfile, n_angles, phase_function, angles):
    w_nc_fid = Dataset(outfile, 'w', format='NETCDF4')
    w_nc_fid.createDimension('n_angles', 2000)
    w_nc_fid.createVariable('phase_function',      'f8',  ('n_angles',))
    w_nc_fid.createVariable('angles',      'f8',  ('n_angles',))
    w_nc_fid.variables['phase_function'][:] = P11
    w_nc_fid.variables['angles'][:] = theta
    w_nc_fid.close()

    

# --- main program ------

# Process dust LUT (This step is needed because the dust data has different wavelength bins)
# extra_dust_wvl('optics_SS.v3_6.nc', 'optics_DU.v15_6.nc', 'optics_DU.v15_6_extrapolated.nc') 

# Read LUTs
GOCART_LUTs = ['optics_DU.v15_6_extrapolated.nc', 'optics_SS.v3_6.nc', 'optics_OC.v1_6.nc',
               'optics_BC.v1_6.nc',  'optics_SU.v1_6.nc', 'optics_NI.v2_6.nc']

# Define other variables needed
# ... delta fit calcualtion
Nstreams  = [2,4,6,8,16]
offset = {2:0,
          4:0,
          6:5,
          8:12,
          16:21}
nangles = 2000
nstr_reconstruct = 256
check_pcoeff_figure = False

# ... output dimensions
n_Wavelengths    = 61
n_Radii          = 5 # maximum number of bins among different LUTs
n_Sigma          = 1
n_Types          = len(GOCART_LUTs)
n_RH             = 36
n_Legendre_Terms = 38
n_Phase_Elements = 1 
tnsl             = 80 

# ... output variables
Aerosol_Type      = [1, 2, 3, 4, 5, 6]
Aerosol_Type_Name = np.chararray((n_Types, tnsl))
Aerosol_Names = ['Dust', 'Sea salt', 'Organic carbon', 'Black carbon', 'Sulfate', 'Nitrate']
for i, Aer in enumerate(Aerosol_Names):
    Aerosol_Type_Name[i, 0:len(list(Aer))] = list(Aer)
Wavelength        = np.zeros(n_Wavelengths)
Reff    = np.zeros((n_Types, n_Radii))
Reff_RH = np.zeros((n_Types, n_Radii, n_RH))
Rsig = np.zeros((n_Types, n_Sigma))
RH   = np.zeros((n_RH))
ke   = np.zeros((n_Types, n_Sigma, n_Radii, n_RH, n_Wavelengths))
w    = np.zeros((n_Types, n_Sigma, n_Radii, n_RH, n_Wavelengths))
g    = np.zeros((n_Types, n_Sigma, n_Radii, n_RH, n_Wavelengths))
pcoeff  = np.zeros((n_Phase_Elements, n_Legendre_Terms, n_Types, n_Sigma, n_Radii, n_RH, n_Wavelengths))
growthf = np.zeros((n_Types, n_Radii, n_RH)) # new variable, growth factor = ratio of wet to dry particle radius

# ... global attributes
Release = 3  # Release 3 is needed for CRTM simulation, this will be modified soon.
Version = 1


# Process individual files for phase function coeffciients and SSPs
for i, infile in enumerate(GOCART_LUTs):
    print('read from: ' + infile)
    [n_rh, n_lambda, n_radius, n_Mom, n_Pol] = read_GOCART_dim(infile)
    [rh, gf, wvl, rds, bext, bsca, g_in, pmom, rEff]  = read_GOCART_var(infile)
    
    # ... output data 
    Wavelength = wvl
    Reff[i,0:n_radius] = rds
    Reff_RH[i,0:n_radius, :] = rEff
    RH = rh
    #Rsig  #Rsig is 0, no values assigned
    ke[i, 0, 0:n_radius, :, :] = bext
    w[i, 0, 0:n_radius, :, :]  = bsca/bext
    g[i, 0, 0:n_radius, :, :]  = g_in
    growthf[i, 0:n_radius, :]  = gf
    
    # ... delta fit and compute pcoeff
    for ir in range(n_radius):
        for irh in range(n_rh):
            for iwvl in range(n_lambda):
                pmom_tmp = np.squeeze(pmom[0, :, ir, irh, iwvl])              
                [theta,P11] = phs_fun(nangles, nstr_reconstruct, pmom_tmp)                
                
                # write phase function for testing purposes
#                 write_phase_function('test_phase_function.nc', nangles, P11, theta)
 
                # Figure, check if the phase function make sense
                if (check_pcoeff_figure):
                    fig = plt.figure(figsize = (20, 5))
                    
                for nstr in Nstreams[:]:    
#                    # delta fit approach 1: Python function by Patrick
#                     [coef, dfit, error] = delta_fit_Hu(theta, P11, nstr)
                    
                    #delta fit approach 2: Fortran program by Mark Liu
                    #Fortran subroutine: bfit(nstr,nn,ang,phs,pfit,ftrunc,Polar,Error_Status)
                    coef2 = bfit_python.bfit(np.int32(nstr),np.int32(nangles),[theta],[P11],np.int32(0))
                    coef = coef2[0][0:nstr+1] / 2 # divide by 2 follow CRTM convention
                    coef[nstr] = coef2[1] # the last term in CRTM LUT is truncation factor
#                     nx = 2.*theta/theta[-1] - 1.
#                     dfit = np.polynomial.legendre.legval(nx, coef)
#                     [theta2, dfit] = phs_fun(nangles, nstr, coef)
#                     error = delta_fit_error(P11, dfit)
                    
                    ilow  = offset[nstr]
                    ihigh = ilow + nstr + 1
                    pcoeff[0, ilow:ihigh, i, 0, ir, irh, iwvl] = coef
                    
                    # Figure
                    if (check_pcoeff_figure):
                        ax1 = fig.add_subplot(1,3,1)
                        if nstr == 2:
                            coef2 = coef
                            ax1.semilogy(theta, P11,  '--k', label='Mie')
                            
#                         if nstr == 4:
#                             nx = 2.*theta/theta[-1] - 1.
#                             dfit_4FOR2 = np.polynomial.legendre.legval(nx, coef[0:2])
#                             ax1.semilogy(theta, dfit_4FOR2, '.b' , label='nstr = 2, coef4 ')
                            
#                             tmp = list(coef2)+list(coef[2:])
#                             print(tmp)
#                             dfit_2FOR4 = np.polynomial.legendre.legval(nx, tmp)
#                             ax1.semilogy(theta, dfit_2FOR4, '.y' , label='nstr = 4, coef2 ')
                            
                        ax1.semilogy(theta, dfit, '--' , label='dFit, nstr = ' + str(nstr))
                        plt.legend(loc=1, prop={'size': 12},ncol=1,fontsize=13)
                        plt.xlabel(r'Scattering Angle $\theta$ [deg]',fontsize=13)
                        plt.ylabel(r'Phase Function $P_{11}$ [-]',fontsize=13)

                        ax2 = fig.add_subplot(1,3,2)
                        ax2.plot(nstr, error, 'o', label='dFit, nstr = ' + str(nstr))
                        plt.legend(loc=1, prop={'size': 12},ncol=1,fontsize=13)
                        plt.ylabel(r'Residual $\varepsilon$',fontsize=13)
                        plt.xlabel(r'Number of Streams $N_{streams}$',fontsize=13)
                        
                        ax3 = fig.add_subplot(1,3,3)
                        ax3.plot(range(ilow,ihigh), abs(coef), 'o', label='dFit, nstr = ' + str(nstr))
                        if nstr == 16:
                            ax3.plot(range(n_Legendre_Terms), abs(pcoeff[0, :, i, 0, ir, iwvl, irh]), '--k')
                        #plt.legend(loc=2, prop={'size': 12},ncol=1,fontsize=13)
                        plt.ylabel(r'dFit coeff. (CRTM pcoeff)',fontsize=13)
                        plt.xlabel(r'Number of Leg. terms with CRTM offset',fontsize=13) 
                        st = plt.suptitle(Aerosol_Names[i] + 
                                          ' ($\mathrm{\lambda}$ = ' + str(round(wvl[iwvl]*1e6,2)) +
                                          '$\mathrm{\mu m}$' + 
                                          ', r = ' + str(round(rds[ir]*1e6,2)) + '$\mathrm{\mu m}$' + 
                                          ', RH = ' + str(rh[irh]) + ')' ,  fontsize=22)
                        st.set_position([.5, 0.98])
                        #plt.savefig('Phase_function.png', bbox_inches="tight")
                    
# Write combined lookup table to a netCDF file
dimensions = [n_Wavelengths, n_Radii, n_Sigma, n_Types, n_RH, n_Legendre_Terms, n_Phase_Elements, tnsl]
variables = [Aerosol_Type, Aerosol_Type_Name, Wavelength, Reff, Reff_RH, Rsig, RH, ke, w, g, pcoeff, growthf]
output_gocart = 'AerosolCoeff.GOCART-GEOS5.nc4'
write_CRTM_GOCART(dimensions, variables, output_gocart)






