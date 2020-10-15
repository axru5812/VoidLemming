""" spechelpers.py
A set of classes and functions useful for dealing with spectroscopic data
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sys
import astropy.io.fits as fits
from .extin import CCM

# =============================================================================
#                     Base class for containing a spectrum
# =============================================================================


class spectrum():
    def __init__(self, specfile, x=None, y=None, ivar=True):
        if x is not None and y is not None:
            self.wl = x
            self.flux = y
        else:
            df, err = self.readspec(specfile, ivar)
            self.wl = df['lobs']
            self.flux = df['fobs']
            self.name = specfile.split('/')[-1].split('.')[0]
            if err:
                self.err = df.err
            else:
                self.err = None

    def readspec(self, specfile, ivar):
        fileending = specfile.split('/')[-1].split('.')[-1]
        if fileending == 'fits':
            # Open a fits file to get data
            hdulist = fits.open(specfile)
            primaryheader = hdulist[0].header
            primarydata = hdulist[1].data

            # Assign the data and errors to df for consistency
            df = pd.DataFrame()
            df['lobs'] = 10 ** primarydata['loglam']
            df['fobs'] = primarydata['flux']
            # Get rid of divide by 0 problem
            ivar = primarydata['ivar']
            ivar[np.where(ivar == 0.)] = 1e-30
            df['err'] = np.sqrt(1 / ivar)
            err = True

        elif fileending == 'fit':
            # Open a fits file to get data
            hdulist = fits.open(specfile)
            primaryheader = hdulist[0].header
            primarydata = hdulist[0].data

            # Assign the data and errors to df for consistency
            df = pd.DataFrame()
            df['fobs'] = primarydata[0, :]
            df['lobs'] = 10**(primaryheader['COEFF0'] +
                              primaryheader['COEFF1'] *
                              np.arange(len(df.fobs)))

            # Get rid of divide by 0 problem
            err = primarydata[2, :]
            err[np.where(err == 0.)] = 1e-30
            df['err'] = err
            err = True

        else:
            # Open a text file
            try:
                # Try to read in 3 column to see if there is an error
                df = pd.read_csv(specfile, delimiter='\s+', comment='#',
                                 header=None, names=['lobs', 'fobs',
                                                     'inv_var'],
                                 usecols=[0, 1, 2])
                if ivar:
                    df['err'] = np.sqrt(1 / df.inv_var)
                else:
                    df['err'] = df.inv_var
                # If this worked set error flag to true
                err = True

            except Exception as e:
                print('Problem occured reading file: ', e)
                print('Attempting to read without error')
                # Other wise just read  wavelength and flux
                df = pd.read_csv(specfile, delimiter='\s+', comment='#',
                                 header=None, names=['lobs', 'fobs'])
                err = False
        return df, err

    def plot(self):
        fig, ax = plt.subplots(figsize=(15, 4))
        plt.plot(self.wl, self.flux)
        return fig, ax

    def save2ascii(self, path, filename=False):
        if not filename:
            filename = path + self.name + '.ascii'
        else:
            filename = path
        with open(filename, 'w') as fn:
            if self.err is not None:
                for i in range(len(self.flux)):
                    line = '{}    {}    {}\n'.format(self.wl.loc[i],
                                                     self.flux.loc[i],
                                                     self.err.loc[i])
                    fn.write(line)
            else:
                for i in range(len(self.flux)):
                    line = '{}    {}\n'.format(self.wl.loc[i],
                                               self.flux.loc[i])
                    fn.write(line)
        print('Saved spectrum to: ', filename)

    def save2fits(self, path, filename=False):
        if not filename:
            filename = path + self.name + '.fits'
        else:
            filename = path
        sys.exit('Not implemented')

    def deRedden(self, EBV):
        """Function that uses the specified extinction law to deredden a spectrum
        given an extinction law and a dust extinction.
        """
        # Calculate the k-values
        karray = CCM(self.wl)
        newf = self.flux / 10**(karray * EBV / -2.5)
        self.flux = newf

    def shift2Restframe(self, z):
        """Function that shifts the spectrum to restframe given a redshift
        """
        new_wl = self.wl / (1 + z)
        self.wl = new_wl
        self.restframe = True

# ==============================================================================
#                      Speed to wavelength and vv. conversions
# ==============================================================================


def wl2v(wave, ref_wl):
    """ Converts a wavelength range and a reference wavelength to a
    velocity range.
    Velocities are in km/s.
    """
    c = 299792458
    # λ / λ0 = 1 + v / c =>
    #      v = (1 + λ / λ0) * c
    v = (wave / ref_wl - 1.) * c / 1000.
    return v


def dwl2v(deltaWave, ref_wl):
    c = 299792458
    # dλ / λ0 = v / c =>
    #      v = (dλ / λ0) * c
    v = (deltaWave / ref_wl) * c / 1000.
    return v


def v2wl(v, ref_wl):
    """ Converts a velocity range and a reference wavelength to a
    wavelength range.
    Velocities are in km/s.
    """
    c = 299792458
    # λ / λ0 = 1 + v / c =>
    #      λ = λ0 * (1 + v / c)
    wave = ref_wl * (1. + v / (c / 1000.))
    return wave


def dv2wl(deltav, ref_wl):
    """ Converts a velocity range and a reference wavelength to a
    wavelength range.
    Velocities are in km/s.
    """
    c = 299792458
    # dλ / λ0 = dv / c =>
    #      dλ = λ0 * (dv / c)
    wave = ref_wl * (deltav / (c / 1000.))
    return wave


# =============================================================================
#                    Integer rebinning function for COS
# =============================================================================
def intRebin(arr, N=6, err=None, pandas=False):
    """
    Rebin 1D in a way that each pixel contributes once and only once.
    Ignores the last remainder of the array.
    """
    Nbin = int(len(arr) / N)
    Npe = Nbin * N
    if pandas:
        twod = arr[:Npe].values.reshape(Nbin, N)
    else:
        twod = arr[:Npe].reshape(Nbin, N)
    binned = twod.mean(axis=1)

    if err is None:
        return binned
    else:
        if pandas:
            twod = err[:Npe].values.reshape(Nbin, N)
        else:
            twod = err[:Npe].reshape(Nbin, N)
        ebinned = sp.sqrt((twod**2).mean(axis=1))
        return binned, ebinned


# =============================================================================
#                      Monte Carlo functions
# =============================================================================
def redrawMCdata(data, error, n=1):
    """ Redraws datavector given an errorvector. The errors are assumed to be
        1 sigma gaussian errors.
    """
    if n == 1:
        errors = np.random.normal(loc=0, scale=1, size=len(error)) * error
        newdata = data + errors
    else:
        newdata = np.ones(shape=(n, len(data)))
        for i in range(n):
            errors = np.random.normal(loc=0, scale=1, size=len(error)) * error
            newdata[i, :] = data + errors

    return newdata


def getMCerrors(*args):
    """ Calculates the 1 sigma errors for an arbitrary number of variables.
        Assumes symmetrical errors.
    """
    errs = []
    for i in range(len(args)):
        # Get the std of the results
        errs.append(args[i].std())
    return errs
