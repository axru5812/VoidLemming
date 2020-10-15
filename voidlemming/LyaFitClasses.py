""" Main class for parametric Lyman alpha fitting
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import lmfit as lm
from collections import OrderedDict as oDict
import datetime
from .lineprofiles import *
from .plotfunctions import fill_between_steps

from . import spechelpers as sh


class baseFitter(sh.spectrum):

    def __init__(self, specfile, rebin=True):
        super().__init__(specfile, ivar=False)
        # Rebin by a factor 6 since it is COS spectra
        if rebin:
            self.wl = sh.intRebin(self.wl, N=6, pandas=True)
            self.flux, self.err = sh.intRebin(self.flux, err=self.err,
                                              N=6, pandas=True)

    # ======= Isolate the fitting window ======================================
    def selectWindow(self, width=2000):
        """ Method that selects a wavelength window around LymanAlpha (km/s)"""
        # Make sure we are in restframe
        if not self.restframe:
            raise ValueError('Spectrum must be in restframe. \
                             Please run shift2Restframe before selectWindow')

        # Set up a spectrum region that is 1000 km/s wide around 1216Å
        ref_wl = 1215.67
        vrange = np.array([-width / 2, width / 2])
        wlrange = sh.v2wl(vrange, ref_wl)
        # print(wlrange)

        # Select this region from spectrum
        selector = np.where((self.wl >= wlrange[0]) & (self.wl <= wlrange[1]))
        self.lya_wl = self.wl[selector]
        self.lya_flux = self.flux[selector]
        self.lya_err = self.err[selector]

    # ======= Simple plot of fitting window ===================================
    def plotLya(self):
        """ Plot the LymanAlpha line"""
        plt.figure(figsize=(15, 4))
        plt.plot(self.lya_wl, self.lya_flux, label='flux')
        plt.plot(self.lya_wl, self.lya_err, label='error')
        plt.legend()


# ==============================================================================
#                        Define main Lya fitter class
# ==============================================================================
'''
class LyaFitter(sh.spectrum):

    def __init__(self, specfile=None, inherit_spectrum=False, x=None, y=None,
                 err=None, name=None):
        if inherit_spectrum is False:
            super().__init__(specfile, ivar=False)
            # Rebin by a factor 6 since it is COS spectra
            self.wl = sh.intRebin(
                , N=6, pandas=True)
            self.flux, self.err = sh.intRebin(self.flux, err=self.err,
                                              N=6, pandas=True)
            self.restframe = False
        else:
            self.restframe = True

            self.wl = x
            self.flux = y
            self.err = err

            self.name = name

    # ======= Isolate the fitting window ======================================
    def selectWindow(self, width=2000, cont_comp=True):
        """ Method that selects a wavelength window around LymanAlpha (km/s)
            Also compensate up to a 'unitary' space, i.e. divide out the
            average.
        """
        # Make sure we are in restframe
        if not self.restframe:
            raise ValueError('Spectrum must be in restframe. \
                             Please run shift2Restframe before selectWindow')

        # Set up a spectrum region that is 1000 km/s wide around 1216Å
        ref_wl = 1215.67
        vrange = np.array([-width / 2, width / 2])
        wlrange = sh.v2wl(vrange, ref_wl)
        # print(wlrange)

        # Select this region from spectrum
        selector = np.where((self.wl >= wlrange[0]) & (self.wl <= wlrange[1]))

        self.lya_wl = self.wl[selector]
        self.lya_flux = self.flux[selector]
        self.lya_err = self.err[selector]

        if cont_comp:
            self.lya_avg = 1e-15  # np.mean(self.flux[selector])
            self.lya_flux = self.lya_flux / self.lya_avg
            self.lya_err = self.lya_err / self.lya_avg

    # ======= Simple plot of fitting window ===================================
    def plotLya(self, vscale=True):
        """ Plot the LymanAlpha line"""
        plt.figure(figsize=(13.5, 5))
        plt.step(self.lya_wl, self.lya_flux, label='Flux')
        fill_between_steps(plt.gca(), self.lya_wl,
                           self.lya_flux - self.lya_err,
                           self.lya_flux + self.lya_err, alpha=0.3,
                           label='Error')
        plt.xlabel('Wavelength [Å]')
        plt.ylabel(r'Flux [$10^{15}$erg s$^{-1}$ Å$^{-1}$ ]')

        plt.legend()

        ax = plt.gca()

        v_scale = sh.wl2v(self.lya_wl, 1215.67)
        ax2 = plt.twiny(ax)
        ax2.plot(v_scale, self.lya_flux, visible=False)
        ax2.set_xlabel('Velocity from linecenter [km\s]')

        plt.tight_layout()

    # ======= Initialize all the available models =============================
    def setupModels(self):
        """ Instantiates the LMfit models"""
        # Create an initial guess for amplitude
        ampInit = np.max(self.lya_flux)

        # === CMG profile: CMG ===
        CMG_model = lm.Model(CMG_Profile)
        CMG_params = CMG_model.make_params(amp=ampInit, center=1216,
                                           sigmaLeft=1, sigmaRight=1.5,
                                           cont=np.mean(self.lya_flux))

        # === Absorption Profile: ABS ===
        Abs_model = lm.Model(absorption_Profile)
        Abs_params = Abs_model.make_params(amp=np.mean(self.lya_flux) / 2,
                                           center=1216, sigma=1,
                                           cont=np.mean(self.lya_flux))
        # Set parameter bounds
        Abs_params['amp'].set(min=0)
        Abs_params['sigma'].set(min=0.2)

        # === P-cygni profile: PCG ===
        PCG_model = lm.Model(PCG_Profile)
        PCG_params = PCG_model.make_params(CMGamp=ampInit, CMGcenter=1216,
                                           CMGsigmaLeft=0.2, CMGsigmaRight=0.5,
                                           ABSamp=5,
                                           ABScenter=1215, ABSsigma=0.6,
                                           cont=0)
        # Set parameter bounds:
        CMGampInit = np.max(self.lya_flux[np.where(self.lya_wl > 1215.67)])
        PCG_params['CMGamp'].set(min=0, value=CMGampInit, max=10 * CMGampInit)
        PCG_params['CMGcenter'].set(min=1215, max=1220)
        PCG_params['CMGsigmaLeft'].set(min=0, max=10)
        PCG_params['CMGsigmaRight'].set(min=0, max=10)
        # constrain absorption
        PCG_params['ABSamp'].set(min=0, max=10)
        # PCG_params['ABScenter'].set(min=1210, max=1216)
        PCG_params['ABSsigma'].set(min=0, max=10)
        # constrain to positive cont level
        PCG_params['cont'].set(min=0, max=np.mean(self.lya_flux))

        PCG_params.add('AbsSep', value=1, min=0.1, max=2)
        PCG_params['ABScenter'].set(expr='(CMGcenter - AbsSep)')

        # === Double Horned profile: DBH ===
        DBH_model = lm.Model(DBH_Profile)
        LampInit = np.max(self.lya_flux[np.where(self.lya_wl < 1215.67)])
        LcenterInit = self.lya_wl[np.where(self.lya_flux == LampInit)]
        RampInit = np.max(self.lya_flux[np.where(self.lya_wl > 1215.67)])
        RcenterInit = self.lya_wl[np.where(self.lya_flux == RampInit)]
        contInit = np.mean(self.lya_flux[np.where(self.lya_wl > 1220)])

        DBH_params = DBH_model.make_params(LCMGamp=LampInit,
                                           LCMGcenter=LcenterInit,
                                           LCMGsigmaLeft=0.7,
                                           LCMGsigmaRight=0.3,
                                           RCMGamp=RampInit,
                                           RCMGcenter=RcenterInit,
                                           RCMGsigmaLeft=0.3,
                                           RCMGsigmaRight=0.7,
                                           ABSamp=0,
                                           ABSsigma=0.1,
                                           cont=contInit)
        # Set parameter bounds:
        DBH_params['LCMGamp'].set(min=0)  # constrain to emission
        DBH_params['RCMGamp'].set(min=0)  # constrain to emission
        sepGuess = self.lya_wl[np.where(self.lya_flux == RampInit)] -\
            self.lya_wl[np.where(self.lya_flux == LampInit)]
        DBH_params.add('PeakSep', value=sepGuess, min=0.8, max=3)
        DBH_params['RCMGcenter'].set(expr='(LCMGcenter + PeakSep)')
        # Constrain absorption:
        DBH_params.add('ABSsep', value=sepGuess / 2, min=0.2, max=1)
        DBH_params['ABScenter'].set(expr='LCMGcenter + ABSsep')

        # Constrain continuum:
        DBH_params['cont'].set(min=0)     # constrain to positive cont level

        # Insert the models into ordered dictionaries
        self.models = oDict(zip(['CMG', 'ABS', 'PCG', 'DBH'],
                                [CMG_model, Abs_model, PCG_model, DBH_model]))
        self.params = oDict(zip(['CMG', 'ABS', 'PCG', 'DBH'],
                                [CMG_params, Abs_params, PCG_params,
                                 DBH_params]))

    # ======= Run a fit to find the best descibing profile ====================
    def determineProfile(self, method='leastsq'):
        """ Runs a set of fits + uses the Chi² to determine best profile"""
        # Method defaults to brute to explore parameter space

        # Add brute_step to all param objects
        if method == 'brute':
            for name, params in self.params.items():
                for key, param in params.items():
                    param.set(brute_step=param.value * 10)

        # Initialize chisqr
        bestChi = 1e9

        # Loop over all models
        for key, model in self.models.items():
            print('Working on {}'.format(model))
            res = model.fit(self.lya_flux, x=self.lya_wl,
                            params=self.params[key],
                            fit_kws={'nan_policy': 'omit'},
                            method=method)
            if res.redchi < bestChi:
                self.bestModel = model
                self.bestName = key
                self.bestParams = self.params[key]
                bestChi = res.redchi

            self.params[key] = res.params
        print('The best model found is:', self.bestName)

    # ======= Fit the best profile again and do MC errors =====================
    def fitProfile(self, prof='best', method='leastsq', nmc=1000):
        """ Runs proper fit of the determined prof + error estimation"""
        # ==== RUN FITTING ===
        if prof == 'best':
            self.bestFit = self.bestModel.fit(self.lya_flux, x=self.lya_wl,
                                              params=self.bestParams,
                                              fit_kws={'nan_policy': 'omit'})
        else:
            try:
                self.bestFit = self.models[prof].fit(self.lya_flux,
                                                     x=self.lya_wl,
                                                     params=self.params[prof],
                                                     fit_kws={'nan_policy':
                                                              'omit'},
                                                     method=method)
                self.bestParams = self.params[prof]
                self.bestModel = self.models[prof]
                self.bestName = prof
            except KeyError:
                raise KeyError('There is no such model defined!')

        # Integrate the lyman alpha line:
        self.parFlux = self.parametricFlux(self.bestModel,
                                           self.bestFit.params)

        # ==== DO ERROR ESTIMATION ====
        mcData = sh.redrawMCdata(self.lya_flux, self.lya_err, n=nmc)
        integrals = []
        rcenters = []
        lsigmas = []
        rsigmas = []
        if self.bestName == 'DBH' or prof == 'DBH':
            peakseps = []
        for d in mcData:
            if prof == 'best':
                bestFit = self.bestModel.fit(d, x=self.lya_wl,
                                             params=self.bestParams,
                                             fit_kws={'nan_policy': 'omit'})
                integrals.append(self.parametricFlux(self.bestModel,
                                                     self.bestFit.params))
                if self.bestName == 'DBH':
                    peakseps.append(bestFit.params['PeakSep'])
                    lsigmas.append(bestFit.params['RCMGsigmaLeft'])
                    rsigmas.append(bestFit.params['RCMGsigmaRight'])
                    rcenters.append(bestFit.params['RCMGcenter'])

                elif self.bestName == 'CMG':
                    lsigmas.append(bestFit.params['sigmaLeft'])
                    rsigmas.append(bestFit.params['sigmaRight'])
                    rcenters.append(bestFit.params['center'])

                elif self.bestName == 'PCG':
                    lsigmas.append(bestFit.params['CMGsigmaLeft'])
                    rsigmas.append(bestFit.params['CMGsigmaRight'])
                    rcenters.append(bestFit.params['CMGcenter'])

            else:
                try:
                    bestFit = self.models[prof].fit(d, x=self.lya_wl,
                                                    params=self.params[prof],
                                                    fit_kws={'nan_policy':
                                                             'omit'},
                                                    method=method)
                    integrals.append(self.parametricFlux(self.models[prof],
                                                         self.bestFit.params))

                    if prof == 'DBH':
                        peakseps.append(bestFit.params['PeakSep'])
                        lsigmas.append(bestFit.params['RCMGsigmaLeft'])
                        rsigmas.append(bestFit.params['RCMGsigmaRight'])
                        rcenters.append(bestFit.params['RCMGcenter'])

                    elif self.bestName == 'CMG':
                        lsigmas.append(bestFit.params['sigmaLeft'])
                        rsigmas.append(bestFit.params['sigmaRight'])
                        rcenters.append(bestFit.params['center'])

                    elif self.bestName == 'PCG':
                        lsigmas.append(bestFit.params['CMGsigmaLeft'])
                        rsigmas.append(bestFit.params['CMGsigmaRight'])
                        rcenters.append(bestFit.params['CMGcenter'])

                except KeyError:
                    raise KeyError('There is no such model defined!')

        self.parFlux_err = sh.getMCerrors(np.array(integrals))[0]

        self.lsigma_err = sh.getMCerrors(np.array(lsigmas))[0]
        # print(self.lsigma_err)
        self.rsigma_err = sh.getMCerrors(np.array(rsigmas))[0]
        # print(self.rsigma_err)
        self.vRedPeak_err = sh.getMCerrors(sh.wl2v(np.array(rcenters),
                                                   1215.67))[0]
        # CMGskewnesses = np.array(rsigmas) / np.array(lsigmas)
        # self.CMGSkewness_err = sh.getMCerrors()

        if self.bestName == 'DBH':
            self.peakSep_err = sh.getMCerrors(sh.wl2v(
                                              np.array(peakseps),
                                              1215.67))[0]
        else:
            self.peakSep_err = np.nan

    # ======= Calculate the science-quantities ================================
    def calculateQuantities(self, plot=False, verbose=False):
        # Weighted Skewness
        self.Sw = self.weightedSkewness(self.lya_wl, self.lya_flux)
        # Errors:
        mcData = sh.redrawMCdata(self.lya_flux, self.lya_err, n=1000)
        swStore = []
        for d in mcData:
            swStore.append(self.weightedSkewness(self.lya_wl, d))
        self.Sw_err = sh.getMCerrors(np.array(swStore))[0]

        # PeakSeparation in km/s:
        try:
            self.peakSep = sh.dwl2v(self.bestFit.params['PeakSep'],
                                    1215.67)
        except KeyError:
            self.peakSep = np.nan

        # vRedPeak in km/s:
        try:
            self.vRedPeak = sh.wl2v(self.bestFit.params['RCMGcenter'], 1215.67)
        except KeyError:
            try:
                self.vRedPeak = sh.wl2v(self.bestFit.params['CMGcenter'],
                                        1215.67)
            except KeyError:
                try:
                    self.vRedPeak = sh.wl2v(self.bestFit.params['center'],
                                            1215.67)
                except KeyError:
                    self.vRedPeak = np.nan

        # Measure of Skewness from CMG:
        # The simplest measure of this is the ratio of the standard deviations
        try:
            self.CMGSkewness = (self.bestFit.params['CMGsigmaRight'] /
                                self.bestFit.params['CMGsigmaLeft'])
            # Error propagation
            term1 = ((1 / self.bestFit.params['CMGsigmaLeft']) *
                     self.rsigma_err)
            term2 = (self.bestFit.params['CMGsigmaRight'] /
                     self.bestFit.params['CMGsigmaLeft']**2) * self.lsigma_err
            self.CMGSkewness_err = np.sqrt(term1**2 + term2**2)
        except KeyError:
            try:
                self.CMGSkewness = (self.bestFit.params['RCMGsigmaRight'] /
                                    self.bestFit.params['RCMGsigmaLeft'])
                # Error propagation
                term1 = ((1 / self.bestFit.params['RCMGsigmaLeft']) *
                         self.rsigma_err)
                term2 = ((self.bestFit.params['RCMGsigmaRight'] /
                         self.bestFit.params['RCMGsigmaLeft']**2) *
                         self.lsigma_err)
                self.CMGSkewness_err = np.sqrt(term1**2 + term2**2)
            except KeyError:
                try:
                    self.CMGSkewness = (self.bestFit.params['sigmaRight'] /
                                        self.bestFit.params['sigmaLeft'])
                    # Error propagation
                    term1 = ((1 / self.bestFit.params['sigmaLeft']) *
                             self.rsigma_err)
                    term2 = ((self.bestFit.params['sigmaRight'] /
                             self.bestFit.params['sigmaLeft']**2) *
                             self.lsigma_err)
                    self.CMGSkewness_err = np.sqrt(term1**2 + term2**2)
                except KeyError:
                    self.CMGSkewness = np.nan
                    self.CMGSkewness_err = np.nan

        # Plot the fitting results:
        self.plotBestFit()

        # Simple gaussian v_peak after smoothing:
        smoothFit = self.fitSimpleGaussian(self.lya_wl, self.lya_flux,
                                           plot=plot, verbose=verbose)
        self.smoothVpeak = sh.wl2v(smoothFit.params['center'], 1215.67)

        # Error estimation
        smoothPeaks = []
        for d in mcData:
            c = self.fitSimpleGaussian(self.lya_wl, d).params['center']
            smoothPeaks.append(sh.wl2v(c, 1215.67))
        self.smoothVpeak_err = sh.getMCerrors(np.array(smoothPeaks))[0]
        # Print the results
        self.printResults()

    def get10percFluxPoints(self, flux, peakFlux):
        peakIndex = np.where(flux == peakFlux)[0][0]

        # iterate to the left from peak
        lFlux = peakFlux
        lIndex = peakIndex
        while lFlux > peakFlux / 10:
            lIndex -= 1
            lFlux = flux[lIndex]
        # Iterate to the right
        rFlux = peakFlux
        rIndex = peakIndex
        while rFlux > peakFlux / 10:
            rIndex += 1
            try:
                rFlux = flux[rIndex]
            except IndexError:
                print('Could not locate right 10% point for Weighted Skewness')
                rIndex -= 1
                rFlux = flux[rIndex]
                break

        # Get corresponding wavelength values
        wl_10b = self.lya_wl[lIndex]
        wl_10r = self.lya_wl[rIndex]

        return wl_10b, wl_10r, [lIndex, rIndex]

    # ======= Calculate Weighted Skewness =====================================
    def weightedSkewness(self, wavelength, flux):
        # 1: calculate cross-points
        peakFlux = np.max(flux[np.where(wavelength > 1215.67)])

        wl_10b, wl_10r, crossIndex = self.get10percFluxPoints(flux, peakFlux)
        Sw_wl = wavelength[crossIndex[0]: crossIndex[1]]
        Sw_flux = flux[crossIndex[0]: crossIndex[1]]

        # Kurk, Shimasaku, Kashikawa statistic:
        # Total I:
        In = Sw_flux.sum()
        # Pixel positions:
        x = np.arange(len(Sw_wl)) + 1.
        # Average pixel position:
        xbar = np.sum(x * Sw_flux) / In
        # Dispersion of x:
        sigma = np.sqrt(np.sum((x - xbar)**2 * Sw_flux) / In)
        # Unweighted Skewness:
        S = np.sum((x - xbar)**3 * Sw_flux) / (In * sigma**3)
        # Weighted Skewness:
        return S * (wl_10r - wl_10b)

    # ======= Caclulate parametric integrated flux ============================
    def parametricFlux(self, model, params):
        f = model.eval(x=self.lya_wl, params=params) - params['cont']
        integral = np.trapz(f * self.lya_avg, x=self.lya_wl)
        return integral

    # ======= Smooth and fit a single gaussian ================================
    def fitSimpleGaussian(self, wavelength, flux, verbose=False, plot=False):
        # Smooth the spectrum
        smooth_flux = filters.gaussian_filter1d(flux, 2)

        # Select the red peak
        cond = np.where(wavelength >= 1215.67)
        fit_wl = wavelength[cond]
        fit_flux = smooth_flux[cond]

        gmodel = lm.Model(gaussCont)
        gparams = gmodel.make_params(amplitude=1, center=1216, sigma=0.5,
                                     cont=np.mean(fit_flux))
        gparams['amplitude'].set(min=0)
        gfit = gmodel.fit(fit_flux, x=fit_wl, params=gparams,
                          fit_kws={'nan_policy': 'omit'})
        if verbose:
            lm.report_fit(gfit)

        # Plot the results
        if plot:
            fig = plt.figure(figsize=(15, 4))
            ax = plt.gca()
            plt.plot(fit_wl, fit_flux, label='Smoothed data')
            ax.set_xlabel(r'Wavelength [$\AA$]')
            ax.set_ylabel((r'Flux$\times${}'
                           ' [erg/cm$^2$/$\AA$]').format(self.lya_avg))

            res = gmodel.eval(x=wavelength, params=gfit.params)
            plt.plot(wavelength, res, label='Best fit gaussian')
            plt.legend()

            ax2 = plt.twiny(ax)
            v_scale = sh.wl2v(self.lya_wl, 1215.67)
            ax2.plot(v_scale, res, visible=False)
            ax2.set_xlabel('Velocity from linecenter [km\s]')
            plt.tight_layout()

        return gfit

    # ======= Plot the best fit ===============================================
    def plotBestFit(self):
        # Get velocity scale
        v_scale = sh.wl2v(self.lya_wl, 1215.67)

        # Evaluate best fit model
        res = self.bestModel.eval(x=self.lya_wl, params=self.bestFit.params)
        self.fit_result = res

        plt.figure(figsize=(13.5, 5))
        plt.step(self.lya_wl, self.lya_flux, label='Flux')
        fill_between_steps(plt.gca(), self.lya_wl,
                           self.lya_flux - self.lya_err,
                           self.lya_flux + self.lya_err, alpha=0.3,
                           label='Error')
        plt.xlabel('Wavelength [Å]')
        plt.ylabel(r'Flux [$10^{15}$erg s$^{-1}$ Å$^{-1}$ ]')

        ax = plt.gca()

        ax.plot(self.lya_wl, res, label='Best fit')

        try:
            plt.axvline(self.bestFit.params['RCMGcenter'],
                        label='Red peak center', color='grey', alpha=0.8)
        except KeyError:
            try:
                plt.axvline(self.bestFit.params['CMGcenter'],
                            label='Red peak center', color='grey', alpha=0.8)
            except KeyError:
                pass
        plt.legend()

        ax2 = plt.twiny(ax)
        ax2.plot(v_scale, res, visible=False)
        ax2.set_xlabel('Velocity from linecenter [km\s]')

        plt.tight_layout()

    # ======= Print all measured quantities ===================================
    def printResults(self):
        """ Print a nice summary of the obtained quantities"""
        print('\033[1m' + 'Derived quantities:' + '\033[0m')
        print('Best Fit Profile:'.ljust(30), self.bestName.ljust(20))
        print('Integrated flux:'.ljust(30), str(self.parFlux).ljust(20),
              '±', str(self.parFlux_err).ljust(20))
        print('Weighted Skewness:'.ljust(30), str(self.Sw).ljust(20), '±',
              str(self.Sw_err).ljust(20))
        print('CMG Skewness:'.ljust(30), str(self.CMGSkewness).ljust(20), '±',
              str(self.CMGSkewness_err).ljust(20))
        print('Peak Separation:'.ljust(30), str(self.peakSep).ljust(20), '±',
              str(self.peakSep_err).ljust(20))
        print('Red Peak velocity:'.ljust(30), str(self.vRedPeak).ljust(20),
              '±', str(self.vRedPeak_err))
        print('Smoothed Red Peak velocity:'.ljust(30),
              str(self.smoothVpeak).ljust(20), '±',
              str(self.smoothVpeak_err).ljust(20))

    # ======= Save the measured quantities to file ============================
    def saveData(self, filename=None):
        """ Save the fitting results to an ascii"""
        if filename is None:
            fn = '/home/axel/PhD/larsData/measurements/parametricLya.list'
        else:
            fn = filename
        # check if the file is emptymhm
        try:
            size = os.stat(fn).st_size
            if size == 0:
                empty = True
            else:
                empty = False
        except FileNotFoundError:
            # file does not exist <=> it is empty
            empty = True

        date = datetime.datetime.today().strftime('%Y-%m-%d')
        with open(fn, 'a') as f:
            header = ('# Lyman alpha parametric fitting results\n'
                      '# File initially generated by FitLya.ipynb on {}\n'
                      '# {}{}{}{}{}{}{}{}{}{}{}{}{}{}\n'
                      ).format(date, 'Galaxy:'.ljust(8),
                               'Best Fit Profile:'.ljust(30),
                               'Integrated flux:'.ljust(30),
                               'Integrated flux err:'.ljust(30),
                               'Weighted Skewness:'.ljust(30),
                               'Weighted Skewness err:'.ljust(30),
                               'CMG Skewness:'.ljust(30),
                               'CMG Skewness err:'.ljust(30),
                               'Peak Separation:'.ljust(30),
                               'Peak Separation err:'.ljust(30),
                               'Red Peak velocity:'.ljust(30),
                               'Red Peak velocity err:'.ljust(30),
                               'Smoothed Red Peak velocity:'.ljust(30),
                               'Smoothed Red Peak velocity err:'.ljust(30))
            if empty:
                f.write(header)

            data = ('{}{}{}{}{}{}{}{}{}{}{}{}{}{}\n'
                    '').format(str(self.name.split('_')[0]).ljust(10),
                               self.bestName.ljust(30),
                               str(self.parFlux).ljust(30),
                               str(self.parFlux_err).ljust(30),
                               str(self.Sw).ljust(30),
                               str(self.Sw_err).ljust(30),
                               str(self.CMGSkewness).ljust(30),
                               str(self.CMGSkewness_err).ljust(30),
                               str(self.peakSep).ljust(30),
                               str(self.peakSep_err).ljust(30),
                               str(self.vRedPeak).ljust(30),
                               str(self.vRedPeak_err).ljust(30),
                               str(self.smoothVpeak).ljust(30),
                               str(self.smoothVpeak_err).ljust(30))
            f.write(data)

        # Save the fit to a pickle
        saveDf = pd.DataFrame()
        saveDf['wl'] = self.lya_wl
        saveDf['flux'] = self.lya_flux
        saveDf['err'] = self.lya_err
        saveDf['parfit'] = self.fit_result

        fname = ('/home/axel/PhD/LARSlarsData/processed/'
                 'Lya_kinematics/parametric fits/'
                 '{}_new.pkl').format(self.name)
        saveDf.to_pickle(fname)
'''