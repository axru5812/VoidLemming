import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
import lmfit as lm
from .LyaFitClasses import baseFitter

from .data import abslines_uv

# from astropy.modeling.models import Voigt1D
from .plotfunctions import fill_between_steps

# sys.path.insert(0, "/home/axel/PhD/LARS/uv-spectroscopy/lyakinematics/")
from . import voigt

# sys.path.insert(0, "../optical-spectroscopy/")
from . import spechelpers as sh

# =============================================================================
#                          MAIN FITTING CLASS
# =============================================================================


class VoigtFitter(baseFitter):
    """ Class for handling matplotlib callback plots and fitting a voigt profile to it.
    The voigt profile is based on a tool written by X Prochaska (I think)

    Parameters
    ----------
    specfile : str
        The spectrum file to be used. Can be standard ascii or fits
    galaxyName : string
        Name of the galaxy in question - mostly used in titles
    user : str (optional)
        Which user is running the fit
    nmc : int
        Number of MC iterations used to calculate errors on returned parameters
    width : float
        the width of the window considered in the fitting. Basically sets what is plotted
    show_ism_lines : bool (default: False)
        Marks MW ISM lines
    vary_center : bool (default: True)
        Whether the centroid of the profile is kept fixed or as a free parameter of the fit
    redshift : float (optional)
        Redshift of the source. If None the spectrum is assumed to be in restframe.
        If none MW ISM lines will not be correctly marked
    velocity : float (optional)
        Extra velocity that can be used to shift the centroid of the absorption line.
        Can for instance be used to set the centroid to that of SII

    Methods
    -------
    export_fit
        This method will export the wavelength vector and voigt fit to an ascii file

    Properties
    ----------


    """

    def __init__(
        self,
        specfile,
        galaxyName,
        user="user",
        nmc=1000,
        width=4000,
        show_ism_lines=False,
        varyCenter=True,
        redshift=None,
        velocity=None,
    ):
        super().__init__(specfile)
        # Set version names
        self.__version__ = "0.0.1"

        # Assign user
        self.user = user

        # Set the name
        self.name = galaxyName

        # Set comment for galaxy
        self.comment = None

        # Set varying width or not
        self.varyCenter = varyCenter
        if self.varyCenter is False:
            self.comment = "Center fixed at vint"

        # Load the redshifts of the sample
        if redshift is not None:
            self.redshift = redshift
            self.restframe = False
        else:
            self.restframe = True
            self.redshift = 0

        if not self.restframe:
            # De-redshift the spectrum
            self.shift2Restframe(self.redshift)

        # Load velocity data
        if velocity is None:
            self.vint = 0
        else:
            self.vint = velocity
        self.vint_dwl = sh.dv2wl(self.vint, 1215.67)

        # SET UP THE PLOT
        # =====================================================================
        # Select the LymanAlpha region
        self.selectWindow(width=width)

        # Initialize variables that will be needed later
        self.contInclude = np.zeros_like(self.lya_wl, dtype=bool)
        self.lyaInclude = np.zeros_like(self.lya_wl, dtype=bool)
        self.maskInclude = np.zeros_like(self.lya_wl, dtype=bool)
        self.lyaStore = np.zeros_like(self.lya_wl, dtype=bool)

        # State variables:
        self.fitCont = True
        self.fitLya = False
        self.fitMask = False

        # Set the number of Monte Carlo iterations
        self.nmc = nmc

        # Set whether or not to show ISM lines
        self.show_ism_lines = show_ism_lines
        if self.show_ism_lines:
            self.ISMlines = self.readISMlines()

        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.createPlot()

    # -------------------------------------------------------------------------
    #                       Graphical methods
    # -------------------------------------------------------------------------
    def createPlot(self):
        # Create necessary axes
        axCont = self.fig.add_axes([0.93, 0.83, 0.068, 0.07])
        axLya = self.fig.add_axes([0.93, 0.73, 0.068, 0.07])
        axMask = self.fig.add_axes([0.93, 0.63, 0.068, 0.07])
        axClear = self.fig.add_axes([0.93, 0.53, 0.068, 0.07])
        axOk = self.fig.add_axes([0.93, 0.43, 0.068, 0.07])

        #                            Add buttons
        # ---------------------------------------------------------------------
        # Toggle continuum
        self.contButton = Button(axCont, "Cont")
        if self.fitCont:
            self.contButton.hovercolor = "#9bc8e8"
            self.contButton.color = "#5DA5DA"
        else:
            self.contButton.hovercolor = "#e6e6e6"
            self.contButton.color = "#cccccc"
        self.contButton.on_clicked(self._on_cont_button)

        # Toggle Lya
        self.lyaButton = Button(axLya, r"Ly $\alpha$")
        if self.fitLya:
            self.lyaButton.hovercolor = "#9bc8e8"
            self.lyaButton.color = "#5DA5DA"
        else:
            self.lyaButton.hovercolor = "#e6e6e6"
            self.lyaButton.color = "#cccccc"
        self.lyaButton.on_clicked(self._on_lya_button)

        # Toggle mask
        self.maskButton = Button(axMask, r"Mask")
        if self.fitMask:
            self.maskButton.hovercolor = "#9bc8e8"
            self.maskButton.color = "#5DA5DA"
        else:
            self.maskButton.hovercolor = "#e6e6e6"
            self.maskButton.color = "#cccccc"
        self.maskButton.on_clicked(self._on_Mask_button)

        # Clear current state
        self.clearButton = Button(axClear, "Clear")
        self.clearButton.color = "#F15854"
        self.clearButton.hovercolor = "#f58884"
        self.clearButton.on_clicked(self._on_clear_button)

        # Save the results
        self.okButton = Button(axOk, "Save")
        self.okButton.color = "#60BD68"
        self.okButton.hovercolor = "#90d095"
        self.okButton.on_clicked(self._on_ok_button)

        self.fig.canvas.draw()

        # Plot the data
        self.ax.step(self.lya_wl, self.lya_flux)
        # Plot the errors
        # self.ax.step(self.lya_wl, self.lya_err)
        fill_between_steps(
            self.ax,
            self.lya_wl,
            self.lya_flux - self.lya_err,
            self.lya_flux + self.lya_err,
            alpha=0.3,
        )
        # Set xlims
        self.ax.set_xlim(np.min(self.lya_wl), np.max(self.lya_wl))
        # Set ylims so that it does not change when fits are added
        ylims = self.ax.get_ylim()
        self.ax.set_ylim(ylims)

        # If showIsmLines is true, plot them
        if self.show_ism_lines:
            self.plotISMlines()

        # Set titles and labels:
        self.ax.set_title(self.name)
        self.ax.set_xlabel(r"Wavelength [$\AA$]")
        self.ax.set_ylabel(r"Flux [ergs s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")

        plt.subplots_adjust(
            left=0.08,
            bottom=0.15,
            right=0.92,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )

        self.span = SpanSelector(
            self.ax, self._onselect, "horizontal", useblit=True, minspan=0.1
        )
        self.fitPlot = None
        self.lyaPlot = None

    #              Deal with Milky Way ISM absorption
    # -------------------------------------------------------------------------
    def readISMlines(self):
        """ Read the Milky way absorption lines from data file
        """
        lines = pd.DataFrame.from_dict(abslines_uv.lines)
        lines.wl = lines.wl / (1 + self.redshift)
        return lines

    def plotISMlines(self):
        """ Plot any relevant lines in the plot
        """
        plotrange = self.ax.get_xlim()
        cond = np.where(
            (self.ISMlines.wl > plotrange[0])
            & (self.ISMlines.wl < plotrange[1])
        )
        plotLineWl = self.ISMlines.wl.values[cond]
        plotLineNm = self.ISMlines.name.values[cond]

        yrange = self.ax.get_ylim()

        for i, ln in enumerate(plotLineWl):
            self.ax.axvline(ln, color="#4d4d4d")
            self.ax.text(
                ln + 0.1, yrange[1] - yrange[1] / 15, plotLineNm[i], fontsize=8
            )
            self.ax.text(
                ln + 0.1,
                yrange[1] - 2 * yrange[1] / 15,
                str(np.round(plotLineWl[i] * (1 + self.redshift), decimals=2)),
                fontsize=8,
            )

    #              Define what happens when selecting a region
    # -------------------------------------------------------------------------
    def _onselect(self, vmin, vmax):
        """ Function called when SpanSelector is used
        """
        # Set everything between vmin and vmax to True.
        mask = np.where((self.lya_wl > vmin) & (self.lya_wl < vmax))
        if self.fitCont:
            self.contInclude[mask] = True
            self.ax.axvspan(
                vmin, vmax, color="#ef5f00", alpha=0.3, zorder=0, picker=True
            )
            self._fitContAndUpdate()

        elif self.fitLya:
            self.lyaInclude[mask] = True
            self.lyaStore[self.maskInclude] = self.lyaInclude[self.maskInclude]
            self.ax.axvspan(
                vmin, vmax, color="#60BD68", alpha=0.3, zorder=0, picker=True
            )
            self._fitLyaAndUpdate()

        elif self.fitMask:
            # Remove the masked regions from lya fitting
            self.lyaStore[mask] = self.lyaInclude[mask]

            self.lyaInclude[mask] = False
            self.maskInclude[mask] = True
            self.ax.axvspan(
                vmin, vmax, color="#F15854", alpha=0.5, zorder=0, picker=True
            )
            self._fitLyaAndUpdate()

        # Update the plot
        self.fig.canvas.draw()

    def replotAxvspan(self, boolarray, color, alpha=0.3):
        if np.any(boolarray):
            start = False
            end = False
            for i, boolean in enumerate(boolarray):
                print(i, boolean)
                if not start:
                    if boolean:
                        start = True
                        start_wl = self.lya_wl[i]
                        print(start_wl)
                else:
                    if not boolean:
                        end = True
                        end_wl = self.lya_wl[i]
                        print(end_wl)
                if end:
                    self.ax.axvspan(
                        start_wl, end_wl, color=color, alpha=alpha, zorder=0
                    )
                    start = False
                    end = False
        else:
            pass

    #                          Callback functions
    # -------------------------------------------------------------------------

    def _on_cont_button(self, event):
        # Set state variable to fit the continuum
        self.fitCont = True
        self.fitLya = False
        self.fitMask = False
        # Update Button colors:
        self.contButton.hovercolor = "#9bc8e8"
        self.contButton.color = "#5DA5DA"
        self.lyaButton.hovercolor = "#e6e6e6"
        self.lyaButton.color = "#cccccc"
        self.maskButton.hovercolor = "#e6e6e6"
        self.maskButton.color = "#cccccc"

    def _on_lya_button(self, event):
        self.fitLya = True
        self.fitCont = False
        self.fitMask = False
        # Update Button colors:
        self.lyaButton.hovercolor = "#9bc8e8"
        self.lyaButton.color = "#5DA5DA"
        self.contButton.hovercolor = "#e6e6e6"
        self.contButton.color = "#cccccc"
        self.maskButton.hovercolor = "#e6e6e6"
        self.maskButton.color = "#cccccc"

    def _on_Mask_button(self, event):
        self.fitLya = False
        self.fitCont = False
        self.fitMask = True
        # Update Button colors:
        self.maskButton.hovercolor = "#9bc8e8"
        self.maskButton.color = "#5DA5DA"
        self.lyaButton.hovercolor = "#e6e6e6"
        self.lyaButton.color = "#cccccc"
        self.contButton.hovercolor = "#e6e6e6"
        self.contButton.color = "#cccccc"

    def _on_clear_button(self, event):
        # Reset Graph
        self.ax.cla()
        self.createPlot()

        # Reset include vector
        if self.fitCont:
            self.contInclude = np.zeros_like(self.lya_wl, dtype=bool)
            # Replot the non-cleared axvspan
            self.replotAxvspan(self.lyaInclude, color="#60BD68")
            self.replotAxvspan(self.maskInclude, color="#F15854", alpha=0.5)

            self.fitPlot = None

        elif self.fitLya:
            self.lyaInclude = np.zeros_like(self.lya_wl, dtype=bool)
            self.lyaStore = np.zeros_like(self.lya_wl, dtype=bool)

            # Replot the non-cleared axvspan
            self.replotAxvspan(self.contInclude, color="#ef5f00")
            self.replotAxvspan(self.maskInclude, color="#F15854", alpha=0.5)
            if self.fitPlot is None:
                self.fitPlot = self.ax.plot(self.lya_wl, self.cont)
            else:
                self.fitPlot[0].set_data(self.lya_wl, self.cont)

        elif self.fitMask:
            # Use maskInclude to reset the lya include to it's previous value
            self.lyaInclude[self.maskInclude] = self.lyaStore[self.maskInclude]

            self.lyaStore = np.zeros_like(self.lya_wl, dtype=bool)
            self.maskInclude = np.zeros_like(self.lya_wl, dtype=bool)

            # Replot the axspans
            self.replotAxvspan(self.contInclude, color="#ef5f00")
            self.replotAxvspan(self.lyaInclude, color="#60BD68")

            # Replot continuum fit
            if self.fitPlot is None:
                self.fitPlot = self.ax.plot(self.lya_wl, self.cont)
            else:
                self.fitPlot[0].set_data(self.lya_wl, self.cont)

    def _on_ok_button(self, event):
        # Save the data to file:
        fn = "./voigtMeasurements.dat"

        # check if the file is empty
        try:
            size = os.stat(fn).st_size
            if size == 0:
                empty = True
            else:
                empty = False
        except FileNotFoundError:
            # file does not exist <=> it is empty
            empty = True

        with open(fn, "a") as f:
            if empty:
                ln = "#Galaxy\tEW\tlogN\tlogN err\tcomment\n"
                f.write(ln)

            if self.comment is None:
                ln = ("{}\t{}\t{}\t{}\n").format(
                    self.name, self.vEW[0], self.logN, self.logNerr
                )
            else:
                ln = ("{}\t{}\t{}\t{}\t#{}\n").format(
                    self.name,
                    self.vEW[0],
                    self.logN,
                    self.logNerr,
                    self.comment,
                )
            f.write(ln)
        # Make sure the figures directory exists
        if not os.path.exists("./voigtFigures/"):
            os.makedirs("./voigtFigures/")
        # Save the plot
        filename = "./voigtFigures/" + "{}.png".format(self.name)
        self.fig.savefig(filename)

        # place a text box in upper left in axes coords
        props = dict(boxstyle="round", facecolor="#60BD68", alpha=0.9)
        textstr = "Galaxy data saved\nsuccessfully"
        self.ax.text(
            0.8,
            0.95,
            textstr,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        # Subtract the current fit
        self.removeVoigt()

        # Save the spectrum and the Voigt fit in an easy-to-load format
        saveDF = pd.DataFrame()
        saveDF["lambda"] = self.lya_wl
        saveDF["flux"] = self.lya_flux
        saveDF["error"] = self.lya_err
        saveDF["voigt"] = self.eval_res

        path = (
            "/home/axel/PhD/larsData/processed/Lya_kinematics/voigtfits/"
            "{}.pkl"
        ).format(self.name)
        saveDF.to_pickle(path)

    # -------------------------------------------------------------------------
    #                        Science functions
    # -------------------------------------------------------------------------
    def _fitContAndUpdate(self):
        """ Fits a simple linear polynomial to the selected continuum regions
        """
        # Include based on self.include
        fit_Data = self.lya_flux[self.contInclude]
        fit_Data_w = self.lya_wl[self.contInclude]

        coeff, cov = np.polyfit(fit_Data_w, fit_Data, 1, cov=True)
        self.k = coeff[0]
        self.m = coeff[1]
        self.k_err, self.m_err = self.contMonteCarlo()

        self.contPoly = np.poly1d(coeff)
        self.cont = self.contPoly(self.lya_wl)
        self.contLC, self.contLCerr = self.calculateLyaCont()

        if self.fitPlot is None:
            self.fitPlot = self.ax.plot(self.lya_wl, self.cont)
        else:
            self.fitPlot[0].set_data(self.lya_wl, self.cont)
        plt.draw()

        self.transf_x = self.lya_wl
        self.transf_y = self.lya_flux - self.cont
        self.transf_err = self.lya_err

    def _fitLyaAndUpdate(self):
        """
            This fits a Voigt Profile to the selected regions
        """
        incFlux = self.lya_flux[self.lyaInclude] / self.cont[self.lyaInclude]
        # global lyacont
        # lyacont = self.cont[self.lyaInclude].copy()
        incFlux = incFlux.copy()
        incWl = self.lya_wl[self.lyaInclude]

        vProfile = self.setupVoigt()

        voigtModel = lm.Model(vProfile)
        voigtModel.set_param_hint("logN", value=21, min=17, max=22)
        voigtModel.set_param_hint("b", value=1, min=0, max=50)
        voigtModel.set_param_hint(
            "wrest",
            value=(1215.67 + self.vint_dwl),
            max=1218,
            min=1213,
            vary=self.varyCenter,
        )
        voigtParams = voigtModel.make_params()

        # Run the fit
        result = voigtModel.fit(incFlux, x=incWl, params=voigtParams)
        self.bestFitParams = result.params
        self.voigtModel = voigtModel
        # Plot the fit
        self.eval_res = (
            voigtModel.eval(x=self.lya_wl, params=result.params) * self.cont
        )

        if self.lyaPlot is None:
            self.lyaPlot = self.ax.plot(self.lya_wl, self.eval_res, c="#B276B2")
        else:
            self.lyaPlot[0].set_data(self.lya_wl, self.eval_res)
        plt.draw()

        # Calculate total flux
        flux = np.trapz(self.cont - self.eval_res, x=self.lya_wl)

        # Save the params
        self.vEW = flux / self.contPoly([1215.67])
        self.logN = self.bestFitParams["logN"].value
        self.logNerr = self.bestFitParams["logN"].stderr

        self.removeVoigt()

    def removeVoigt(self):
        self.transf_x = self.lya_wl
        self.transf_y = self.lya_flux - self.eval_res
        self.transf_err = self.lya_err

    def contMonteCarlo(self):
        ks = []
        ms = []
        for i in range(self.nmc):
            data = self.redrawData()
            # Include based on self.include
            fit_Data = data[self.contInclude]
            fit_Data_w = self.lya_wl[self.contInclude]

            coeff, cov = np.polyfit(fit_Data_w, fit_Data, 1, cov=True)
            ks.append(coeff[0])
            ms.append(coeff[1])
        k_err = np.std(np.array(ks))
        m_err = np.std(np.array(ms))
        return k_err, m_err

    def lyaMonteCarlo(self):
        ints = []
        for i in range(self.nmc):
            data = self.redrawData()
            intF = data[self.lyaInclude] - self.cont[self.lyaInclude]
            intW = self.lya_wl[self.lyaInclude]

            ints.append(np.trapz(intF, intW))
        int_err = np.std(np.array(ints))
        return int_err

    def redrawData(self):
        errors = (
            np.random.normal(loc=0, scale=1, size=len(self.lya_err))
            * self.lya_err
        )
        newdata = self.lya_flux + errors
        return newdata

    def calculateLyaCont(self):
        poly = np.poly1d([self.k, self.m])
        contLC = poly([1215.67])
        contLCerr = np.sqrt(1215.67 * self.k_err ** 2 + self.m_err ** 2)
        return contLC[0], contLCerr

    def calculateEW(self):
        # Calculate the EW:
        lyaEW = self.lyaIntegral / self.contLC
        lyaEWerr = np.sqrt(
            ((1 / self.contLC) * self.lyaIntErr) ** 2
            + ((self.lyaIntegral / self.lyaIntegral ** 2) * self.contLCerr) ** 2
        )
        return lyaEW, lyaEWerr

    def setupVoigt(self):
        """
            This sets up a function for the current voigt profile giving it
            access to whatever parameters it needs from the main class

            Parameters:
            ------------
                None

            Returns:
            ------------

        """
        dwl = self.vint_dwl

        def profile(x, logN, b, wrest):
            # convert wavelengths to cm:
            wave = x / 1e8
            # Set redshift to 0 since we are in restframe
            z = 0.0

            # Set restframe wavelength to lyman alpha
            #   This might be altered when we take the outflow velocity into
            #   account
            # wrest = 1215.67 + dwl

            # Set the oscillator strength
            #   www.uio.no/studier/emner/matnat/astro/AST4320/h14/timeplan/lecture14slides.pdf
            f = 0.416

            # Set the gamma = gj + gk = gk  = 6.3*1e8
            #   http://www.ita.uni-heidelberg.de/~rowan/ISM_lectures/script-part1.pdf
            # 6.25 from Dijksta enl. Jens
            gamma = 6.25 * 1e8

            # Get taus
            tau = voigt.voigt_tau(
                wave, [logN, z, b * 1e5, wrest / 1e8, f, gamma]
            )
            flux = np.exp(-1 * tau)

            return flux

        return profile

    def set_comment(self, comment):
        if comment == "":
            self.comment = None
        else:
            self.comment = comment

    def export_fit(self, filename):
        df = pd.DataFrame()
        df["wl"] = self.wl
        df["voigt"] = self.voigtModel.eval(
            x=self.wl, params=self.bestFitParams
        ) * self.contPoly(self.wl)
        df.to_csv(filename, sep="\t", index=False)


# ============================================================================
# DEFINE VOIGT MODEL


def voigt_Profile(x, center, g_width, l_width, l_amp):
    """

    """
    g_fwhm = g_width * 2.355
    voigt = Voigt1D(
        x_0=center, amplitude_L=l_amp, fwhm_L=l_width, fwhm_G=g_fwhm
    )

    res = -1 * voigt(x)

    return res
