""" Main class for parametric Lyman alpha fitting
"""
import numpy as np
from astropy.modeling.models import Voigt1D


# ==============================================================================
#                         Lyman Alpha line models
# ==============================================================================
def gaussian(x, amplitude, center, sigma):
    """
        Gaussian profile
    """
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))


def gaussCont(x, amplitude, center, sigma, cont):
    """
        Gaussian function with an additional continuum term
    """
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + cont


def CMG_Profile(x, amp, center, sigmaLeft, sigmaRight, cont):
    """Returns  a CompositeMultiGaussian profile.
        i.e. a half-and-half composition of 2 gaussians of with the same center
        and amplitude but different sigma
    """
    # Step 1: Set up the split in 2 halves
    Left = np.where(x <= center)
    Right = np.where(x > center)

    # Step 2: Calculate the functions
    res = np.ones_like(x)
    res[Left] = gaussian(x, amp, center, sigmaLeft)[Left]
    res[Right] = gaussian(x, amp, center, sigmaRight)[Right]

    # Add in a continuum level as well
    return res + cont


def absorption_Profile(x, amp, center, sigma, cont):
    gauss = gaussian(x, amp, center, sigma)

    # Compensate for potential saturation
    res = np.ones_like(gauss) * cont - gauss
    return res


def PCG_Profile(x, CMGamp, CMGcenter, CMGsigmaLeft, CMGsigmaRight,
                ABSamp, ABScenter, ABSsigma,
                cont):
    """ Returns a classic P-cygni profile but with a CMG peak
    """
    # Step 1: Calculate the CMG profile
    Left = np.where(x <= CMGcenter)
    Right = np.where(x > CMGcenter)

    CMG = np.ones_like(x)
    CMG[Left] = gaussian(x, CMGamp, CMGcenter, CMGsigmaLeft)[Left]
    CMG[Right] = gaussian(x, CMGamp, CMGcenter, CMGsigmaRight)[Right]

    # Put in a gaussian absorption
    gauss = gaussian(x, ABSamp, ABScenter, ABSsigma)

    # Subtract and add from the continuum level
    res = cont - gauss + CMG

    return res


def DBH_Profile(x, LCMGamp, LCMGcenter, LCMGsigmaLeft, LCMGsigmaRight,
                RCMGamp, RCMGcenter, RCMGsigmaLeft, RCMGsigmaRight,
                ABSamp, ABScenter, ABSsigma,
                cont):
    """Returns a 'Double Horn' profile with two line peaks
       and absorption
    """
    # Step 1: Calculate the left CMG profile
    Left = np.where(x <= LCMGcenter)
    Right = np.where(x > LCMGcenter)

    LCMG = np.ones_like(x)
    LCMG[Left] = gaussian(x, LCMGamp, LCMGcenter, LCMGsigmaLeft)[Left]
    LCMG[Right] = gaussian(x, LCMGamp,
                           LCMGcenter, LCMGsigmaRight)[Right]

    # Step 2: Calculate the right CMG profile
    Left = np.where(x <= RCMGcenter)
    Right = np.where(x > RCMGcenter)

    RCMG = np.ones_like(x)
    RCMG[Left] = gaussian(x, RCMGamp, RCMGcenter, RCMGsigmaLeft)[Left]
    RCMG[Right] = gaussian(x, RCMGamp,
                           RCMGcenter, RCMGsigmaRight)[Right]

    # Step3: Put in a gaussian absorption
    ABS = gaussian(x, ABSamp, ABScenter, ABSsigma)

    # Subtract and add from the continuum level
    res = cont - ABS + LCMG + RCMG

    return res
