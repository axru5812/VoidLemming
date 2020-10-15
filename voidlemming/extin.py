import scipy as S
#import numarray as NA
from astropy.io import fits as PF
#import specfunc as SF
import sys

""" 
	
	extin.py -- Extinction functions 
	--------------------------------

	Reddening laws implemented:
		Cal   Calzetti starburst law
		CCM   Cardelli, Clayton & Mathis Galactic law
		SMC   SMC starburst law

	.fits functionality
		FitsDeredden  Deredden a fits image given an extinction map
"""
	

#--------------------------------------------------------------------------
#EXTINCTION LAWS
#--------------------------------------------------------------------------
ER=3.2
Rv=4.05

def Cal(lam):
	"""
	CALZETTI Starburst law;  Calzetti et al 1994, Calzetti 2000 
	input: 
		lam = wavelength vector scipy array. 
		      must be vector but can write CCM(S.array([6563.]), 0.1)
	output: 
		ext = vector len(lam) with klam
	"""
	mulam=lam/1.e4
	klam=S.where( mulam<0.6300, \
		2.659*(-2.156 + 1.509/mulam - 0.198/mulam**2 + 0.011/mulam**3) + Rv, \
		2.659*(-1.857 + 1.040/mulam) + Rv )
	return klam


		
def Cal2(lam):
	if S.iterable(lam): 
		wl=lam
	else: 
		wl=S.array([lam])
	p11=1./0.11
	ff11=2.659*(-2.156+1.509*p11-0.198*p11**2+0.011*p11**3)+Rv
	p12=1./0.12
	ff12=2.659*(-2.156+1.509*p12-0.198*p12**2+0.011*p12**3)+Rv
	slope1=(ff12-ff11)/100.
	ff99=2.659*(-1.857+1.040/2.19)+Rv
	ff100=2.659*(-1.857+1.040/2.2)+Rv
	slope2=(ff100-ff99)/100.

	#loop began here: 
	ala=wl*1.E-4
	p=1./ala
	ff= S.zeros_like(p)

	i = (0.63 <= ala) & (ala <= 2.2)
	ff[i]=2.659*(-1.857+1.040*p[i])+Rv
	i = (0.12 <= ala) & (ala < 0.63)
	ff[i]=2.659*(-2.156+1.509*p[i]-0.198*p[i]**2+0.011*p[i]**3)+Rv
	i = (ala < 0.12)
	ff[i]=ff11+(wl[i]-1100.)*slope1
	i = (2.2 < ala)
	ff[i]=ff99+(wl[i]-21900.)*slope2
	
	return ff






def CCM(lam): 
	"""
	Galactic law;  Cardelli, Clayton & Mathis 1989
	input: 
		lam = wavelength vector scipy array. 
		      must be vector but can write CCM(S.array([6563.]), 0.1)
	output: 
		ext = vector len(lam) with klam
	"""


	lam1 = S.array([ x for x in lam if x <= 1250. ])
	lam2 = S.array([ x for x in lam if x > 1250. and x <= 1695. ])
	lam3 = S.array([ x for x in lam if x > 1695. and x <= 3030. ])
	lam4 = S.array([ x for x in lam if x > 3030. and x <= 9090. ])
	lam5 = S.array([ x for x in lam if x > 9090. ])

	lamx1 = 1.e4/lam1
	lamx2 = 1.e4/lam2
	lamx3 = 1.e4/lam3
	lamx4 = 1.e4/lam4
	lamx5 = 1.e4/lam5

	y1 = lamx1 - 8.
	ax1 = -1.073 - 0.628*y1 + 0.137*y1**2 - 0.070*y1**3
	bx1 = 13.670 + 4.257*y1 - 0.420*y1**2 + 0.374*y1**3
	xkl1 = ax1*ER + bx1

	y2 = lamx2 - 5.9
	ax2 =  1.752 - 0.316*lamx2 - 0.104 / ((lamx2-4.67)**2 + 0.341) - \
			0.04473*y2**2 - 0.009779*y2**3
	bx2 = -3.090 + 1.825*lamx2 + 1.206 / ((lamx2-4.62)**2 + 0.263) + \
			0.2130*y2**2 + 0.1207*y2**3
	xkl2 = ax2*ER + bx2

	y3 = lamx3 - 5.9
	ax3 =  1.752 - 0.316*lamx3 - 0.104 / ((lamx3-4.67)**2 + 0.341)
	bx3 = -3.090 + 1.825*lamx3 + 1.206 / ((lamx3-4.62)**2 + 0.263)
	xkl3 = ax3*ER + bx3

	y4 = lamx4 - 1.82
	ax4 = 1 + 0.17699*y4 - 0.50447*y4**2 - 0.02427*y4**3 + \
			0.72085*y4**4 + 0.01979*y4**5 - 0.77530*y4**6 + \
			0.32999*y4**7
	bx4 = 1.41338*y4 + 2.28305*y4**2 + 1.07233*y4**3 - 5.38434*y4**4 - \
			0.62251*y4**5 + 5.30260*y4**6 - 2.09002*y4**7
	xkl4 = ax4*ER + bx4

	ax5 =  0.574 * S.power(lamx5, 1.61);
	bx5 = -0.527 * S.power(lamx5, 1.61);
	xkl5 = ax5*ER + bx5

	xkl =  S.r_[xkl1,xkl2,xkl3,xkl4,xkl5]
	return xkl




def SMC(lam):
	"""
	SMC law; Prevot 1984
	input: 
		lam = wavelength vector scipy array. 
		      must be vector but can write SMC(S.array([6563.]), 0.1)
	output: 
		ext = vector len(lam) with klam
	"""

	xlamdat = S.array( \
			[ 1000., 1050., 1100., 1150., 1200., 1275., 1330., \
			  1385., 1435., 1490., 1545., 1595., 1647., 1700., \
			  1755., 1810., 1860., 1910., 2000., 2115., 2220., \
			  2335., 2445., 2550., 2665., 2778., 2890., 2995., \
			  3105., 3400., 3420., 3440., 3460., 3480., 3500. ])

	xextsmc = S.array( \
		    [ 20.58, 19.03, 17.62, 16.33, 15.15, 13.54, 12.52, \
			  11.51, 10.80, 9.84,  9.28,  9.06,  8.49,  8.01,  \
			  7.71,  7.17,  6.90,  6.76,  6.38,  5.85,  5.30,  \
			  4.53,  4.24,  3.91,  3.49,  3.15,  3.00,  2.65,  \
			  2.29,  1.91,  1.89,  1.87,  1.86,  1.84,  1.83 ])

	if not S.iterable(lam): lam = S.array([ lam ])
	lam1 = S.array([ x for x in lam if x <  xlamdat[0] ])
	lam2 = S.array([ x for x in lam if x >= xlamdat[0] and x < xlamdat[-1] ])
	lam3 = S.array([ x for x in lam if x >= xlamdat[-1] ])

	m=(xextsmc[0]-xextsmc[5])/(xlamdat[0]-xlamdat[5])
	xkl1=ER+xextsmc[0]+m*(lam1-xlamdat[0])

	xkl2=ER+SF.Rebin1D(xlamdat, xextsmc, lam2)

	xkl3=CCM(lam3)

	xkl =S.r_[xkl1,xkl2,xkl3]	
	if len(xkl)==1: xkl=xkl[0]
	return xkl




def SMCmix(lam,ebv):
	klam = SMC(lam)
	alpha = 0.4*klam*ebv*S.log(10.)
	fesc = (1. - S.e**(-alpha))/alpha
	return fesc




#--------------------------------------------------------------------------
#FITS MANIPULATION ROUTINES
#--------------------------------------------------------------------------

#####
##### The following method is commented out, pending conversion from numarray
#####	(NA) to numpy
#####
def FitsDeredden (inimage, outimage, ebvimage, lam, elaw='smc'):
	"""
	FitsDeredden  
		Deredden a fits image given an extinction map
	input: 
		inimage  = input .fits image
		outimage = input .fits image
		ebvimage = extinction map E(B-V) .fits file
		lam      = central wavelength of input .fits
		elaw     = 'smc'|'ccm'|'cal'
	"""

	Laws=['smc','ccm','cal']
	if elaw not in Laws:
		print (elaw, "not an implemented reddening law")
		print ("allowed laws:", Laws)
		sys.exit(1)

	hdulistIn =PF.open(inimage)
	hdulistEbv=PF.open(ebvimage)
	datIn =hdulistIn [0].data
	datEbv=hdulistEbv[0].data
	hdulistIn .close()
	hdulistEbv.close()

	if datIn.shape != datEbv.shape: 
		print (inimage, "and", ebvimage, "not of same dimensions")
		sys.exit(1)

	wavelength=S.array([lam])

	xdim=datEbv.shape[0]
	ydim=datEbv.shape[1]

	datEbv.shape=xdim*ydim
	datCor=NA.array( [ SMC(wavelength, ext)[0] for ext in datEbv ] )
	
	datCor.shape=xdim,ydim
	
	hduCor=PF.PrimaryHDU(datCor)
	hdulistCor=PF.HDUList([hduCor])
	hdulistCor.writeto(outimage)
	

