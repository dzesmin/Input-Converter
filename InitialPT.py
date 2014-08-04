#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from PT import *

# ============================================================================
# Module that has two functions to produce an initial PT profile and plots it.
# It uses the non-inverted PT profile based on the Equation 2 in Madhusudhan 
# and Seager 2009. The profile is then smoothed using 1D Gaussian filter. The
# pressure array needs to be equally spaced in log space.
# ============================================================================

# 2014-04-08 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
# 2014-06-26 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Reversion
#                 instead from atm file the functions now take the pressure
#                 from the pressure file  
#                 plots are now automatically placed in the plots/ directory
# 2014-07-23 0.3  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#                 deleted initialPT_freeParams() function because it is not
#                 needed for the BART project


# generates initial PT profile
def PT_initial(tepfile, press_file, a1, a2, p1, p3, T3_fac):
     '''
     This is a initial PT profile generator that uses similar methodology as in
     Madhusudhan and Seager 2009. It takes a pressure array from the pressure
     file (needs to be equally spaced in the log space and in bar units), the
     tepfile and initial PT parameters. It returns an initial profile, that is
     semi-adiabatic, a set of arrays for every layer in the atmosphere, a
     Gaussian smoothed temperature array, and a pressure array.

     Parameters:
     -----------
     tepfile: tep file, ASCII file
     press_file: pressure file, ASCII file
     a1: float
     a2: float
     p1: float
     p3: float
     T3_fac: float
      
     Returns:
     --------
     PT        : tuple of 1D arrays of floats
     T_smooth  : 1D array of floats
     p         : 1D array of floats

     Notes:
     ------
     layer: there are 3 major layers in the atmosphere (see Madhusudhan and 
         Seager 2009)
     level: division of the atmosphere from the bottom to the top on equally
         spaced parts in log space
     p : array needs to be equally spaced in log space
     a1: exponential factor, 
         empirically determined to be within range (0.2, 0.6)
     a2: exponential factor, 
         empirically determined to be within range (0.04, 0.5)  
     b1: exponential factor, empirically determined to be 0.5
     b2: exponential factor, empirically determined to be 0.5
     T3_fac: factor the multiplies the T3 temperature value, empirically
         determing to be between (1, 1.5) in that way accounting for the
         possible spectral features

     Revisions:
     ----------
     2014-04-05 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-07-02 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     reads the pressure from the pressure file
     2014-07-23 0.3  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     introduced PT profile arguments
     '''

     # takes Teff to constrain PT profile
     Teff = planet_Teff(tepfile)

     # Calculate T3 temperature
     T3 = float(T3_fac) * Teff

     # takes pressure from the pressure file provided
     p = read_press_file(press_file)

     # reads pressures at the top and the bottom of the atmosphere
     bottom = max(p)
     top    = min(p)
     p0     = min(p)

     # reads number of layers from the pressure file
     noLevels = len(p)

     # the following set of equations derived using Equation 2
     # Madhusudhan and Seager 2009

     # temperature at point 1
     T1 = T3 - (np.log(p3/p1) / a2)**2

     # temperature at the top of the atmosphere
     T0 = T1 - (np.log(p1/p0) / a1)**2

     # error message when temperatures are < 0
     if T0<0 or T1<0 or T3<0:
          print 'T0, T1 and T3 temperatures are: ', T0, T1, T3
          raise ValueError('Input parameters give non-physical profile. Try again.')

     # log
     b  = np.log10(bottom)
     t  = np.log10(top)

     # equally spaced pressure space
     p = np.logspace(t, b, num=noLevels,  endpoint=True, base=10.0)

     # defining arrays for every part of the PT profile
     p_l1     = p[(np.where((p >= top) & (p < p1)))]
     p_l2_neg = p[(np.where((p >= p1)  & (p < p3)))]
     p_l3     = p[(np.where((p >= p3)  & (p <= bottom)))]

     # Layer 1 temperatures 
     T_l1 = (np.log(p_l1/p0) / a1)**2 + T0

     # Layer 2 temperatures decreasing part
     T_l2_neg = (np.log(p_l2_neg/p1) / a2)**2 + T1

     # Layer 3 temperatures
     T_l3 = np.linspace(T3, T3, len(p_l3))

     # concatenating all temperature arrays
     T_conc = np.concatenate((T_l1, T_l2_neg, T_l3))

     # PT profile
     PT = (T_l1, p_l1, T_l2_neg, p_l2_neg, T_l3, p_l3, T_conc, p, T0, T1, T3)

     # smoothing with Gaussian_filter1d
     sigma = 4
     T_smooth = gaussian_filter1d(T_conc, sigma, mode='nearest')

     return PT, T_smooth, p


# plots initial PT profile
def plot_initialPT(date_dir, tepfile, press_file, a1, a2, p1, p3, T3_fac):
     '''
     This function plots two figures:
     1.
     Initial PT profile part by part. It uses returned arrays from
     the PT_initial function.
     2. 
     Smoothed PT profile without kinks on layer transitions.

     Parameters:
     -----------
     date_dir: string
     tepfile: tep file, ASCII file
     press_file: pressure file, ASCII file
     a1: float
     a2: float
     p1: float
     p3: float
     T3_fac: float

     Notes:
     ------
     date_dir: current working directory where initial PT profile will be
     generated
      
     Returns:
     --------
     None

     Revisions:
     ----------
     2014-04-08 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-07-02 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Reversion
                     plots are now automatically placed in the plots/ directory
     2014-07-23 0.3  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     introduced current date directory and PT profile arguments
     '''

     # generates initial PT profile
     PT, T_smooth, p = PT_initial(tepfile, press_file, a1, a2, p1, p3, T3_fac)

     # takes temperatures from PT generator
     T, T0, T1, T3 = PT[6], PT[8], PT[9], PT[10]

     # sets plots in the middle 
     minT= T0 * 0.9
     maxT= T3 * 1.1

     # plots raw PT profile
     plt.figure(1)
     plt.clf()
     plt.semilogy(PT[0], PT[1], '.', color = 'r'     )
     plt.semilogy(PT[2], PT[3], '.', color = 'b'     )
     plt.semilogy(PT[4], PT[5], '.', color = 'orange')
     plt.title('Initial PT', fontsize=14)
     plt.xlabel('T [K]', fontsize=14)
     plt.ylabel('logP [bar]', fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(max(p), min(p))

     # Place plot into plots directory with appropriate name 
     plot_out1 = date_dir + '/InitialPT.png'
     plt.savefig(plot_out1) 

     # plots Gaussian smoothing
     plt.figure(2)
     plt.clf()
     plt.semilogy(T_smooth, p, '-', color = 'b', linewidth=1)
     plt.title('Initial PT Smoothed', fontsize=14)
     plt.xlabel('T [K]'     , fontsize=14)
     plt.ylabel('logP [bar]', fontsize=14)
     plt.ylim(max(p), min(p))
     plt.xlim(minT, maxT)

     # Place plot into the current working directory with appropriate name 
     plot_out2 = date_dir + '/InitialPTSmoothed.png'
     plt.savefig(plot_out2) 

     return 

