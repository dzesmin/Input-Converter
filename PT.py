#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.ion()

# 2013-11-17 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
# 2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#                 changed free parameter T0 to T3
#                 changed equations accordingly 

# ===============================================================
# Code takes 6 free parameters for inversion case, 5 for 
# non-inversion case and number of levels in the atmosphere 
# and generates a parametric PT profile separately for inversion 
# and non inversion case, based on Equation 2 in Madhusudhan and 
# Seager 2009. Inner functions divide pressure on equally spaced 
# parts in log space, so pressure is not a parameter.
# The profile is then smoothed using 1D Gaussian filter.
# ===============================================================

# sets global variables empirically determined for hot Jupiters
# pressures at the top and the bottom of the atmosphere
bottom = 100    # bar
top    = 1e-5   # bar
p0     = 1e-5   # bar

def PT_Inversion(a1, a2, p1, p2, p3, T3, noLevels):
     '''
     Calculates PT profile for inversion case.
 
     Parameters
     ----------
     a1: float
     a2: float
     p1: float
     p2: float
     p3: float
     T3: float
      
     Returns
     -------
     PT_Inver:  tupple of arrays that includes:
              - temperature and pressure arrays for every layer of the atmosphere 
                (PT profile)
              - concatenated array of temperatures, 
              - temperatures at point 1, 2 and 3 (see Figure 1, Madhusudhan & Seager 2009)
          T_conc:   1D array of floats, temperatures concatenated for all levels 
          T_l1:     1D array of floats, temperatures for layer 1
          T_l2_pos: 1D array of floats, temperatures for layer 2 inversion part 
                    (pos-increase in temperatures)
          T_l2_neg: 1D array of floats, temperatures for layer 2 negative part
                    (neg-decrease in temperatures)
          T_l3:     1D array of floats, temperatures for layer 3 (isothermal part)   
          p_l1:     1D array of floats, pressures for layer 1   
          p_l2_pos: 1D array of floats, pressures for layer 2 inversion part 
                    (pos-increase in temperatures)   
          p_l2_neg: 1D array of floats, pressures for layer 2 negative part 
                    (neg-decrease in temperatures)    
          p_l3:     1D array of floats, pressures for layer 3 (isothermal part)    
          T1:       float, temperature at point 1  
          T2:       float, temperature at point 2 
          T3:       float, temperature at point 3  
     T_smooth:  1D array of floats, Gaussian smoothed temperatures, 
                no kinks on layer boundaries 

     Notes
     -----
     layer : there are 3 major layers in the atmosphere (see Madhusudhan and 
             Seager 2009)
     level : division of the atmosphere from the bottom to top on equally
             spaced parts in log space
     p : array needs to be equally spaced in log space from bottom to top 
         of the atmosphere
     a1: exponential factor, 
         empirically determined to be within range (0.2, 0.6)
     a2: exponential factor, 
         empirically determined to be within range (0.04, 0.5)  
     b1: exponential factor, empirically determined to be 0.5
     b2: exponential factor, empirically determined to be 0.5
 
     p0:     pressure at the top of the atmosphere
     top:    pressure at the top of the atmosphere
     bottom: pressure at the bottom of the atmosphere

     Example:
     # random values imitate DEMC
     a1 = np.random.uniform(0.2  , 0.6 )
     a2 = np.random.uniform(0.04 , 0.5 )
     p3 = np.random.uniform(0.5  , 10  )
     p2 = np.random.uniform(0.01 , 1   )
     p1 = np.random.uniform(0.001, 0.01)
     T3 = np.random.uniform(1500 , 1700)

     # sets number of levels
     noLevels = 100

     # generates PT profile
     PT_Inv, T_smooth = PT_Inversion(a1, a2, p1, p2, p3, T3, noLevels)

     # returns full temperature array and temperatures at every point
     T, p, T0, T1, T2, T3 = PT_Inv[8], PT_Inv[9], PT_Inv[10], PT_Inv[11], PT_Inv[12], PT_Inv[13]

     # sets plots in the middle 
     minT= min(T0, T2)*0.75
     maxT= max(T1, T3)*1.25

     # plots raw PT profile with equally spaced points in log space
     plt.figure(1) #, facecolor='red')
     plt.clf()
     plt.semilogy(PT_Inv[0], PT_Inv[1], '.', color = 'r')
     plt.semilogy(PT_Inv[2], PT_Inv[3], '.', color = 'b'     )
     plt.semilogy(PT_Inv[4], PT_Inv[5], '.', color = 'orange')
     plt.semilogy(PT_Inv[6], PT_Inv[7], '.', color = 'g'     )
     plt.title('Thermal Inversion Raw', fontsize=14)
     plt.xlabel('T [K]'               , fontsize=14)
     plt.ylabel('logP [bar]'          , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(bottom, top )
     #plt.savefig('ThermInverRaw.png', format='png')
     #plt.savefig('ThermInverRaw.ps' , format='ps' )

     # plots smoothed PT profile
     plt.figure(2)
     plt.clf()
     plt.semilogy(T        , p, color = 'r')
     plt.semilogy(T_smooth , p, color = 'k')
     plt.title('Thermal Inversion Smoothed', fontsize=14)
     plt.xlabel('T [K]'                    , fontsize=14)
     plt.ylabel('logP [bar]'               , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(bottom, top )
     #plt.savefig('ThermInverSmoothed.png', format='png')
     #plt.savefig('ThermInverSmoothed.ps' , format='ps' )

     Revisions
     ---------
     2013-11-14 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     added T3 as free parameter instead of T0
                     changed boundary condition equations accordingly
     '''

     # the following set of equations derived using Equation 2
     # Madhusudhan and Seager 2009

     # temperature at point 2
     # calculated from boundary condition between layer 2 and 3
     T2 = T3 - (np.log(p3/p2) / a2)**2

     # temperature at the top of the atmosphere
     # calculated from boundary condition between layer 1 and 2
     T0 = T2 + (np.log(p1/p2) / -a2)**2 - (np.log(p1/p0) / a1)**2 

     # temperature at point 1
     T1 = T0 + (np.log(p1/p0) / a1)**2

     # error message when temperatures ar point 1, 2 or 3 are < 0
     if T0<0 or T1<0 or T2<0 or T3<0:
          print 'T0, T1, T2 and T3 temperatures are: ', T0, T1, T2, T3
          raise ValueError('Input parameters give non-physical profile. Try again.')

     # log
     b  = np.log10(bottom)
     t  = np.log10(top)

     # equally spaced pressure space
     p = np.logspace(t, b, num=noLevels,  endpoint=True, base=10.0)

     # defining arrays of pressures for every part of the PT profile
     p_l1     = p[(np.where((p >= top) & (p < p1)))]
     p_l2_pos = p[(np.where((p >= p1)  & (p < p2)))]
     p_l2_neg = p[(np.where((p >= p2)  & (p < p3)))]
     p_l3     = p[(np.where((p >= p3)  & (p <= bottom)))]

     # sanity check for total number of levels
     check = len(p_l1) + len(p_l2_pos) + len(p_l2_neg) + len(p_l3)
     print  'Total number of levels: ', noLevels
     print  'Levels per layers (l1, l2_pos, l2_neg, l3) are respectively: ', len(p_l1), len(p_l2_pos), len(p_l2_neg), len(p_l3)
     print  'Checking total number of levels: ', check

     # the following set of equations derived using Equation 2
     # Madhusudhan and Seager 2009

     # Layer 1 temperatures
     T_l1 = (np.log(p_l1/p0) / a1)**2 + T0  
 
     # Layer 2 temperatures (inversion part)
     T_l2_pos = (np.log(p_l2_pos/p2) / -a2)**2 + T2
 
     # Layer 2 temperatures (decreasing part)
     T_l2_neg = (np.log(p_l2_neg/p2) / a2)**2 + T2

     # Layer 3 temperatures
     T_l3     = np.linspace(T3, T3, len(p_l3))
      
     # concatenating all temperature and pressure arrays
     T_conc = np.concatenate((T_l1, T_l2_pos, T_l2_neg, T_l3))
     p_conc = np.concatenate((p_l1, p_l2_pos, p_l2_neg, p_l3))

     # PT profile
     PT_Inver = (T_l1, p_l1, T_l2_pos, p_l2_pos, T_l2_neg, p_l2_neg, T_l3, p_l3, T_conc, p_conc, T0, T1, T2, T3)

     # smoothing with Gaussian_filter1d
     sigma = 4
     T_smooth = gaussian_filter1d(T_conc, sigma, mode='nearest')

     return PT_Inver, T_smooth



def PT_NoInversion(a1, a2, p1, p3, T3, noLevels):
     '''
     Calculates PT profile for non-inversion case.
 
     Parameters
     ----------
     p:  1D array of floats
     a1: float
     a2: float
     p1: float
     p3: float
     T3: float
      
     Returns
     -------
     PT_NoInver:  tupple of arrays that includes:
              - temperature and pressure arrays of every layer of the atmosphere 
                (PT profile)
              - concatenated array of temperatures, 
              - temperatures at point 1 and 3 (see Figure 1, Madhusudhan & Seager 2009)
          T_conc:   1D array of floats, temperatures concatenated for all levels 
          T_l1:     1D array of floats, temperatures for layer 1
          T_l2_neg: 1D array of floats, temperatures for layer 2
          T_l3:     1D array of floats, temperatures for layer 3 (isothermal part)  
          p_l1:     1D array of floats, pressures for layer 1   
          p_l2_neg: 1D array of floats, pressures for layer 2     
          p_l3:     1D array of floats, pressures for layer 3 (isothermal part)    
          T1:       float, temperature at point 1  
          T3:       float, temperature at point 3  
     T_smooth:  1D array of floats, Gaussian smoothed temperatures, 
                no kinks on layer boundaries 

     Notes
     -----
     The code uses just one equation for layer 2, assuming that decrease 
     in temperature in layer 2 is the same from point 3 to point 2 
     as is from point 2 to point 1.

     layer : there are 3 major layers in the atmosphere (see Madhusudhan and 
             Seager 2009)
     level : division of the atmosphere from the bottom to top on equally
             spaced parts in log space
     p : array needs to be equally spaced in log space from bottom to top 
         of the atmosphere
     a1: exponential factor, 
         empirically determined to be within range (0.2, 0.6)
     a2: exponential factor, 
         empirically determined to be within range (0.04, 0.5)  
     b1: exponential factor, empirically determined to be 0.5
     b2: exponential factor, empirically determined to be 0.5
 
     p0:     pressure at the top of the atmosphere
     top:    pressure at the top of the atmosphere
     bottom: pressure at the bottom of the atmosphere

     Example:
     # random values imitate DEMC
     a1 = np.random.uniform(0.2  , 0.6 )
     a2 = np.random.uniform(0.04 , 0.5 )
     p3 = np.random.uniform(0.5  , 10  )
     p1 = np.random.uniform(0.001, 0.01)
     T3 = np.random.uniform(1500 , 1700)

     # generates raw and smoothed PT profile
     PT_NoInv, T_smooth = PT_NoInversion(a1, a2, p1, p3, T3, noLevels)

     # returns full temperature array and temperatures at every point
     T, p, T0, T1, T3 = PT_NoInv[6], PT_NoInv[7], PT_NoInv[8], PT_NoInv[9], PT_NoInv[10]

     # sets plots in the middle 
     minT= T0*0.75
     maxT= max(T1, T3)*1.25

     # plots raw PT profile with equally spaced points in log space
     plt.figure(3)
     plt.clf()
     plt.semilogy(PT_NoInv[0], PT_NoInv[1], '.', color = 'r'     )
     plt.semilogy(PT_NoInv[2], PT_NoInv[3], '.', color = 'b'     )
     plt.semilogy(PT_NoInv[4], PT_NoInv[5], '.', color = 'orange')
     plt.title('No Thermal Inversion Raw', fontsize=14)
     plt.xlabel('T [K]'                  , fontsize=14)
     plt.ylabel('logP [bar]'             , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(bottom, top )
     #plt.savefig('NoThermInverRaw.png', format='png')
     #plt.savefig('NoThermInverRaw.png', format='ps' )

     # plots smoothed PT profile
     plt.figure(4)
     plt.clf()
     plt.semilogy(T       , p, color = 'r')
     plt.semilogy(T_smooth, p, color = 'k')
     plt.title('No Thermal Inversion Smoothed', fontsize=14)
     plt.xlabel('T [K]'                       , fontsize=14)
     plt.ylabel('logP [bar]'                  , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(bottom, top )
     #plt.savefig('NoThermInverSmoothed.png', format='png')
     #plt.savefig('NoThermInverSmoothed.ps' , format='ps' )

     Revisions
     ---------
     2013-11-16 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     added T3 as free parameter instead of T0
                     changed boundary condition equations accordingly
     '''

     # the following set of equations derived using Equation 2
     # Madhusudhan and Seager 2009

     # temperature at point 3
     # calculated from boundary condition between layer 2 and 3
     T1 = T3 - (np.log(p3/p1) / a2)**2

     # temperature at point 1
     T0 = T1 - (np.log(p1/p0) / a1)**2

     # error message when Ts are < 0
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

     # sanity check for total number of levels
     check = len(p_l1) + len(p_l2_neg) + len(p_l3)
     print  'Total number of levels: ', noLevels
     print  'Levels per layers (l1, l2_neg, l3) are respectively: ', len(p_l1), len(p_l2_neg), len(p_l3)
     print  'Checking total number of levels: ', check

     # the following set of equations derived using Equation 2
     # Madhusudhan and Seager 2009

     # Layer 1 temperatures 
     T_l1 = (np.log(p_l1/p0) / a1)**2 + T0

     # Layer 2 temperatures decreasing part
     T_l2_neg = (np.log(p_l2_neg/p1) / a2)**2 + T1

     # Layer 3 temperatures
     T_l3 = np.linspace(T3, T3, len(p_l3))

     # concatenating all temperature arrays
     T_conc = np.concatenate((T_l1, T_l2_neg, T_l3))

     # PT profile
     PT_NoInver = (T_l1, p_l1, T_l2_neg, p_l2_neg, T_l3, p_l3, T_conc, p, T0, T1, T3)

     # smoothing with Gaussian_filter1d
     sigma = 6
     T_smooth = gaussian_filter1d(T_conc, sigma, mode='nearest')

     return PT_NoInver, T_smooth


