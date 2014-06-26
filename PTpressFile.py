#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d 
import string

plt.ion()

# 2013-11-17 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
# 2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#                 changed free parameter T0 to T3
#                 changed equations accordingly 
# 2014-06-25 0.3  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#                 instead from atm file the functions now take the pressure
#                 from the pressure file
#                 the module name changed to PTpressFile.py  

# ==========================================================================
# Code takes a pressure array (e.g., extracted from a pressure file), 
# and 6 free parameters for inversion case, 5 for non-inversion case and
# generates parametric PT profile separately for inversion and non inversion 
# case, based on Equation 2 in Madhusudhan and Seager 2009. The profile is 
# then smoothed using 1D Gaussian filter. The pressure array needs to be 
# equally spaced in log space.
# ==========================================================================

def PT_Inversion(p, a1, a2, p1, p2, p3, T3):
     '''
     Calculates PT profile for inversion case
     based on Equation (2) from Madhusudhan & Seager 2009.
 
     Parameters
     ----------
     p:  1D array of floats
     a1: float
     a2: float
     p1: float
     p2: float
     p3: float
     T3: float
      
     Returns
     -------
     PT_Inver:  tupple of arrays that includes:
              - temperature and pressure arrays of every layer of the atmosphere 
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
                no kinks on Layer boundaries 

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
 

     Example:
    
     # array of pressures, equally spaced in log space 

     p = np.array([  1.00000000e-05,   1.17680000e-05,   1.38480000e-05,
                     1.62970000e-05,   1.91790000e-05,   2.25700000e-05,
                     2.65600000e-05,   3.12570000e-05,   3.67830000e-05,
                     4.32870000e-05,   5.09410000e-05,   5.99400000e-05,
                     7.05480000e-05,   8.30280000e-05,   9.77000000e-05,
                     1.14970000e-04,   1.35300000e-04,   1.59220000e-04,
                     1.87380000e-04,   2.20510000e-04,   2.59500000e-04,
                     3.05380000e-04,   3.59380000e-04,   4.22920000e-04,
                     4.97700000e-04,   5.85700000e-04,   6.89260000e-04,
                     8.11130000e-04,   9.54540000e-04,   1.12330000e-03,
                     1.32190000e-03,   1.55560000e-03,   1.83070000e-03,
                     2.15440000e-03,   2.53530000e-03,   2.98360000e-03,
                     3.51110000e-03,   4.13200000e-03,   4.86260000e-03,
                     5.72230000e-03,   6.73410000e-03,   7.92480000e-03,
                     9.32600000e-03,   1.09740000e-02,   1.29150000e-02,
                     1.51990000e-02,   1.78860000e-02,   2.10490000e-02,
                     2.47700000e-02,   2.91500000e-02,   3.43040000e-02,
                     4.03700000e-02,   4.75080000e-02,   5.59080000e-02,
                     6.57930000e-02,   7.74260000e-02,   9.11160000e-02,
                     1.07220000e-01,   1.26180000e-01,   1.48490000e-01,
                     1.74750000e-01,   2.05650000e-01,   2.42010000e-01,
                     2.84800000e-01,   3.35160000e-01,   3.94420000e-01,
                     4.64150000e-01,   5.46220000e-01,   6.42800000e-01,
                     7.56460000e-01,   8.90210000e-01,   1.04760000e+00,
                     1.23280000e+00,   1.45080000e+00,   1.70730000e+00,
                     2.00920000e+00,   2.36440000e+00,   2.78250000e+00,
                     3.27450000e+00,   3.85350000e+00,   4.53480000e+00,
                     5.33660000e+00,   6.28020000e+00,   7.39070000e+00,
                     8.69740000e+00,   1.02350000e+01,   1.20450000e+01,
                     1.41740000e+01,   1.66810000e+01,   1.96300000e+01,
                     2.31010000e+01,   2.71850000e+01,   3.19920000e+01,
                     3.76490000e+01,   4.43060000e+01,   5.21400000e+01,
                     6.13590000e+01,   7.22080000e+01,   8.49750000e+01,
                     1.00000000e+02])

     
     # random values imitate DEMC
     a1 = np.random.uniform(0.2  , 0.6 )
     a2 = np.random.uniform(0.04 , 0.5 )
     p3 = np.random.uniform(0.5  , 10  )
     p2 = np.random.uniform(0.01 , 1   )
     p1 = np.random.uniform(0.001, 0.01)
     T3 = np.random.uniform(1500 , 1700)

     # generates raw and smoothed PT profile
     PT_Inv, T_smooth = PT_Inversion(p, a1, a2, p1, p2, p3, T3)

     # returns full temperature array and temperatures at every point
     T, T0, T1, T2, T3 = PT_Inv[8], PT_Inv[9], PT_Inv[10], PT_Inv[11], PT_Inv[12]

     # sets plots in the middle 
     minT= min(T0, T2)*0.75
     maxT= max(T1, T3)*1.25

     # plots raw PT profile with equally spaced points in log space
     plt.figure(1)
     plt.clf()
     plt.semilogy(PT_Inv[0], PT_Inv[1], '.', color = 'r'     )
     plt.semilogy(PT_Inv[2], PT_Inv[3], '.', color = 'b'     )
     plt.semilogy(PT_Inv[4], PT_Inv[5], '.', color = 'orange')
     plt.semilogy(PT_Inv[6], PT_Inv[7], '.', color = 'g'     )
     plt.title('Thermal Inversion Raw', fontsize=14)
     plt.xlabel('T [K]'               , fontsize=14)
     plt.ylabel('logP [bar]'          , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(max(p), min(p))
     #plt.savefig('ThermInverRaw.png', format='png')
     #plt.savefig('ThermInverRaw.ps' , format='ps' )

     # plots smoothed PT profile
     plt.figure(2)
     plt.clf()
     plt.semilogy(T       , p, color = 'r')
     plt.semilogy(T_smooth, p, color = 'k')
     plt.title('Thermal Inversion Smoothed', fontsize=14)
     plt.xlabel('T [K]'                    , fontsize=14)
     plt.ylabel('logP [bar]'               , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(max(p), min(p) )
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

     # sets top of the atmosphere to p0 to have easy understandable equations
     p0 = min(p)
     print p0

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

     # defining arrays of pressures for every part of the PT profile
     p_l1     = p[(np.where((p >= min(p)) & (p < p1)))]
     p_l2_pos = p[(np.where((p >= p1)  & (p < p2)))]
     p_l2_neg = p[(np.where((p >= p2)  & (p < p3)))]
     p_l3     = p[(np.where((p >= p3)  & (p <= max(p))))]

     # sanity check for total number of levels
     check = len(p_l1) + len(p_l2_pos) + len(p_l2_neg) + len(p_l3)
     print  'Total number of levels in p: ', len(p)
     print  '\nLevels per levels in inversion case (l1, l2_pos, l2_neg, l3) are respectively: ', len(p_l1), len(p_l2_pos), len(p_l2_neg), len(p_l3)
     print  'Checking total number of levels in inversion case: ', check

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
      
     # concatenating all temperature arrays
     T_conc = np.concatenate((T_l1, T_l2_pos, T_l2_neg, T_l3))

     # PT profile
     PT_Inver = (T_l1, p_l1, T_l2_pos, p_l2_pos, T_l2_neg, p_l2_neg, T_l3, p_l3, T_conc, T0, T1, T2, T3)

     # smoothing with Gaussian_filter1d
     sigma = 4
     T_smooth = gaussian_filter1d(T_conc, sigma, mode='nearest')
    
     return PT_Inver, T_smooth



def PT_NoInversion(p, a1, a2, p1, p3, T3):
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

     Example:
     # array of pressures, equally spaced in log space 

     p = np.array([  1.00000000e-05,   1.17680000e-05,   1.38480000e-05,
                     1.62970000e-05,   1.91790000e-05,   2.25700000e-05,
                     2.65600000e-05,   3.12570000e-05,   3.67830000e-05,
                     4.32870000e-05,   5.09410000e-05,   5.99400000e-05,
                     7.05480000e-05,   8.30280000e-05,   9.77000000e-05,
                     1.14970000e-04,   1.35300000e-04,   1.59220000e-04,
                     1.87380000e-04,   2.20510000e-04,   2.59500000e-04,
                     3.05380000e-04,   3.59380000e-04,   4.22920000e-04,
                     4.97700000e-04,   5.85700000e-04,   6.89260000e-04,
                     8.11130000e-04,   9.54540000e-04,   1.12330000e-03,
                     1.32190000e-03,   1.55560000e-03,   1.83070000e-03,
                     2.15440000e-03,   2.53530000e-03,   2.98360000e-03,
                     3.51110000e-03,   4.13200000e-03,   4.86260000e-03,
                     5.72230000e-03,   6.73410000e-03,   7.92480000e-03,
                     9.32600000e-03,   1.09740000e-02,   1.29150000e-02,
                     1.51990000e-02,   1.78860000e-02,   2.10490000e-02,
                     2.47700000e-02,   2.91500000e-02,   3.43040000e-02,
                     4.03700000e-02,   4.75080000e-02,   5.59080000e-02,
                     6.57930000e-02,   7.74260000e-02,   9.11160000e-02,
                     1.07220000e-01,   1.26180000e-01,   1.48490000e-01,
                     1.74750000e-01,   2.05650000e-01,   2.42010000e-01,
                     2.84800000e-01,   3.35160000e-01,   3.94420000e-01,
                     4.64150000e-01,   5.46220000e-01,   6.42800000e-01,
                     7.56460000e-01,   8.90210000e-01,   1.04760000e+00,
                     1.23280000e+00,   1.45080000e+00,   1.70730000e+00,
                     2.00920000e+00,   2.36440000e+00,   2.78250000e+00,
                     3.27450000e+00,   3.85350000e+00,   4.53480000e+00,
                     5.33660000e+00,   6.28020000e+00,   7.39070000e+00,
                     8.69740000e+00,   1.02350000e+01,   1.20450000e+01,
                     1.41740000e+01,   1.66810000e+01,   1.96300000e+01,
                     2.31010000e+01,   2.71850000e+01,   3.19920000e+01,
                     3.76490000e+01,   4.43060000e+01,   5.21400000e+01,
                     6.13590000e+01,   7.22080000e+01,   8.49750000e+01,
                     1.00000000e+02])


     # random values imitate DEMC
     a1 = np.random.uniform(0.2  , 0.6 )
     a2 = np.random.uniform(0.04 , 0.5 )
     p3 = np.random.uniform(0.5  , 10  )
     p1 = np.random.uniform(0.001, 0.01)
     T3 = np.random.uniform(1500 , 1700)

     # generates raw and smoothed PT profile
     PT_NoInv, T_smooth = PT_NoInversion(p, a1, a2, p1, p3, T3)

     # returns full temperature array and temperatures at every point
     T, T0, T1, T3 = PT_NoInv[6], PT_NoInv[7], PT_NoInv[8], PT_NoInv[9]

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
     plt.ylim(max(p), min(p))
     #plt.savefig('NoThermInverRaw.png', format='png')
     #plt.savefig('NoThermInverRaw.ps' , format='ps' )


     # plots smoothed PT profile
     plt.figure(4)
     plt.clf()
     plt.semilogy(T       , p, color = 'r')
     plt.semilogy(T_smooth, p, color = 'k')
     plt.title('No Thermal Inversion Smoothed', fontsize=14)
     plt.xlabel('T [K]'                       , fontsize=14)
     plt.ylabel('logP [bar]'                  , fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(max(p), min(p))
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

     # sets top of the atmosphere to p0 to have easy understandable equations
     p0 = min(p)

     # temperature at point 3
     # calculated from boundary condition between layer 2 and 3
     T1 = T3 - (np.log(p3/p1) / a2)**2

     # temperature at point 1
     T0 = T1 - (np.log(p1/p0) / a1)**2

     # error message when Ts are < 0
     if T0<0 or T1<0 or T3<0:
          print 'T0, T1 and T3 temperatures are: ', T0, T1, T3
          raise ValueError('Input parameters give non-physical profile. Try again.')

     # defining arrays for every part of the PT profile
     p_l1     = p[(np.where((p >= min(p)) & (p < p1)))]
     p_l2_neg = p[(np.where((p >= p1)  & (p < p3)))]
     p_l3     = p[(np.where((p >= p3)  & (p <= max(p))))]

     # sanity check for total number of levels
     check = len(p_l1) + len(p_l2_neg) + len(p_l3)
     print  'Total number of levels: ', len(p)
     print  '\nLevels per levels in non inversion case (l1, l2_neg, l3) are respectively: ', len(p_l1), len(p_l2_neg), len(p_l3)
     print  'Checking total number of levels in non inversion case: ', check


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
     PT_NoInver = (T_l1, p_l1, T_l2_neg, p_l2_neg, T_l3, p_l3, T_conc, T0, T1, T3)

     # smoothing with Gaussian_filter1d
     sigma = 4
     T_smooth = gaussian_filter1d(T_conc, sigma, mode='nearest')

     return PT_NoInver, T_smooth


