from InputConverter import *
import os

# ====================================================================
# Module that has three functions to produce free parameters of an 
# initial PT profile, generates the profile, and plots it. Free 
# parameters returned by the function initialPT_freeParams will be
# used as an initial guess for DEMC.  
# ====================================================================

# 2014-04-08 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
# 2014-06-26 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Reversion
#                 instead from atm file the functions now take the pressure
#                 from the pressure file  
#                 plots are now automatically placed in the plots/ directory

# generates free parameters for initial PT profile
def initialPT_freeParams(tepfile):
     '''
     This function produces free parameters for the initial PT profile.
     It randomly samples empirically determined ranges for parameters.
     Temperature of the isothermal layer is constrained based on planet's
     effective temperature, assuming zero albedo and zero redistribution
     of the energy to the night side, and possible inclusion of the 
     spectral features in the range (1, 1.5)*Teff. 

     Parameters
     ----------
     tepfile: tep file, ASCII file
      
     Returns
     -------
     PT_params: 1D array of floats, free parameters

     Notes
     -----
     a1   , exponential factor
     a2   , exponential factor
     p1   , pressure at point 1
     p3   , pressure at point 3
     T3   , temperature of the layer 3, isothermal layer
     
     Revisions
     ---------
     2014-04-08 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     '''

     # takes Teff to constrain PT profile
     Teff = planet_Teff(tepfile)
     
     # sets empirical ranges of free parameters
     # temperature T3 constrained based on the planet dayside effective
     # temperature assuming zero albedo and zero redistribution to the night
     # side. The range set to Teff * (1, 1.5) to account for spectral features
     PTparams_range = np.array([
                            (0.99 , 0.999    ), # a1
                            (0.19 , 0.21     ), # a2
	                        (0.1  , 0.01     ), # p1
	                        (1    , 5        ), # p3 
	                        (Teff , Teff*1.5 ), # T3 
                                            ])

     # total number of free parameters
     noFreeParams = PTparams_range.shape[0]

     # calls random uniform for all free parameters
     base         = np.random.uniform(0., 1, noFreeParams)

     # free parameters of the PT profile
     PT_params    = base * (PTparams_range[:,1] - PTparams_range[:,0]) + PTparams_range[:,0]

     return PT_params


# calculates initial PT profile
def PT_initial(tepfile, press_file, PT_params):
     '''
     This is a PT profile generator that uses similar methodology as derived in
     Madhusudhan and Seager 2009. It takes an a pressure array from the pressure
     file, that needs to be equally spaced in the log space and in bar units and
     free parameters from initialPT_freeParams function.
     It returns an initial profile, that is semi-adiabatic, a set of arrays for
     every layer in the atmosphere, a Gaussian smoothed temperature array, and a
     pressure array.

     Parameters
     ----------
     tepfile: tep file, ASCII file
      
     Returns
     -------
     PT        : tuple of 1D arrays of floats
     T_smooth  : 1D array of floats
     p         : 1D array of floats

     Notes
     -----
     PT_NoInv      : tuple of temperatures and pressures for every layer in 
                     the atmosphere in non-inversion case
     T_smooth_Inv  : temperatures smoothed with Gaussian for inversion case
     p             : pressure in the atmosphere 

     Revisions
     ---------
     2014-04-05 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-07-02 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     reads the pressure from the pressure file
     '''

     # takes free parameters
     a1, a2, p1, p3, T3 = PT_params

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
     sigma = 6
     T_smooth = gaussian_filter1d(T_conc, sigma, mode='nearest')

     return PT, T_smooth, p


# plots PT profiles
def plot_initialPT(tepfile, press_file, PT_params):
     '''
     This function plots two figures:
     1.
     Initial PT profile part by part. It uses returned arrays from
     the PT_initial function.
     2. 
     Smoothed PT profile without kinks on layer transitions.

     Parameters
     ----------
     tepfile: tep file, ASCII file
      
     Returns
     -------
     None

     Revisions
     ---------
     2014-04-08 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-07-02 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Reversion
                     plots are now automatically placed in the plots/ directory
     '''
  
     # Get plots directory, create if non-existent
     #plots_dir = input_dir + "initialPT-plots/"
     #if not os.path.exists(plots_dir): os.makedirs(plots_dir)    

     # generates initial PT profile
     PT, T_smooth, p = PT_initial(tepfile, press_file, PT_params)

     # takes temperatures from PT generator
     T, T0, T1, T3 = PT[6], PT[8], PT[9], PT[10]

     # sets plots in the middle 
     minT= T0 * 0.9
     maxT= T3 * 1.1

     # plots raw PT profile
     plt.figure(1)
     plt.semilogy(PT[0], PT[1], '.', color = 'r'     )
     plt.semilogy(PT[2], PT[3], '.', color = 'b'     )
     plt.semilogy(PT[4], PT[5], '.', color = 'orange')
     plt.title('Initial PT', fontsize=14)
     plt.xlabel('T [K]', fontsize=14)
     plt.ylabel('logP [bar]', fontsize=14)
     plt.xlim(minT  , maxT)
     plt.ylim(max(p), min(p))

     # Place plot into plots directory with appropriate name 
     plot_out1 = 'InitialPT.png'
     plt.savefig(plot_out1) 
     #plt.savefig('InitialPT.ps' , format='ps' )

     # plots Gaussian smoothing
     plt.figure(2)
     plt.semilogy(T_smooth, p, '-', color = 'b', linewidth=1)
     plt.title('Initial PT Smoothed', fontsize=14)
     plt.xlabel('T [K]'     , fontsize=14)
     plt.ylabel('logP [bar]', fontsize=14)
     plt.ylim(max(p), min(p))
     plt.xlim(minT, maxT)

     # Place plot into plots directory with appropriate name 
     plot_out2 = 'InitialPTSmoothed.png'
     plt.savefig(plot_out2) 
     #plt.savefig('InitialPTSmoothed.ps' , format='ps' )

     return 

