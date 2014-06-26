#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from PTpressFile import *
import sys
import string
import reader as rd
from initialPT import *
import makeP as mP

plt.ion()

# =============================================================================
# This code serves as an input generator for Transit code and as a free 
# parameter generator for DEMC. It works with parametrized PT profile done 
# in a similar fashion as in Madhusudhan and Seager 2009, but has
# a capability to be extended for other PT profiles. 
# 1. 
# The input generator for the Transit code: reads a pressure file, extracts
# list of pressures, reads a tepfile, calculates planet's effective temperature
# to constrain one free parameter (T3), gets array of free parameters from DEMC
# (for inversion: x species multipliers and 6 for PT profile; 
# for non-inversion: x species multipliers and 5 for PT profile),
# evaluates PT from free parameters and the pressure array, makes 1D array 
# of species multipliers and temperatures, and sends this array to Transit. 
# The code takes 4 arguments on the command line: 
# "InputConverter.py pressureFile tepfile number_of_species MadhuPT_Inv/MadhuPT_NoINv"
# Example: InputConverter.py pressure_file.txt WASP-43b.tep 4 MadhuPT_Inv
# User should edit the write_transit_input function, where the mark "EDIT" is,
# to set the names of the molecular species of interest. 
# 2.
# The initialPT profile is a free parameter generator for DEMC: it reads a
# pressure file, extracts list of pressures, reads a tepfile, calculates 
# planet's effective temperature to constrain one free parameter (T3), 
# generates 5 free parameters and plots an non-inverted (semi-adiabatic)
# initial PT profile. It uses initialPT module and returns free parameters 
# from the function initialPT_freeParams().
# The code takes 3 arguments on the command line: 
# "InputConverter.py pressureFile tepfile initialPT"
# Example: InputConverter.py pressure_file.txt WASP-43b.tep initialPT
# ============================================================================

# 2013-11-17 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
# 2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#      added new function planet_Teff
#      changed free parameter T0 to T3 and equations and functions accordingly
# 2014-04-17 0.3  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#      added initialPT profile and free parameter generator
# 2014-06-19 0.4   Jasmina Blecic, jasmina@physics.ucf.edu   Revision
#      added makeP submodule that produces a pressure file and changed the code
#      to read the pressure from the pressure file provided

# extracts a pressure array from an atm file provided (obsolete function)
def read_atm_file(atmfile):
     '''
     Reads an atmospheric file made in Transit (Patricio Rojo) format.
     The function takes the column with pressures and converts it to floats.
 
     Parameters
     ----------
     atmfile: ASCII file, atmosphric file that contains a column of pressures  
 
     Returns
     -------
     p: list of floats, pressures in the atmosphere 

     Notes
     -----
     The header of the atmospheric file has many lines and then the line
     that looks like:
     #rad-z     pres        temp              q_i...
     #-6797.21  0.3488E+07  5276.19885762853  2.288755146204095E-005 
     The function reads the lines after this header line and takes the data. 
     The pressure column is placed between 11th and 20th character place.

     Example
     -------
     atmfile = "atmosphJB.dat"
     p = read_atm_file(atmfile)
 
     Revisions
     ---------
     2013-11-17 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     '''
     
     # opens the atmospheric file to read
     f = open(atmfile, 'r')

     # allocates array of pressures
     pressures = []

     # finds the line of interest in the header
     for line in f:
          if string.find(line, '#rad-z pres temp q_i...') > -1:

               # reads the atmospheric file from the row below the header
               for line in f:
                    data = line.strip()

                    # appends the data in every row from character 11th to 20th
                    pressures = np.append(pressures, data[10:20])

               # converts strings to floats
               p = pressures.astype(float)

     return p


# extracts a pressure array from an pressure file provided
def read_press_file(press_file):
     '''
     Reads a pressure file. The function takes the column with pressures
     and converts it to floats.
 
     Parameters
     ----------
     press_file: ASCII file, pressure file that contains a column of pressures  
 
     Returns
     -------
     p: list of floats, pressures in the atmosphere 

     Example
     -------
     press_file = "pressure_file.txt"
     p = read_atm_file(press_file)
 
     Revisions
     ---------
     2014-06-19 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     '''
     
     # opens the atmospheric file to read
     f = open(press_file, 'r')

     # allocates array of pressure data and pressures
     press_data = []
     pressure   = []

     # reads lines and store in pressure data
     for line in f.readlines():
         l = [value for value in line.split()]
         press_data.append(l)
         pres_data_size = len(press_data)

     # reads pressures strings and convert them to floats
     for i in np.arange(pres_data_size - 1):
         pressure =  np.append(pressure, press_data[i+1][1])
     p = pressure.astype(float)
     f.close()

     return p


# reads the tep file and calculates planet's effective temperature
def planet_Teff(tepfile):
     '''
     Calculates planetary effective temperature. Calls tep reader 
     and gets data nedded for effective temperature calculation.
     The effective temperature is calculated assuming zero albedo, and 
     zero redistribution to the night side, i.e uniform dayside 
     redistribution.

     Parameters
     ----------
     tepfile: tep file, ASCII file
 
     Returns
     -------
     Teff: float 

     Example
     -------
     tepfile = "WASP-43b.tep"
     planet_Teff(tepfile)

     Revisions
     ---------
     2014-04-05 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     '''

     # opens tepfile to read and get data
     tep = rd.File(tepfile)

     # get stellar temperature in K
     stellarT = tep.getvalue('Ts')
     Tstar    = np.float(stellarT[0])

     # get stellar radius in units of Rsun
     stellarR = tep.getvalue('Rs')
     Rstar    = np.float(stellarR[0])

     # get semimajor axis in AU
     semimajor = tep.getvalue('a')
     a         = np.float(semimajor[0])

     # conversion to km
     AU   = 149597870.7   # km
     Rsun = 695500.0      # km

     # radius fo the star and semimajor axis in km
     Rstar = Rstar * Rsun # km
     a     = a * AU        # km

     # effective temperature of the planet 
     # Teff^4 = Teff*^4 * f * (Rstar/a)^2 * (1-A)
     # zero albedo, no energy redistribution to the night side A=0, f=1/2 
     Teff = Tstar * (Rstar/a)**0.5 * (1./2.)**0.25

     return Teff


# generates free parameters
def MPI_call_DEMC(noSpecies, MadhuPT, tepfile):
     '''
     This is a dummy function. A place holder for MPI call.
     It calls DEMC and returns free parameters, 
     6 for TP profile in inversion case and 5 in non-inversion case
     and x multipliers for molecular species.

     Parameters
     ----------
     noSpecies: float
     MadhuPT  : string
      
     Returns
     -------
     free_params: 1D array of floats, free parameters

     Notes
     -----
     This is an example for 4 major molecular species.
     Add other species of interest.
     fH2O , H20 multipleir
     fCO  , CO multiplier
     fCO2 , CO2 multipleir
     fNH3 , NH3 multiplier
     a1   , exponential factor
     a2   , exponential factor
     p1   , pressure at point 1
     p2   , pressure at point 2
     p3   , pressure at point 3
     T3   , temperature of the layer 3, isothermal layer
     
     Revisions
     ---------
     2013-11-20 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     T3 as free parameter instead of T0
     '''

     # takes Teff to constrain PT profile
     Teff = planet_Teff(tepfile)

     # range of free parameters for factors of molecular species 
     # (e. g., fH2O, fCO, fCO2, fNH3)
     species_range  = np.array([0.01, 20])

     # range of free parameters for Madhu's PT profile in inversion case
     # !!! Please talk to Jasmina before implementing this into DEMC !!!
     if MadhuPT == 'MadhuPT_Inv':
          print '\n ========= Inversion case ===========' 
          PTparams_range = np.array([
	                        (0.2 , 0.6  ),     # a1
	                        (0.04, 0.5  ),     # a2 
	                        (0.01, 0.001),     # p1
	                        (0.01, 1    ),     # p2
	                        (0.5 , 10   ),     # p3 
	                        (Teff , Teff*1.5), # T3 
                                            ])

     # range of free parameters for Madhu's PT profile in non-inversion case
     # !!! Please talk to Jasmina before implementing this into DEMC !!!
     elif MadhuPT == 'MadhuPT_NoInv':
          print '\n ========= Non-inversion case ===========' 
          PTparams_range = np.array([
                            (0.99 , 0.999    ), # a1
                            (0.19 , 0.21     ), # a2
	                        (0.1  , 0.01     ), # p1
	                        (1    , 5        ), # p3 
	                        (Teff , Teff*1.5 ), # T3 
                                            ])

     # total number of free parameters
     noFreeParams = noSpecies + PTparams_range.shape[0]

     # calls random uniform for all free parameters
     base = np.random.uniform(0., 1, noFreeParams)

     # free parameters of the species factors
     species_params = base[0:noSpecies] * (species_range[1] - species_range[0]) + species_range[0]

     # free parameters of the PT profile
     PT_params      = base[noSpecies:noFreeParams] * (PTparams_range[:,1] - PTparams_range[:,0]) + PTparams_range[:,0]

     # concatenates all free parameters
     free_params = np.concatenate((species_params, PT_params))

     # prints in terminal free parameters
     print '\nFree parameters of the species factors :\n' + str(free_params[0: noSpecies]) 
     print '\nFree parameters of the PT profile :\n' + str(free_params[noSpecies: noFreeParams])
     print

     return free_params


# generates PT profile
def PT_generator(p, free_params, noSpecies, MadhuPT):
     '''
     This is a PT generator, that takes the pressure array from a pressure 
     file, free parameters from DEMC, a number of molecular species, 
     and a string that defines inversion or non-inverison case and
     generates Madhu's PT profiles. It calls the PTpressFile.py module and
     returns inversion or non-inversion set of arrays for every layer in the
     atmosphere, and a Gaussian smoothed temperature array. 

     Parameters
     ----------
     p           : 1D array of floats 
     free_params : 1D array of floats
     noSpecies   : integer
     MadhuPT     : string
      
     Returns
     -------
     PT_Inv        : tuple of 1D arrays of floats
     T_smooth_Inv  : 1D array of floats
     PT_NoInv      : tuple of 1D arrays of floats
     T_smooth_NoInv: 1D array of floats

     Notes
     -----
     p             : pressure in the atmosphere 
     free_params   : free parameters taken from DEMC
     noSpecies     : number of species in the atmosphere
     MadhuPT       : string that defines inversion or non-inversion case
     PT_Inv        : tuple of temperatures and pressures for every layer in 
                     the atmosphere in inversion case
     T_smooth_Inv  : temperatures smoothed with Gaussian for inversion case
     PT_NoInv      : tuple of temperatures and pressures for every layer in 
                     the atmosphere in non-inversion case
     T_smooth_NoInv: temperatures smoothed with Gaussian for inversion case

     Inversion case:
          PT_Inv  = (T_conc, T_l1, p_l1, T_l2_pos, p_l2_pos, T_l2_neg, 
                          p_l2_neg,T_l3, p_l3, T1, T2, T3)
      
     Non-inversion case:
          PT_NoInv = (T_conc, T_l1, p_l1, T_l2_neg, p_l2_neg, T_l3, p_l3,
                          T1, T3)

     Revisions
     ---------
     2013-11-23 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     T3 as free parameter instead of T0
     '''

     # if sys.argument(4) defines inversion case, returns inverted PT profile
     if MadhuPT == 'MadhuPT_Inv':

          # pulls out 6 parameters for PT profile, inversion case
          a1, a2, p1, p2, p3, T3 = free_params[noSpecies:noSpecies+6]

          # call PTpressFile.py module, PT_Inversion function
          PT_Inv, T_smooth_Inv = PT_Inversion(p, a1, a2, p1, p2, p3, T3)
          return PT_Inv, T_smooth_Inv

     # if sys.argument(4) defines non-inversion case, returns non-inverted PT profile
     elif MadhuPT == 'MadhuPT_NoInv':

          # pulls out 5 parameters for PT profile, non-inversion case
          a1, a2, p1, p3, T3 = free_params[noSpecies:noSpecies+5]

          # call PTpressFile.py module, PT_NoInversion function
          PT_NoInv, T_smooth_NoInv = PT_NoInversion(p, a1, a2, p1, p3, T3)
          return PT_NoInv, T_smooth_NoInv


# plotting function
def plot(x, y, title, minX, maxX, linestyle, color, fig):
     '''
     This is a plotting function used for Gaussian smoothed PT profile.

     Parameters
     ----------
     x         : 1D array of floats 
     y         : 1D array of floats
     title     : string
     minX      : float
     maxX      : float
     linestyle : string
     color     : string
     fig       : integer

     Returns
     -------
     None

     Notes
     -----
     minX, maxX  : set plot in the middle of the figure 

     Revisions
     ---------
     2013-11-20 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     ''' 

     plt.figure(fig)
     plt.clf()
     plt.semilogy(x, y, color=color, linestyle=linestyle, linewidth=1)
     plt.title(title        , fontsize=14)
     plt.xlabel('T [K]'     , fontsize=14)
     plt.ylabel('logP [bar]', fontsize=14)
     plt.ylim(max(y), min(y))
     plt.xlim(minX, maxX)
     return


# plots PT profiles
def plot_PT(p, PT, T_smooth, MadhuPT):
     '''
     This function plots two figures:
     1.
     Madhu's PT profile for inversion or non-inversion case
     based on the string in sys.argument(4). It uses returned arrays from
     the PT generator,  so it can plot part by part PT curves.
     2. 
     Smoothed PT profile without kinks on layer transitions.

     Parameters
     ----------
     p           : 1D array of floats 
     PT          : tuple of 1D arrays of floats
     T_smooth    : 1D array of floats
     MadhuPT     : string
      
     Returns
     -------
     None

     Notes
     -----
     p             : pressure in the atmosphere 
     PT            : tuple of temperatures and pressures for every layer in 
                     the atmosphere
     T_smooth      : temperatures smoothed with Gaussian for inversion case
     MadhuPT       : string that defines inversion or non-inversion case

     Inversion case:
          PT_Inv  = (T_conc, T_l1, p_l1, T_l2_pos, p_l2_pos, T_l2_neg, 
                          p_l2_neg,T_l3, p_l3, T1, T2, T3)
      
     Non-inversion case:
          PT_NoInv = (T_conc, T_l1, p_l1, T_l2_neg, p_l2_neg, T_l3, p_l3,
                          T1, T3)

     Revisions
     ---------
     2013-11-20 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     excluded free_params as argument
                     added T0 to be read from the PT generator
     ''' 
    
     if MadhuPT == 'MadhuPT_Inv':
          # takes temperatures from PT generator
          T, T0, T1, T2, T3 = PT[8], PT[9], PT[10], PT[11], PT[12]

          # sets plots in the middle 
          minT= min(T0, T2)*0.75
          maxT= max(T1, T3)*1.25

          # plots raw PT profile
          plt.figure(1)
          plt.clf()
          plt.semilogy(PT[0], PT[1], '.', color = 'r'     )
          plt.semilogy(PT[2], PT[3], '.', color = 'b'     )
          plt.semilogy(PT[4], PT[5], '.', color = 'orange')
          plt.semilogy(PT[6], PT[7], '.', color = 'g'     )
          plt.title('Thermal Inversion Raw', fontsize=14)
          plt.xlabel('T [K]'               , fontsize=14)
          plt.ylabel('logP [bar]'          , fontsize=14)
          plt.xlim(minT  , maxT)
          plt.ylim(max(p), min(p))
          #plt.savefig('ThermInverRaw.png', format='png')
          #plt.savefig('ThermInverRaw.ps' , format='ps' )

          # plots Gaussian smoothing
          plot(T_smooth, p, 'Thermal Inversion Smoothed with Gaussian', minT, maxT, '-', 'b', 2)
          # overplots with the raw PT
          plt.semilogy(T, p, color = 'r')
          return

     elif MadhuPT == 'MadhuPT_NoInv':  
          # takes temperatures from PT generator
          T, T0, T1, T3 = PT[6], PT[7], PT[8], PT[9]

          # sets plots in the middle 
          minT= T0*0.75
          maxT= max(T1, T3)*1.25

          plt.figure(3)
          plt.clf()
          plt.semilogy(PT[0], PT[1], '.', color = 'r'     )
          plt.semilogy(PT[2], PT[3], '.', color = 'b'     )
          plt.semilogy(PT[4], PT[5], '.', color = 'orange')
          plt.title('No Thermal Inversion Raw', fontsize=14)
          plt.xlabel('T [K]'                  , fontsize=14)
          plt.ylabel('logP [bar]'             , fontsize=14)
          plt.xlim(minT  , maxT)
          plt.ylim(max(p), min(p))
          #plt.savefig('NoThermInverRaw.png', format='png')
          #plt.savefig('NoThermInverRaw.ps' , format='ps' )

          # plots Gaussian smoothing
          plot(T_smooth, p, 'No Thermal Inversion Smoothed with Gaussian', minT, maxT, '-', 'b', 4)
         # overplots with the raw PT
          plt.semilogy(T, p, color = 'r')
          return


# writes output file ==> Transit input file
def write_transit_input(Transit_input, p, MadhuPT):
     '''
     This function writes a Transit input file, that has species factors and 
     Gaussian smoothed temperatures of a PT profile in the second column.
     Where there is a comment "EDIT" type in names of molecular species.
     User needs to edit line 'species_strings', with correct number of
     species used and their factor's names.

     Parameters
     ----------
     Transit_input : 1D array of floats
     p             : 1D array of floats 
     MadhuPT       : string
      
     Returns
     -------
     None

     Notes
     -----
     First row gives header that defines inversion or non-inversion case.
     First column gives header of species factors and smoothed temperatures
     at every level.

     User needs to edit line 'species_strings', with correct number of
     species used and their factor's names.

     Transit_input : concatenated array of species factors and 
                     Gaussian smoothed temperatures of PT profile
     p             : pressure in the atmosphere 
     MadhuPT       : string that defines inversion or non-inversion case

     Revisions
     ---------
     2013-11-23 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     ''' 

     # opens file to write the output (i.e., Transit input)
     f = open("TransitInput"  , 'w')

     # creates file vertical header (first column)
     # allocates arrays of strings for species factors and 
     # smoothed temperatures at every level
     species_strings = ['fH20', 'fCO', 'fCO2', 'fNH3']   # EDIT
     T_strings = np.empty((len(p),), dtype='|O')
     for i in range(len(T_strings)):
          T_strings[i] = 'T(level' + str(i+1) + ')'
     header = np.concatenate((species_strings, T_strings))

     # catches if number of strings in the vertical header is not changed
     if int(len(header)) != int(len(Transit_input)):
          raise ValueError('\n\nPlease EDIT line "species_strings" in Input_converter.py in function "write_transit_input" with the names of factors of the molecular species used. Provide their correct number and names.\n')          

     # writes file horizontal header (first row)
     # to distinguish inversion and non-inversion case file 
     if MadhuPT == 'MadhuPT_Inv':
          f.write  ('# Inversion array     \n\n')

     elif MadhuPT == 'MadhuPT_NoInv':
          f.write('# Non-Inversion array \n\n') 

     # writes species factors and smoothed temperatures in one column
     for i in range(len(Transit_input)):
          f.write  ("%11s %0.2f \n" % (header[i], Transit_input[i]))

     # closes the file
     f.close()
     return


# plots PT profile and writes file with Transit input
def MPI_call_Transit(free_params, p, PT, T_smooth, Transit_input, MadhuPT):
     '''
     This is a dummy function. A place holder for MPI call. The MPI is called to
     hand out 1D array of species strings and Gaussian smoothed temperatures to
     Transit. This function plots PT profile and writes a Transit input file.

     Parameters
     ---------- 
     free_params : 1D array of floats
     p           : 1D array of floats
     PT          : tuple of 1D arrays of floats
     T_smooth    : 1D array of floats
     Transit_input : 1D array of floats
     MadhuPT       : string

     Returns
     -------
     file: ASCII file

     Notes
     ------
     free_params   : free parameters taken from DEMC
     p             : pressure in the atmospher
     PT            : tuple of temperatures and pressures for every layer in 
                     the atmosphere
     T_smooth      : temperatures smoothed with Gaussian for inversion case
     Transit_input : concatenated array of species factors and 
                     Gaussian smoothed temperatures of PT profile
     MadhuPT       : string that defines inversion or non-inversion case      

     Revisions
     ---------
     2013-11-20 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     '''

     # plots PT profile
     plot_PT(p, PT, T_smooth, MadhuPT)

     # writes transit input file
     return write_transit_input(Transit_input, p, MadhuPT)


# the loop
def loop(p, noSpecies, MadhuPT, tepfile):
     '''
     This is a loop that generates Madhu's PT profile (separately for inversion
     and non-inversion case) and species factors and sends 1D array of species
     factors and Gaussian smoothed temperatures to Transit. 
     The loop is stopped when Transit sends the command that first species factor <0.
     The loop gets an array of free parameters from DEMC, evaluates PT profile,
     catches-non physical cases, makes 1D array of species multipliers and smoothed
     temperatures, and writes this array to a file to be used as a Transit input.
     Currently, the loop waits for user to press 'Enter' to continue the loop.

     Parameters
     ---------- 
     p           : 1D array of floats
     noSpecies   : integer
     MadhuPT     : string
      
     Returns
     -------
     None

     Notes
     -----
     p             : pressure in the atmosphere 
     noSpecies     : number of species in the atmosphere
     MadhuPT       : string

     Revisions
     ---------
     2013-11-25 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     added tepfile as argument for free_params function
     '''

     while(True):
          # calls DEMC for random floats
          free_params = MPI_call_DEMC(noSpecies, MadhuPT, tepfile)

          # pulls out factors for molecular species of e.g., H2O, CO, CO2, NH3
          species_fact  = free_params[0:noSpecies]

          # checks if first species factor is less then 0 to end the loop
          if species_fact[0] < 0.:   #fH2O < 0
               break

          # generates PT profile for inversion      
          if MadhuPT == 'MadhuPT_Inv':
               # checks for non-physical profile
               try:
                    PT_Inv, T_smooth_Inv = PT_generator(p, free_params, noSpecies, 'MadhuPT_Inv')
               except ValueError:
                    print 'Input parameters give non-physical profile. Try again.\n'
                    raw_input("Press Enter to continue...")
                    continue

               # makes 1D array for Transit input
               Transit_input_Inv   = np.concatenate((species_fact, T_smooth_Inv))

               # plots PT and writes 1D array to a file to be used as Transit input
               MPI_call_Transit(free_params, p, PT_Inv, T_smooth_Inv, Transit_input_Inv, MadhuPT)

          # generates PT profile for non-inversion case 
          elif MadhuPT == 'MadhuPT_NoInv':

               # checks for non-physical profile
               try:
                    PT_NoInv, T_smooth_NoInv = PT_generator(p, free_params, noSpecies, MadhuPT)
               except ValueError:
                    print 'Input parameters give non-physical profile. Try again.\n'

               # makes 1D array for Transit input
               Transit_input_NoInv = np.concatenate((species_fact, T_smooth_NoInv))

               # plots PT and writes 1D array to a file to be used as a Transit input
               MPI_call_Transit(free_params, p, PT_NoInv, T_smooth_NoInv, Transit_input_NoInv, MadhuPT)

          # waits for user to continue the loop
          raw_input("Press Enter to continue...")


def main():
     '''
     This function defines number of arguments, reads the atmospheric file 
     given, extracts list of pressures from the atmospheric file, and 
     depending on the number of arguments either calls the loop to generate
     Transit input or generate free parameters for the initialPT profile for
     DEMC.

     Parameters
     ---------- 
     None
      
     Returns
     -------
     None or free parameters of the initialPT profile

     Notes
     -----
     To call main, user needs to provide the following arguments:
     arg(1), atmospheric file made in a format that Transit code uses 
             (Patricio Rojo)
     arg(2), tepfile
     arg(3), integer number of molecular species 
             or string that defines initial PT
     arg(4), to define whether it is a case of Madhu's inverted or 
             non-invereted atmosphere by specifying the string 
             "MadhuPT_Inv" for inverted atmosphere and the string 
             "MadhuPT-NoInv" for non-inverted atmosphere. 

     Revisions
     ---------
     2013-11-25 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
     2014-04-05 0.2  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     added new argument for tepfile
                     added planet_Teff function to calculate Teff
     2014-04-17 0.3  Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     added new argument for initial PT profile and free
                     parameter generator

     2014-06-19 0.4 Jasmina Blecic, jasmina@physics.ucf.edu   Revision
                     added makeP submodule that produces a pressure file 
                     and changed the code to read the pressure from the 
                     pressure file provided
     '''

     # counts number of arguments given
     noArguments = len(sys.argv)

     # prints usage if number of arguments different from 4 or initialPT profile
     if noArguments != 5 and not (noArguments == 4 and sys.argv[3] == 'initialPT'):
        print '\nUsage: InputConverter.py pressureFile tepfile number_of_species MadhuPT_Inv/MadhuPT_NoInv'

        print '\nExample for an inverted atmosphere:\nInputConverter.py pressure_file.dat WASP-43b.tep 4 MadhuPT_Inv\n'
        return
     
     # sets that the argument given is pressure file
     press_file = sys.argv[1]

     # sets that the argument given is tepfile
     tepfile = sys.argv[2]


     ################ INITIAL PT GENERATOR ####################
     # generates initial PT and returns free parameters
     # used by DEMC as an initial guess
     if sys.argv[3]=='initialPT':
          # reads press file and returns array of pressures
          p = read_press_file(press_file)

          # reads tepfile and returns Teff
          Teff = planet_Teff(tepfile)

          # produces initial free parameters
          PT_params = initialPT_freeParams(tepfile)

          # plots the initialPT profile for a sanity check
          plot_initialPT(tepfile, press_file)

          # prints in terminal free parameters
          print '\n   Free parameters of the initial PT profile are: \n' + '        a1                a2               p1               p3             T3\n\n' + str(PT_params)
          print

          # shows the plots until user closes them
          plt.show(block=True)

          return PT_params
     ############################################################

     # sets that the argument given is atmospheric file
     noSpecies = int(sys.argv[3])

     # sets inversion or non-inversion case
     MadhuPT = sys.argv[4]

     # catches wrong 4th argument
     if sys.argv[4] != 'MadhuPT_Inv' and sys.argv[4] != 'MadhuPT_NoInv':
        print '\nUsage: InputConverter.py pressureFile tepfile number_of_species MadhuPT_Inv/MadhuPT_NoInv'

        print '\nUse either MadhuPT_Inv string for inverted atmosphere, or MadhuPT_NoInv string for non-inverted atmosphere\n'
        return
     
     # reads pressure file and returns array of pressures
     p = read_press_file(press_file)
     print 'pressure file', p

     # reads tepfile and returns Teff
     Teff = planet_Teff(tepfile)

     # checking number of levels in the atmospheric file
     print '\nNumber of levels in the atmospheric file is: ', len(p)

     # calls the loop
     loop(p, noSpecies, MadhuPT, tepfile)


if __name__ == "__main__":
    main()




