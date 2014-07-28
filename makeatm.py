#! /usr/bin/env python

# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
# 
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# We welcome your feedback, but do not guarantee support.  Please send
# feedback or inquiries to both:
# 
# Jasmina Blecic <jasmina@physics.ucf.edu>
# Joseph Harrington <jh@physics.ucf.edu>
# 
# or alternatively,
# 
# Jasmina Blecic and Joseph Harrington
# UCF PSB 441
# 4000 Central Florida Blvd
# Orlando, FL 32816-2385
# USA
# 
# Thank you for testing TEA!
# ******************************* END LICENSE *******************************

import numpy as np
from initialPT import *
import os

# =============================================================================
# This code produces a pre-atm file in the format that TEA can read. It 
# consists of 4 functions:
# get_g()            calculates surface gravity, by taking the necessary 
#                    info from the tepfile provided
# radpress()         calculates radii for each pressure in the 
#                    atmosphere, by taking a tepfile, temperature, pressure 
#                    and  mean molecular mass array
# read_press_file()  reads the pressure file provided
# make_preatm()      writes a pre-atm file
#
# Possible user errors in making pre-atm file that conflict with TEA:
#  - H, and He elements as in_elem must be included
#  - H_g, He_ref and H2_ref species in out_spec must be included
#  - all in_elem must be included in the list of out_species with their states
#  - use elements names as they appear in the periodic table
#  - use species names as readJANAF.py produces them. See sorted version of the
#    conversion-record.txt for the correct names of the species. 
#  - If the code stalls at the first iteration of the first temperature, check 
#    if all elements that appear in the species list are included with their 
#    correct names.
#
# This code runs with the simple call: makeatm.py
# =============================================================================

# 2014-06-01      Oliver Bowman and Jasmina Blecic made first release TEA 
#                 version of makeatm()
# 2014-07-14      Jasmina Blecic, jasmina@physics.ucf.edu   
#                 Modified: makeatm() to add mean molar mass and radii 
#                           calculation, added get_g(), radpress() and 
#                           read_press_file() functions

# reads the tep file and calculates surface gravity
def get_g(tepfile):
    '''
    Calculates planetary surface gravity. Calls tep reader and 
    gets data needed for calculation (g = G*M/r^2). Returns
    surface gravity and surface radius.

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
    2014-06-11 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # universal gravitational constant
    G = 6.67384e-11 # m3/kg/s2 or Nm2/kg2

    # opens tepfile to read and get data
    tep = rd.File(tepfile)

    # get planet mass in Mjup
    planet_mass = tep.getvalue('Mp')
    planet_mass = np.float(planet_mass[0])

    # get planet radius in units of Rjup
    planet_rad = tep.getvalue('Rp')
    planet_rad = np.float(planet_rad[0])

    # conversion to kg and km
    Mjup = 1.89813e27    # kg
    Rjup = 69911         # km

    # mass and radius of the star in kg and km
    Mp = planet_mass * Mjup  # kg
    Rp = planet_rad  * Rjup * 1000.  # m

    # planet surface gravity
    # g = G*Mp/ Rp^2 in m/s^2
    g = G * Mp / (Rp**2)

    # convert Rp back to km
    Rp = Rp / 1000.

    return g, Rp


# calculate radii for each pressure in the atmosphere
def radpress(tepfile, temp, mu, pres):
    '''
    Given a pressure in bar, temperature in K, mean molecular
    mass in g/mol, the function calculates radii in km for each 
    layer in the atmosphere. It declares constants, calls get_g()
    function, allocates array for radii, and calculates radii for
    each pressure. The input pressure and temperature arrays must
    in descending order, because reference radius Rp is placed on
    the bottom of the atmosphere where pressure is largest. 
    All upper radii are calculated based on the reference level. 
    The final radii array is given in km, and converted back to 
    ascending order for pre-atm file.

    Parameters
    ----------
    g: float
       Surface gravity in m/s^2.
    R0: float
       Surface radius in km.
    T: array of floats
       Array containing temperatures in K for each layer in the atmosphere.
    mu: array of floats
       Array containing mean molecular mass in g/mol for each layer 
       from in the atmosphere.
    P: array of floats
       Array containing pressures in bar for each layer in the atmosphere.

    Returns
    -------
    rad: array of floats
        Array containing radii in km for each pressure layer in the atmosphere.

    Revisions
    ---------
    2014-07-04 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version

    '''

    # Define physical constants
    Boltzmann = 1.38065e-23 # J/K == (N*m) / K = kg m/s^2 * m / K
    Avogadro  = 6.022e23    # 1/mol

    # Calculate surface gravity
    g, Rp = get_g(tepfile)

    # Number of layers in the atmosphere
    n = len(pres)

    # Allocate array of radii
    rad = np.zeros(n)

    # Reverse the order of pressure and temperature array, so it starts from 
    #         the bottom of the atmosphere where first radii is defined as Rp
    pres = pres[::-1]
    temp = temp[::-1]

    # Andrews:"Introduction to Atmospheric Physics" page 26
    # rad2 - rad1 = (R * T1)/g * ln(p1/p2)
    # R = R*/mu, R* - universal gas constant, mu - mean molecular mass
    # R* = Avogadro * Boltzmann
    # rad2 = rad1 + (Avogadro * Bolztmann * T1) / (g * mu1) * ln(p1/p2)
    # to convert from g to kg, mu[i] need to be multiplied with 1000
    for i in np.arange(n):
        if i == 0:
            rad[0] = Rp
        else:
            rad[i] = rad[i-1] + (Avogadro * Boltzmann * temp[i-1]  * np.log(pres[i-1]/pres[i])) / (g * mu[i-1])

    # Reverse the order of calculated radii to write them in the right order
    #         in pre-atm file
    rad = rad[::-1]

    return rad


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


def make_preatm(curr_dir, tepfile, press_file, abun_file, in_elem, out_spec, pre_atm):
    '''
    This code produces a pre-atm file in the format that TEA can read it. It  
    defines the directory with user inputs, then it reads the pressure file and
    elemental dex abundance data file (default: abundances.txt). It trims the 
    abundance data to the elements of interest, and takes the column with the
    weights information to calculate mean molecular weight, based on an assumption
    that 85% of the atmosphere is filled with H2 and 15% with He. Then, it converts
    dex abundances of all elements of interest to number density and divides them 
    by the sum of all number densities in the mixture to get fractional abundances.
    It calls the get_Teff() function to calculate effective temperature of the 
    planer, produces initial PT free parameters for DEMC, plots the figures for
    user to check the output, call radpress() to calculate radii, and writes
    the data (radii, pressure, temperature, elemental abundances) into a pre-atm
    file.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Revisions
    ---------
    2014-07-12 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # Get inputs directory
    #inputs_dir = "TEA/inputs/pre_atm/"
    #if not os.path.exists(inputs_dir): os.makedirs(inputs_dir)
   
    # Read pressure data
    pres = read_press_file(press_file)

    # Read abundance data and convert to an array
    f = open(abun_file, 'r')
    abundata = []
    for line in f.readlines():
        if line.startswith('#'):
            continue
        else:
            l = [value for value in line.split()]
            abundata.append(l)
    abundata = np.asarray(abundata)
    f.close()

    # Trim abundata to elements we need
    in_elem_split = in_elem.split(" ")
    nelem  = np.size(in_elem_split)
    data_slice = np.zeros(abundata.shape[0], dtype=bool)
    for i in np.arange(nelem):
        data_slice += (abundata[:,1] == in_elem_split[i])

    # List of elements of interest and their corresponding data
    abun_trim = abundata[data_slice]

    # Take data and create list
    out_elem = abun_trim[:,1].tolist()
    out_dex  = abun_trim[:,2].tolist()

    # Convert strings to floats
    out_dex  = map(float, abun_trim[:,2])

    # Convert dex exponents to elemental counts
    out_num  = 10**np.array(out_dex)

    # Get fractions of elemental counts to the 
    #     total sum of elemental counts in the mixture
    out_abn  = (out_num / np.sum(out_num)).tolist()

    # Calculate mean molecular weight guess from H2 and He only
    H_weight  = np.float(abundata[1][4])
    He_weight = np.float(abundata[2][4])
    mu = 0.85 *  H_weight * 2 + 0.15 * He_weight
    mu_array = np.linspace(mu, mu, len(pres)) 

    # reads tepfile and returns Teff
    Teff = planet_Teff(tepfile)

    # produces initial free parameters
    PT_params = initialPT_freeParams(tepfile)

    # generates initial PT profile
    PT, T_smooth, p = PT_initial(tepfile, press_file, PT_params)

    # prints in terminal free parameters
    print '\n   Free parameters of the initial PT profile are: \n' + \
           '        a1                a2               p1               p3             T3\n\n' + str(PT_params)
    print

    # Call radpress function to calculate  radii
    rad = radpress(tepfile, T_smooth, mu_array, pres)
    
    # Convert fractions to strings in scientific notation
    for n in np.arange(np.size(out_abn)):
        out_abn[n] = str('%1.10e'%out_abn[n])

    # Make a list of labels
    out = ['#Radius'.ljust(10)] + ['Pressure'.ljust(10)] + ['Temp'.ljust(7)]
    for i in np.arange(nelem):
         out = out + [out_elem[i].ljust(16)]
    out = [out]

    # Number of layers in the atmosphere
    n_layers = len(pres)

    # Fill in data list
    for i in np.arange(n_layers):
        out.append(['%8.3f'%rad[i]] + ['%8.4e'%pres[i]] + \
                          ['%7.2f'%T_smooth[i]] + out_abn)

    # Pre-atm header with basic instructions
    header      = "# This is a TEA pre-atmosphere input file.            \n\
# TEA accepts a file in this format to produce species abundances as \n\
# a function of pressure and temperature.                            \n\
# Output species must be added in the line immediately following the \n\
# FINDSPEC marker and must be named to match JANAF converted names. "

    # Place file into inputs directory 
    #inputs_out = input_dir + pre_atm

    # Write pre-atm file
    f = open(pre_atm, 'w+')
    f.write(header + '\n\n')
    f.write('#FINDSPEC\n' + out_spec + '\n\n')
    f.write('#FINDTEA\n')
    for i in np.arange(n_layers + 1):
        # Radius list
        f.write(out[i][0].ljust(10) + ' ')
      
        # Pressure list
        f.write(out[i][1].ljust(10) + ' ')
    
        # Temp list
        f.write(out[i][2].ljust(7) + ' ')
    
        # Elemental abundance list
        for j in np.arange(nelem):
            f.write(out[i][j+3].ljust(16)+' ')
        f.write('\n')
    f.close()

    # plots the initialPT profile for a sanity check
    plot_out = plot_initialPT(curr_dir, tepfile, press_file, PT_params)
   
    # shows the plots until user closes them
    plt.show(block=True)

