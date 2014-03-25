



PT.py 
# ===============================================================
# Code takes 6 free parameters for inversion case, 5 for 
# non-inversion case and number of levels in the atmosphere 
# and generates a parametric PT profile separately for inversion 
# and non inversion case, based on Equation 2 in Madhusudhan and 
# Seager 2009. Inner functions divide pressure on equally spaced 
# parts in log space, so pressure is not a parameter.
# The profile is then smoothed using 1D Gaussian filter.
# ===============================================================



PTatmfile.py 
# ==========================================================================
# Code takes a pressure array (e.g., extracted from an atmospheric file), 
# and 6 free parameters for inversion case, 5 for non-inversion case and
# generates parametric PT profile separately for inversion and non inversion 
# case, based on Equation 2 in Madhusudhan and Seager 2009. The profile is 
# then smoothed using 1D Gaussian filter. The pressure array needs to be 
# equally spaced in log space.
# ==========================================================================


atmosphJB.dat
#====================================================================
# An example of an atmospheric file format used for InputConverter.py
# with pressure equally spaced in log space.
#====================================================================



TransitInput
#=====================================================================
# Output of InputConverter.py and Input for Transit code. 
# First column is a header which lists molecular species of interest 
# and temperatures at each level in the atmosphere. Second column 
# contains data, 1D array of factors of molecular species and Gaussian 
# smoothed temperatures for PT profile. Top row defines whether it is 
# an inversion or non-inversion case. Number of levels is set in the 
# atmospheric file provided.
#=====================================================================



InputConverter.py
# ==============================================================================
# This is an input generator for Transit code that currently works with 
# parametrized PT profile similar as in Madhusudhan and Seager 2009, but has a
# capability to be extended for other PT profiles. It reads an atmospheric file, 
# extracts list of pressures, gets array of free parameters from DEMC (for 
# inversion: x species multipliers and 6 for PT  profile; for non-inversion:
# x species multipliers and 5 for PT profile), evaluates PT from free 
# parameters and the pressure array, makes 1D array of species multipliers and 
# temperatures, and sends this array to Transit. 
# The code takes 3 arguments on the command line: 
# "InputConverter.py atmfile number_of_species MadhuPT_Inv/MadhuPT_NoINv"
# The atmospheric file needs to provide a pressure array equally spaced in log
# space between 100 bar and 1e-5 bar. User should edit the write_transit_input 
# function, where the mark "EDIT" is to set names of the molecular species of 
# interest. The code uses MPI twice, to communicate with DEMC and Transit. 
# ==============================================================================
