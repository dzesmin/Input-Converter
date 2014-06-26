#! /usr/bin/env python

import numpy as np

def makeP(n_layers, p_top, p_bottom, filename, log=True): 
    '''
    Function to make the pressure array file to be used by BART.
  
    Parameters:
    -----------
    n_layers: integer
              Number of layers in the atmosphere.
    p_top: float
              Pressure at the top of the atmosphere.
    p_bottom:
              Pressure at the bottom of the atmopshere.
    filename: string
              Name of the file to be produced.
    log: boolean
             If True, logarithmic sampling of the pressure range is choosen,
             else linear.

    Returns:
    --------
             None

    Revisions
    ---------
    2014-06-22 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # If log=True the logarithmic sampling is chosen, else linear
    if log:
        pres = np.logspace( np.log10(np.float(p_bottom)), np.log10(np.float(p_top)), n_layers)  # Equispaced pressure 
    else:
        pres = np.linspace( p_bottom, p_top, n_layers)   # Equispaced pressure array

    # Write header line
    header      = "Layer  P [bar] \n"

    # Open file to write 
    f = open(str(filename), 'w+')
    f.write(header)

    for i in np.arange(n_layers):
        # Layer number
        f.write(str(i+1).ljust(6) + ' ')

        # Pressure list
        f.write(str(pres[i]).ljust(10) + ' ' + '\n')
    f.close()


# Call the function to execute
if __name__ == '__main__':
   
    # Set parameters
    n_layers = 100
    p_top    = 100
    p_bottom = 1e-5

    # Set filename
    filename = 'pressure_file.txt'  

    # Call the function
    makeP(n_layers, p_top, p_bottom, filename)

