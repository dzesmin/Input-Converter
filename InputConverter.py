import numpy as np

# reads final atm file and returns data of interest for BART
def readatm(atm_file, spec_mark='#FINDSPEC', tea_mark='#FINDTEA'):
    '''
    This function reads a TEA final atmospheric file and returns a list
    containing number of layers in the atmosphere, species names, radii,
    temperature, pressure, and each species abundances separately. 
    It opens an atm file to find markers for species and the TEA data, 
    retrieves the species list, reads data below the markers, and fills 
    out data into corresponding arrays. The final atmospheric file is 
    produces based on the initial PT profile, thus containing just initial
    PT values.

    Parameters:
    -----------
    atm_file:  ASCII file
               Pre-atm file that contains species, radius, pressure, 
               temperature, and elemental abundances data.
    spec_mark: string
               Marker used to locate species data in pre-atm file
               (located in the line immediately preceding the data).
    tea_mark:  string
               Marker used to locate radius, pressure, temperature, and 
               elemental abundances data (located in the line immediately
               preceding the data).

    Returns:
    --------
    PTdata: list of lists

    Notes:
    ------
    PTdata contains the following data as a list of lists:
    [nLayers, spec_list, radi_arr, pres_arr, temp_arr, spec1_abun, ... ,specN_abun]

    nLayers: float
               Number of runs TEA will execute for each T-P 
               (i.e., number of layers in the atmopshere) 
    spec_list: string list
               List containing names of molecular species.
    radi_arr:  float list
               List containing radius data.
    pres_arr:  float list
               List containing pressure data.
    temp_arr:  float list
               List containing temperature data.
    spec1_abun: float list
               List containing species 1 abundances for all layers.
    spec2_abun: float list
               List containing species 2 abundances for all layers.
    ...
    ...

    specN_abun: float list
               List containing species N abundances for all layers.

    Revisions:
    ----------
    2014-06-01 0.1  Oliver Bowman and Jasmina Blecic           Original version
    2014-08-03 0.2  Jasmina Blecic, jasmina@physics.ucf.edu    Revision
                    adapted for BART project, changed how the
                    file is read and what is returned. Made a 
                    list of abundances and a returning list of
                    lists containing all data of interest.
    '''
    
    # Open file to read
    f = open(atm_file, 'r')

    # Read data
    info = []
    for line in f.readlines():
        l = [value for value in line.split()]
        info.append(l)    
    f.close()

    # Initiate list of species and TEA markers 
    marker = np.zeros(2, dtype=int) 

    # Number of rows in file
    ninfo  = np.size(info)         
    
    # Set marker list to the lines where data start
    for i in np.arange(ninfo):
        if info[i] == [spec_mark]:
            marker[0] = i + 1
        if info[i] == [tea_mark]:
            marker[1] = i + 1
    
    # Retrieve species list using the species marker 
    spec_list  = info[marker[0]]       
    
    # Retrieve labels for data array
    data_label = np.array(info[marker[1]]) 

    # Number of labels in data array
    ncols      = np.size(data_label) 
 
    # Number of lines to read for data table (inc. label)  
    nrows      = ninfo - marker[1]     
    
    # Allocate data array
    data = np.empty((nrows, ncols), dtype=np.object)
    
    # Fill in data array
    for i in np.arange(nrows):
        data[i] = np.array(info[marker[1] + i])

    # Number of layers in the atmosphere
    nLayers = data.shape[0] - 1 
    
    # Take column numbers of non-element data
    iradi = np.where(data_label == '#Radius' )[0][0]
    ipres = np.where(data_label == 'Pressure')[0][0]
    itemp = np.where(data_label == 'Temp'    )[0][0]

    # Mark number of columns preceding element columns
    iatom = 3 
    
    # Place data into corresponding arrays, exclude labels
    radi_arr  = data[1:,iradi].tolist()      
    pres_arr  = data[1:,ipres].tolist()      
    temp_arr  = data[1:,itemp].tolist()      
    abund_arr = data[1:,iatom:]

    # Number of species
    nspec = len(spec_list) 

    # Allocate and fill out abundances in a list
    abundances = np.empty((len(spec_list), nLayers))
    for i in np.arange(len(spec_list)):
        abundances[i] = abund_arr[:, i].tolist() 
    abundances = abundances.tolist()
 
    # Make a list of data to return (without abundances)
    ReturnData = [nLayers, spec_list, radi_arr, pres_arr, temp_arr]

    # Total number of elements in the returning list
    nlists = nspec + len(ReturnData)

    # Make a list of lists to fillout all data of interest to return
    PTdata = [[] for _ in range(nlists)]
    for i in np.arange(len(ReturnData)):
        PTdata[i].append(ReturnData[i])
    for j in range(len(ReturnData), nlists, 1):
        PTdata[j].append(abundances[j-len(ReturnData)])
   
    # Returns the following data as a list of lists:
    # [nLayers, spec_list, radi_arr, pres_arr, temp_arr, spec1_abun, ... ,specN_abun]
    return PTdata

