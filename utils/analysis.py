import sys, os, glob, csv, json, datetime
import warnings
import numpy as np
import mne

def pev(epochs_fname,predictor,condition):
    """ calculate PEV (proportion of explained variance) w.r.t. predictor
        for now PEV = r2 from linear regression
        pev for categorical predictors will be added later

    Parameters
    ----------
    epochs_fname : str   
        name of epochs file to load
    predictor :  str 
        it should be the name of one of the metadata columns. 
        predictor should be numerical (categorical predictors will be added later)
    selection :  str
        mne query to extract subset of the data (f.e. 'paradigm == auditory')

    Returns
    -------
    pev : float
        proportion of explained variance, <= 1
    """

    epochs = mne.read_epochs(epochs_fname)
   
    predictor_values = epochs.metadata[predictor]

    # get avg for each epoch
    responses = np.mean(np.squeeze(epochs[condition].get_data()),axis=-1)
    
    return np.corrcoef(predictor_values,responses)[0,1]**2 

