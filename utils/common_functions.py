# import packages
import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time

from sklearn.preprocessing import RobustScaler
from scipy.spatial import distance_matrix


# import local functions
import utils.preprocessing as pr
import config as cfg


################################################################################
#
#   LOAD SUBJECT PARAMETERS
#
################################################################################


def sub_params(sub):
    """ Add subject info to P

    Get file/folder names for subject data and electrode info and add it to parameters P (see below).
    The variable P has to contain the field 'sub' : int 

    Parameters
    ----------
  
    Returns
    -------
    Paths:
    SP['RawPath'] : str, path to raw data folder
    SP['DerivativesPath'] : str, path to derivatives folder.
        Derivatives folder contains epochs, TFRepochs and in general any preprocessed data (i.e. not raw or source)

    Files:
    SP['RawFile'] : str, *raw.fif file in RawPath
    SP['MetadataFile'] : str, metadata file (csv or tsv DataFrame)
    SP['ChInfoFile'] : str, channel info file (json)

    Notes
    -----
    The function automatically creates derivatives paths if they do not exist. For example, the first time running the pipeline.
    """

    # alocate subject parameters dict
    SP = {}

    SP['sub']=sub

    # paths-----------------------------------------------------------------------------------
    SP['RawPath'] = check_path(['..','Data', 'raw',  sub])
    SP['SourcePath'] = os.path.join('..','Data', 'source',  sub, 'ieeg')
    SP['DerivativesPath'] = check_path(['..','Data', 'derivatives' + cfg.DerivativesTag ,  sub])
    SP['FiguresPath'] = check_path(['..','Figures'])

    # files --------------------------------------------------------------------------------
    #SP['source_files'] = sorted(glob.glob(SP['SourcePath'] + '/*.ns3'))
    SP['RawFile'] = glob.glob(SP['RawPath'] + '/*raw.fif')[0]
    SP['MetadataFile'] = glob.glob(SP['RawPath'] +'/*metadata.*')[0]
    SP['ChInfoFile'] = os.path.join(SP['DerivativesPath'],f"{sub}_info.json")

    # SOA ------------------------------------------------------------------------------------
    df = pd.read_csv(os.path.join(SP['RawPath'],sub+'_metadata.csv'))
    ttl = df[df['eventtype']=='wordonset']['time']
    SP['soa'] = np.median(np.diff(ttl))

    # channel info ---------------------------------------------------------------------------

    #print(SP['ChInfoFile'])
    if os.path.isfile(SP['ChInfoFile']): SP['ChInfo'] = json.load(open(SP['ChInfoFile']))
 
    return SP



################################################################################
#
#   NAVIGATION, FILE OPERATIONS AND MESSAGING
#
################################################################################


def check_path(folders):
    """ check recursively if folders exist, if they doesnT then create them

    Parameters
    ----------
    folders : list of str or str   
        each element of the list is a folder. The last str is the deeper level.
        if str it is transformed into a one element list

    Returns
    -------
    path : str
        concatenated path

    Example usage
    -------------
    file_name = os.path.join( check_path(['my_folder','my_deeper_folder']) ,'my_file.txt')
    if ./my_folder/ does not exist it is created
    if ./my_folder/my_deeper_folder/ does not exist it is created
    """
    
    # transform floders into one element list if it is a string
    if type(folders) == str: folders = [folders]

    path = '.'
    for folder in folders:
        path = os.path.join(path,folder)
        if not os.path.isdir(path): os.mkdir(path)
    return path






def display_progress(text, counter, total, time_start):
    """ print progress on loops, elapsed time and expected remaining time

    Parameters
    ----------
    text : str    
        string to print at the begining of the message
    couter : int
        iteration of the loop
    total :  int
        length of the iterator
    time_start : int (time)
        time when the loop started

    """

    elapsed_time = int((time.time() - time_start)/60.)
    remaining_time = int(((time.time() - time_start)/(max(counter,1)))*(total-counter)/60.)
    #message = f"{text} ({counter+1}/{total}) e.t. {elapsed_time} min"
    #if counter > 0 : message += f", r.t. {remaining_time} min" 
    message = f"{text} ({counter+1}/{total})"
    if counter > 0 : message += f", -{remaining_time} min" 

    print_d(message)     



def print_d(text):
    ''' Print dynamic
        this function uses stdout and flush to print over the same line
        and pads with empty spaces to erase previous text
    '''

    # pad text with spaces up to column 80 
    text += max(0,80-len(text))*' '

    # write and flush
    sys.stdout.write('\r' + text)
    sys.stdout.flush()






def concatenate_pdfs(FiguresPath,label,OutputName,remove=False):
    '''Use pdfunite to concatenate pdf figures into a single file

    Parameters
    ----------
    FiguresPath : str
        path to the folder containing the figures to concatenate
    label : str
        string present in the names of the all pdf files to concatenate
    OutputName : str
        name of the concatenated file
    remove : bool
        if True remove individual figures
    '''
            
    print_d("merging figures")

    # get current working directory to come back later
    OriginalPath = os.getcwd()    
    # move to the directory with the figures
    os.chdir(FiguresPath)

    # get list of files
    files = sorted(glob.glob('*' + label + '*pdf'))

    # make and launch command
    command = "pdfunite "
    for f in files: command += f + ' '
    command += OutputName

    os.system(command)
    # remove individual figures
    if remove: os.system("rm *" + label + "*.pdf")

    # go back to original path and carry on
    os.chdir(OriginalPath)




################################################################################
#
#   LOAD DATA (Epochs, test results, events, etc)
#
################################################################################



def load_epochs(sub, band, EpochsSetup, picks='good', InputFormat = cfg.InputFormat):
    ''' Load raw (or epochs), smooth, clip, make epochs (if raw), apply baseline
        power bands are transformed to dB

    Parameters
    ----------
    sub : str
    band : dict
    EpochsSetup : dict
    picks : str or list of str
    InputFormat : str, 'raw' or 'epochs' 

    Returns
    -------
    epochs : mne epochs

    '''
    SP = sub_params(sub)

    # --- RAW INPUT ----------------------------------------------------------
    if InputFormat == 'raw':
        raw_file = os.path.join(SP['DerivativesPath'], f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(raw_file, preload = True)
        raw = raw.pick_channels(get_picks(SP, picks, band = band))

        # some operations do not apply to complex bands (used for phase)
        if band['method'] != 'complex':
            # clip and smooth
            raw.apply_function(clip_n_rescale, picks = 'all', channel_wise = True, c = EpochsSetup['clip'])
            raw.apply_function(gaussian_convolution, picks = 'all', channel_wise = True, sigma=EpochsSetup['smooth'])

        # transform to dB
        if band['method'] in ['wavelet','hilbert']:
            raw.apply_function(np.log10, picks = 'all')

        # high pass filter baseline applied here
        if EpochsSetup['baseline'] == 'hpf' and band['method']!='complex': 
            raw.filter(cfg.hpf, None, phase = 'zero', picks='all')

        # make epochs
        if SP['MetadataFile'][-3:] == 'csv' : delimiter = ','
        elif SP['MetadataFile'][-3:] == 'tsv' : delimiter = '\t'
        metadata = pd.read_csv(SP['MetadataFile'], delimiter = delimiter)
        epochs = make_epochs(raw, metadata, EpochsSetup)

        # trialwise baseline applied here
        if isinstance(EpochsSetup['baseline'], dict) and band['method'] != 'complex' : 
            # make epochs that will be used as baseline
            BaseEpochs = make_epochs(raw, metadata, EpochsSetup['baseline'])
            epochs = trialwise_baseline(epochs, BaseEpochs)

    # --- EPOCHS INPUT ----------------------------------------------------------
    elif InputFormat == 'epochs':
        # load epochs file here
        EpochsFile = os.path.join(SP['DerivativesPath'],f"{sub}_band-{band['name']}_epochs-{EpochsSetup['name']}-epo.fif")
        epochs = mne.read_epochs(EpochsFile)
        epochs.crop(EpochsSetup['tmin'], EpochsSetup['tmax'])                    

    # filter epochs of interest
    if 'sample' in EpochsSetup.keys(): epochs = epochs[EpochsSetup['sample']]

    return epochs





def make_epochs(raw, metadata, EpochsSetup):
    ''' make epochs from mne raw
    
    Parameters
    ----------
    raw : mne raw
    metadata :  pd DataFrame
    EpochsSetup : dict

    Returns
    -------
    epochs : mne epochs

    '''

    # filter metadata according to epoch_setup
    EpochsMetadata = metadata[metadata[EpochsSetup['key']].isin(EpochsSetup['values'])]

    # make events array (MNE formatting [time sample, duration, label])
    time = [int(cfg.srate*t) for t in EpochsMetadata['time']]
    duration = [0]*len(EpochsMetadata['time'])
    label = np.ones_like(time).astype(int)
    events = np.array([time, duration, label]).T
    # make epochs
    epochs = mne.Epochs(raw, events, tmin=EpochsSetup['tmin'], tmax=EpochsSetup['tmax'], picks='all', metadata = EpochsMetadata, preload=True, baseline=None,reject_by_annotation=False, verbose=False)
       
    return epochs




def trialwise_baseline(epochs, BaseEpochs):
    ''' apply trialwise baseline, 
        for example baseline every event in a trial wrt fixation period of the given trial

    '''
    # times of epochs and baseline epochs
    tBase = BaseEpochs.metadata['time'].values
    tEpochs = epochs.metadata['time'].values

    # for each epoch find closest previous event in baseline epochs
    b_idx = [np.argmin(t-tBase[t_base<=t]) for t in tEpochs]
    
    # array of baseline means                  
    m = BaseEpochs.get_data().mean(axis=-1)[b_idx,:,np.newaxis]
    #      new axis to be able to broadcast to epochs  ^
    data = epochs.get_data() - m

    # make epochs with baselined data
    epochs = mne.EpochsArray(data, epochs.info, events = epochs.events, metadata = epochs.metadata, tmin = epochs.tmin)

    return epochs


'''
def load_df(file_tag,subs=cfg.subs,bands=cfg.bands):
    """ colect data from csv files across subjects 

    files should be named
    {sub}_{file_tag}.csv

    Parameters
    ----------
    file_tag : str
        
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]

    fdr_correction : bool
        perform fdr correctionand add a field with corrected p values
 
    Returns
    -------
    df : pandas dataframe
    """

    frames = []
    
    # iterate over subjects ------------------------------------------------
    for sub in subs:
        # load subject params
        SP = sub_params(sub)
        for band in bands:
            # load dataframe
            try:
                filename = os.path.join(SP['DerivativesPath'], f"{sub}_band-{band['name']}_{file_tag}.csv")
                frames += [pd.read_csv(filename)]  #'coords': eval, 
            except:
                filename = os.path.join(SP['DerivativesPath'], f"{sub}_band-{band['name']}_{file_tag.replace('_','-')}.csv")
                frames += [pd.read_csv(filename)]  #'coords': eval, 
    df = pd.concat(frames)

    df['coords'] =[c for c in df[['x','y','z']].values]

    return df
'''




def get_picks(SP, picks = 'good', band = None):
    """ Shortcut to select groups of channels

    picks can be:
    - list of channel names (then nothing is done)
    - 'good' / 'bad' from rejection
    - 'allROIs' (pick only ROIs)
    - 'notROI' 
    - 'EOI-xxxxx' read from EOIs file
    - [ 'ROI-xxxx'] single ROI, return channels inside ROI

    Returns
    -------
    picks : list of channel names
    """

    ChNames = SP['ChInfo']['ChNames']

    if isinstance(picks,list): PicksList = picks
    elif picks == 'good' : PicksList = [ch for ch in ChNames if ch not in SP['ChInfo']['bad']]
    elif picks == 'bad' : PicksList = SP['ChInfo']['bad']
    elif picks == None or picks == 'all': PicksList = ChNames       
    #elif picks == 'allROIs': PicksList = [ch for ch in ChNames if ch[:3] =='ROI']
    #elif picks == 'notROI': PicksList = [ch for ch in ChNames if ch[:3] !='ROI']
    elif 'EOI' in picks: PicksList = SP['ChInfo'][f"{picks}-{band['name']}"]
    elif Picks[:3]=='not': 
        notPicksList = SP['ChInfo'][f"{picks[3:]}-{band['name']}"]
        PicksList = [ch for ch in raw.ch_names if ch not in notPicksList]
    else: sys.exit('Picks not recognized :(')

    return PicksList



def get_coords(SP,picks='good'):
    """ Extract array of ch coordinates either from subject parameters dict or 
        from montage of mne.raw
        it also works for epochs objects instead of raw
    
    Parameters
    ----------
    X : either subject info dict or mne raw / epochs
    PicksList : list of str (with channel names)

    Returns
    -------
    coords : np.array (n_channels,3)

    """

    PicksList = get_picks(SP,picks)
    coords = np.array([SP['ChInfo']['coords'][SP['ChInfo']['ChNames'].index(ChName)] for ChName in PicksList])

    return coords


'''
def get_coords_from_raw(raw):
    """ Extract array of ch coordinates from montage of mne.raw
        it also works for epochs objects instead of raw
    
    Parameters
    ----------
    raw : mne raw / epochs

    Returns
    -------
    coords : np.array (n_channels,3)
    """

    PositionsDict = X.get_montage().get_positions()['ch_pos']
    coords = np.array([PositionsDict[ChName] for ChName in PicksList])

    return coords
'''





def get_ROI_labels(coords,tol=cfg.ROITolerance):
    """Argument:
    - coords: (N, [x, y, z])

    Returns:
    - ROIs (N)
    """

    # load atlas
    from nilearn import datasets, image

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)

    labels = atlas['labels']
    atlas_map = image.load_img(atlas['maps'])

    affine = atlas_map.affine[:3, :3]
    translation = atlas_map.affine[:3, 3]
    ROIs = []

    for coord in coords:
        data_coords = np.matmul(np.linalg.inv(affine), np.array(coord) - translation)
        a,b,c = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords).astype(int)
        index = atlas_map.get_data()[a, b, c]
        roi = ''.join([ s[0] for s in labels[index].split() if s[0].isalnum()]) # label acronyms
        ROIs.append(roi)

    ROIs = np.array(ROIs)
    if tol>0:
        j = np.where(ROIs!='B')[0]
        distances = distance_matrix(coords,coords)
        np.fill_diagonal(distances,distances.max())
        mfnd = np.median(distances.min(axis=-1))   # median first neightbor distance
        for i in np.where(ROIs=='B')[0]:
            fn = distances[i,j].argmin()           # first neighbor in known region (idexed wrt j)
            if distances[i,j[fn]]<tol*mfnd: ROIs[i] = ROIs[j[fn]] 


    ROIs = [cfg.replacements.get(x, x) for x in ROIs]

    return ROIs


def clip_n_rescale(x, c = cfg.clip, zscore = cfg.zscore):
    ''' Remove outliers using quantile range and zscore

    Parameters
    ----------
    x : np.array
        array to rescale
    qr : (float,float)
        lower and upper quantile for scaling
    c : float 
        parameter for clipping, if c==0 no clipping
    zscore : bool
        switch on and off zscoring
        
    Returns
    -------
    x : np.array
        rescaled array

    '''
    if c > 0:
        thresholds =np.squeeze(RobustScaler().fit(x.reshape(-1, 1)).inverse_transform([[-c],[c]]))
        x = np.clip(x,thresholds[0],thresholds[1])

    if zscore:
        x = (x - x.mean())/x.std()

    return x



def gaussian_convolution(x,sigma=cfg.smooth):
    ''' convolve a time series with a gaussian filter

    Parameters
    ----------
    x : np.array
        array to rescale
    sigma : float 
        std of the filter in seconds
        
    Returns
    -------
    x : np.array
        convolved array

    '''

    if sigma>0: 
        # transforme std from seconds to samples
        sigma *= cfg.srate
        # gausian pdf
        gx = np.arange(-3*sigma, 3*sigma)
        g = np.exp(-(gx/sigma)**2/2)
        # normalize
        g /= np.sum(g)

        x = np.convolve(x,g,mode="same")

    return x





"""

    _____\    _______
   /      \  |      /\
  /_______/  |_____/  \
 |   \   /        /   /
  \   \         \/   /
   \  /          \__/_
    \/ ____    /\
      /  \    /  \
     /\   \  /   /
       \   \/   /
        \___\__/




def fdr_correction(df,key,condition=None):
    ''' add a column with fdr corrected pvalues to a data frame 

    Parameters
    ----------
    df : pandas dataframe
    key : str
        name of the column on which to perform frd correction
    condition : str or None
        alows to prefilter p values for negative regression scores
        condition = 'r2' filters out p values with r2 < 0

    Returns
    -------
    df : pandas dataframe
        with  one column named key_fdr added
    '''


    if isinstance(condition,str):
        corrected = np.ones(len(df))
        #print(df[df[condition]>0][key])
        #print(mne.stats.fdr_correction(df[df[condition]>0][key],cfg.alpha)[1])
        corrected[df[condition]>0] = mne.stats.fdr_correction(df[df[condition]>0][key],cfg.alpha)[1]
        df[key + '_fdr'] = corrected
    else:
        #df[key + '_fdr'] = mne.stats.fdr_correction(df[key],cfg.alpha)[1]

        p_fdr = np.ones(len(df))

        p_fdr[df[key]<0.9] = mne.stats.fdr_correction(df[df[key]<0.9][key],cfg.alpha)[1]

        df['p_fdr'] = p_fdr

    return df




def get_filename_from_dict(d, sep='_', keys_to_use=[], extension=''):
    '''
    This function generates a filename that contains chosen keys-values pairs from a dictionary.
    For example, the dict can represent hyperparameters of a model or settings.

    USAGE EXAMPLE:
    filename = get_filename_from_dict(my_dict, '-', ['frequency_band', 'smoothing'])

    :param d: (dict) keys and corresponding values to be used in generating the filename.
    :param sep: (str) separator to use in filename, between keys and values of d.
    :param keys_to_use: (list) subset of keys of d. If empty then all keys in d will be appear in filename.
    :param extension: (str) extension of the file name (e.g., 'csv', 'png').
    :return: (str) the filename generated from key-value pairs of d.
    '''

    # Check if dictionary is empry
    if not d:
        raise('Dictionary is empty')
    # filter the dictionary based on keys_to_use
    if keys_to_use:
        for i, k in enumerate(keys_to_use):
            if k not in d.keys():
                warnings.warn('Chosen key (%s) is not in dict' % k)
                keys_to_use.pop(i)
    else:
        keys_to_use = sorted(d.keys())

    # add extension
    if len(extension) > 0:
        extension = '.' + extension
    return sep.join([sep.join((str(k), str(d[k]))) for k in keys_to_use]) + extension


"""

