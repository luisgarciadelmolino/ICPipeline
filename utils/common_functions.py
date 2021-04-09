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
    dp['raw_path'] : str
        path to raw data folder
    dp['derivatives_path'] : str
        path to derivatives folder.
        Derivatives folder contains epochs, TFRepochs and in general any preprocessed data (i.e. not raw or source)

    Files:
    dp['raw_files'] : list of str
        names of ieeg.csv / raw.fif files in raw folder
    dp['event_files'] : list of str
         names of metadata files (they must match the number of raw_files)

    Notes
    -----
    The function automatically creates derivatives paths if they do not exist. For example, the first time running the pipeline.

    """

    # alocate dynamic parameters dict
    dp = {}

    dp['sub']=sub

    # paths-----------------------------------------------------------------------------------
    dp['raw_path'] = check_path(['..','Data', 'raw',  sub])
    dp['source_path'] = os.path.join('..','Data', 'source',  sub, 'ieeg')

    dp['derivatives_path'] = check_path(['..','Data', 'derivatives' + cfg.derivatives_tag ,  sub])
    dp['figures_path'] = check_path(['..','Figures'])

    # files --------------------------------------------------------------------------------
    #dp['source_files'] = sorted(glob.glob(dp['source_path'] + '/*.ns3'))
    #dp['raw_file'] = glob.glob(dp['raw_path'] + '/*ieeg.csv')[0]
    dp['metadata_file'] = glob.glob(dp['raw_path'] +'/*metadata.*')[0]

    # SOA ------------------------------------------------------------------------------------
    df = pd.read_csv(os.path.join(dp['raw_path'],sub+'_metadata.csv'))
    ttl = df[df['eventtype']=='wordonset']['time']
    dp['soa'] = np.median(np.diff(ttl))

    # electrodes info ----------------------------------------------------------------------------
    electrodes_file = glob.glob(os.path.join(dp['raw_path'],  sub + '_electrodes.*'))

    try:
        # electrode file is a json       
        electrodes =  json.load(open(electrodes_file[0]))
        ch_names = electrodes['names'] 
        num_channels = len(ch_names)
        dp['coords'] = electrodes['coords']
        dp['probe_names'] = [''.join(filter(lambda x: not x.isdigit(), ch_name)) for ch_name in ch_names]
        dp['contact_numbers'] = [''.join(filter(lambda x: x.isdigit(), ch_name)) for ch_name in ch_names]
        dp['ch_names'] = [str(i+1).zfill(3) + '-' + dp['probe_names'][i] + dp['contact_numbers'][i] for i in range(num_channels)]
        if 'gyri' in electrodes.keys(): dp['gyri'] =  electrodes['gyri']
        # often last channel is ttl, we want it OUT
        if 'MARKER' in ch_names:
            del  dp['coords'][ch_names.index('MARKER')]
            del  dp['ch_names'][ch_names.index('MARKER')]
            
    except:
        print(f"electrodes not found")
        dp['ch_names'] = []
        dp['coords'] = []

    if not (len(dp['ch_names'])==len(dp['coords'])) : sys.exit('Error: inconsistent number of electrodes / coordinates\n')

    return dp



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
    message = f"{text} ({counter+1}/{total}) e.t. {elapsed_time} min"
    if counter > 0 : message += f", r.t. {remaining_time} min" 
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






def concatenate_pdfs(figures_path,label,output_name,remove=False):
    '''Use pdfunite to concatenate pdf figures into a single file

    Parameters
    ----------
    figures_path : str
        path to the folder containing the figures to concatenate
    label : str
        string present in the names of the all pdf files to concatenate
    output_name : str
        name of the concatenated file
    remove : bool
        if True remove individual figures
    '''
            
    print_d("merging figures")

    # get current working directory to come back later
    original_path = os.getcwd()    
    # move to the directory with the figures
    os.chdir(figures_path)

    # get list of files
    files = sorted(glob.glob('*' + label + '*pdf'))

    # make and launch command
    command = "pdfunite "
    for f in files: command += f + ' '
    command += output_name

    os.system(command)
    # remove individual figures
    if remove: os.system("rm *" + label + "*.pdf")

    # go back to original path and carry on
    os.chdir(original_path)








################################################################################
#
#   LOAD DATA (Epochs, test results, events, etc)
#
################################################################################



def load_epochs(sub,band,ep_setup,picks=None, input_format=cfg.input_format):


    dp = sub_params(sub)

    # --- RAW INPUT ----------------------------------------------------------
    if input_format == 'raw':
        raw_file = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(raw_file, preload=True)

        picks = get_picks(raw,picks,sub=sub,band=band)
        raw = raw.pick_channels(picks)

        # some operations do not apply to complex bands (used for phase)
        if band['method']!='complex':
            # clip and smooth
            raw.apply_function(clip_n_rescale,picks='all',channel_wise=True,c=ep_setup['clip'])
            raw.apply_function(gaussian_convolution,picks='all',channel_wise=True, sigma=ep_setup['smooth'])

        # transform to dB
        if band['method'] in ['wavelet','hilbert']:
            raw.apply_function(np.log,picks='all')

        # make epochs
        if dp['metadata_file'][-3:]=='csv': delimiter = ','
        elif dp['metadata_file'][-3:]=='tsv': delimiter = '\t'
        metadata = pd.read_csv(dp['metadata_file'], delimiter=delimiter)
        epochs = make_epochs(raw,metadata,ep_setup)

        # trialwise baseline applied here
        if isinstance(ep_setup['baseline'],dict) and band['method']!='complex': 
            base_epochs = make_epochs(raw,metadata,ep_setup['baseline'])
            epochs = trialwise_baseline(epochs,base_epochs,band['method'])

    # --- EPOCHS INPUT ----------------------------------------------------------
    elif input_format == 'epochs':
        # load epochs file here
        epochs_file = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_epochs-{ep_setup['name']}-epo.fif")
        epochs = mne.read_epochs(epochs_file)
        epochs.crop(ep_setup['tmin'],ep_setup['tmax'])                    

    if 'query' in ep_setup.keys(): epochs = epochs[ep_setup['query']]

    return epochs





def make_epochs(raw,metadata,ep_setup):

    # filter metadata according to epoch_setup
    epo_metadata = metadata[metadata[ep_setup['key']].isin(ep_setup['values'])]

    # make events array (MNE formatting [time sample, duration, label])
    time = [int(cfg.srate*t) for t in epo_metadata['time']]
    duration = [0]*len(epo_metadata['time'])
    label = np.ones_like(time).astype(int)
    events = np.array([time, duration, label]).T
    # make epochs
    epochs = mne.Epochs(raw, events, tmin=ep_setup['tmin'], tmax=ep_setup['tmax'], picks='all', metadata = epo_metadata, preload=True, baseline=None,reject_by_annotation=False, verbose=False)
       
    return epochs


def trialwise_baseline(epochs,base_epochs,method):

    # times of epochs and baseline epochs
    t_base = base_epochs.metadata['time'].values
    t_epochs = epochs.metadata['time'].values

    # for each epoch find closest previous event in baseline epochs
    b_idx = [np.argmin(t-t_base[t_base<=t]) for t in t_epochs]
    

    # array of baseline means                  new axis to be able to broadcast to epochs
    m = base_epochs.get_data().mean(axis=-1)[b_idx,:,np.newaxis]

    # find if data is filtered or amplitude and baseline
    data = epochs.get_data()
    if method in ['complex']: pass
    else: data -= m

    # make epochs with baselined data
    epochs = mne.EpochsArray(data, epochs.info, events=epochs.events, metadata=epochs.metadata,tmin=epochs.tmin)

    return epochs



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
        dp = sub_params(sub)
        for band in bands:
            # load dataframe
            try:
                filename = os.path.join(dp['derivatives_path'], f"{sub}_band-{band['name']}_{file_tag}.csv")
                frames += [pd.read_csv(filename)]  #'coords': eval, 
            except:
                filename = os.path.join(dp['derivatives_path'], f"{sub}_band-{band['name']}_{file_tag.replace('_','-')}.csv")
                frames += [pd.read_csv(filename)]  #'coords': eval, 
    df = pd.concat(frames)

    df['coords'] =[c for c in df[['x','y','z']].values]

    return df


def get_picks(raw,picks,sub=None,band=None):
    """ Shortcut to select groups of channels

    picks can be:
    1 : list of channel names
    2 : 'allROIs' (pick only ROIs)
    3 : 'notROI' 
    4 : 'EOI-xxxxx' read from EOIs file
    5 : [ 'ROI-xxxx'] single ROI, return channels inside ROI

    Returns
    -------
    picks : list of channel names
    """

    if isinstance(picks,list): pass     
    elif picks == None or picks == 'all': picks= raw.ch_names       
    elif picks == 'allROIs': picks = [ch for ch in raw.ch_names if ch[:3] =='ROI']
    elif picks == 'notROI': picks = [ch for ch in raw.ch_names if ch[:3] !='ROI']
    elif picks[:3]=='EOI': 
        dp = sub_params(sub)
        EOI_file = os.path.join(dp['derivatives_path'],f"{sub}_EOIs.json")
        picks = json.load(open(EOI_file))[f"{picks}-{band['name']}"]
    elif picks[:3]=='ROI':
        p = [ch for ch in raw.ch_names if ch[:3]!='ROI']
        coords = get_coords(raw, picks=p)
        labels = get_ROI_labels(coords)
        picks = [p[j] for j, l in enumerate(labels) if l==picks[0][4:] ]
    else: sys.exit('Picks not recognized :(')

    return picks





def get_coords(raw,picks):
    """ Extract array of ch coordinates from montage
        it also works for epochs objects instead of raw
    """
    positions_dict = raw.get_montage().get_positions()['ch_pos']

    return np.array([positions_dict[ch_name] for ch_name in picks])


def get_ROI_labels(coords,tol=cfg.ROI_tolerance):
    """Argument:
    - coords: (N, x, y, z)

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
        #x = np.squeeze(RobustScaler().fit_transform(x.reshape(-1, 1)))
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
        #gx = np.arange(-3*sigma, 3*sigma)
        #g = np.exp(-(gx/sigma)**2/2)

        # uniform window
        g = np.ones(int(sigma))

        # normalize
        g /= np.sum(g)



        #print(x.shape)
        x = np.convolve(x,g,mode="same")
        #print(x.shape)

    return x


def fdr_correction(df,key,condition=None):
    """ add a column with fdr corrected pvalues to a data frame 

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
    """


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



def filter_df(df,file_tag,key,alpha=cfg.alpha, subs=cfg.subs,bands=cfg.bands):
    """ Filter one df with respect to values of column 'key' of another df

    Parameters
    ----------
    df : pandas dataframe
        one of the columns has to 
    file_tag : str
        tag to identify the file with the condition to filter
    key : str
        name of the column on which to perform frd correction

    Returns
    -------
    df : pandas dataframe
        filtered df
    """

    df_filter = load_df(file_tag)
    df = df.filter(df_filter[key]<alpha,axis=index)


    return df





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

