# import packages
import numpy as np
import pandas as pd
import scipy as sc
import sys, os, glob, csv, json, mne, time
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import kurtosis, skew, linregress
from sklearn.preprocessing import RobustScaler
from scipy.signal import periodogram

# import local functions
import config as cfg
import utils.common_functions as cf
import utils.plots as pl



##############################################################################
#
#   SUBJECT INFO DICTIONARY
#
##############################################################################


def ch_info(subs=cfg.subs):

    print("\n======== Subject info =============================================\n")



    for sub in subs:
        SP = cf.sub_params(sub)

        # load raw to extract ch info
        raw = mne.io.read_raw_fif(SP['RawFile']) 

        ChInfo={}
        ChInfo['ChNames'] = raw.ch_names
        PositionsDict = raw.get_montage().get_positions()['ch_pos']
        ChInfo['coords'] =[PositionsDict[ChName].tolist() for ChName in raw.ch_names]
        #'ROIalbels' : cf.get_ROI_labels(raw.get_montage().get_positions()['ch_pos'])

        json.dump(ChInfo, open(SP['ChInfoFile'],'w'))



##############################################################################
#
#   REJECTION
#
##############################################################################

def rejection_wrap(subs=cfg.subs, figures=True):
    """ Identify bad channels early in the processing and discard them
        completely from raw files.

        Rejection based on std of V and dVdt
    """

    print("\n======== rejection ================================================\n")
    print(f"SUB\tTOTAL\tKEPT\tREJECTED")
    for iSub, sub in enumerate(subs): 
       
        SP = cf.sub_params(sub)

        raw = mne.io.read_raw_fif(SP['RawFile'],preload=True)
        kept, stats, thresholds = rejection(raw)

        if figures: rejection_figure(SP,kept,stats,thresholds)

        # mark bad channels in ChInfo and save
        SP['ChInfo']['bad'] = [ch for iCh, ch in enumerate(raw.ch_names) if ~kept[iCh]]
        json.dump(SP['ChInfo'], open(SP['ChInfoFile'],'w'))
    
        print(f"{SP['sub'][4:]}\t{len(kept)}\t{np.sum(kept)}\t{np.sum(~kept)}\t{int(100*np.sum(~kept)/len(kept))}%")



def rejection(raw):
    """  Channel rejection based on std of voltage and voltage increments

    Parameters
    ----------
    raw : mne raw

    Returns
    -------
    raw :  mne raw
        (with 'bads' updated)
    kept : list of bool
        good channels
    (s_v,s_vd) : np.array dim (2,n_ch)
        std of voltages and voltage increments
    """

    data = raw.get_data()

    # std of voltage and voltage increments
    stdv = np.std(data,axis=-1)
    stddv = np.std(np.diff(data,axis=-1),axis=-1)
    slopes = fit_power_law(raw)

    # use robust scaler to get relative values for stdv and stddv
    x = np.squeeze(RobustScaler().fit_transform(stdv.reshape(-1, 1)))
    y = np.squeeze(RobustScaler().fit_transform(stddv.reshape(-1, 1)))

    # get thresholds in original scale
    thresholds = [
         np.squeeze(RobustScaler().fit(stdv.reshape(-1, 1)).inverse_transform([[cfg.vlow],[cfg.vhigh]])),
         np.squeeze(RobustScaler().fit(stddv.reshape(-1, 1)).inverse_transform([[cfg.dvlow],[cfg.dvhigh]])),
         [cfg.slow,cfg.shigh]
                  ]

    # identify good channels (those within the thresholds)
    kept = (x>cfg.vlow)*(x<cfg.vhigh)*(y>cfg.dvlow)*(y<cfg.dvhigh)*(slopes>cfg.slow)*(slopes<cfg.shigh)
    
    return kept, np.array([stdv,stddv,slopes]), thresholds


def fit_power_law(raw):
    """ Estimate exponential slope of power spectral density
        between 10 and 200Hz

    Parameters
    ----------
    raw : mne raw

    Returns
    -------
    slopes : np.array (n_channels)
        
    """

    psds, f = mne.time_frequency.psd_welch(raw,fmin=10,fmax=200,n_fft=512)

    slopes = np.array([])
    for i in range(len(raw.ch_names)):
        if np.sum(psds[i]>0) == 0:      # for channels with 0 var
            slopes = np.append(slopes,0)
        else:
            res = linregress(np.log(f),np.log(psds[i]))
            slopes = np.append(slopes,res.slope*0.5)
        
    return slopes


def rejection_figure(SP,kept,stats,thresholds,save=True,plot=False):
    """ Make rejection summary figure for single subject
        
    """

    #create figure
    fig, ax = plt.subplots(2,1,figsize=(7,7))
    fig.suptitle(f"{SP['sub']}, rejected {np.sum(~kept)}/{len(kept)}, {int(100*np.sum(~kept)/len(kept))}%",fontsize = 15)

    # rejected channel positions
    pl.brain_plot(ax[0],SP['ChInfo']['coords'],1.*(~kept),mask=kept,mode='symmetric',interval=[0,1])

    # scatter with rejection summary
    pl.scatter_plot(ax[1],stats[0],stats[1],stats[2],colorbar='s',mask=kept,
                    ylabel='std(dVdt)',xlabel='std(V)', vlines = thresholds[0],hlines=thresholds[1])

    # save figure
    FigName = os.path.join(cf.check_path([SP['FiguresPath'],'rejection']),f"{SP['sub']}.pdf")
    if save: fig.savefig(FigName, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()

    

def rejection_epochs(subs=cfg.subs,EpochsSetups=cfg.EpochsSetups):
    """ Identify bad channels early in the processing and discard them
        completely from epochs files.

        Rejection based on std of V and dVdt
    """

    print("\n======== rejection ==============================================\n")
    for iSub, sub in enumerate(subs): 
       
        for iEp, EpochsSetup in enumerate(EpochsSetups):
            SP = cf.sub_params(sub)
            # load epochs
            EpochsFile = os.path.join(SP['DerivativesPath'],f"{sub}_epochs-{EpochsSetup['name']}-epo.fif")
            epochs = mne.read_epochs(EpochsFile)

            data = np.swapaxes(epochs.get_data(),0,1).reshape(len(epochs.ch_names),-1)

            # std of voltage and voltage increments
            s_v = np.std(data,axis=-1)
            s_dv = np.std(np.diff(data,axis=-1),axis=-1)

            # use robust scaler to get relative values
            x = np.squeeze(RobustScaler().fit_transform(s_v.reshape(-1, 1)))
            y = np.squeeze(RobustScaler().fit_transform(s_dv.reshape(-1, 1)))

            # identify good channels (those within the thresholds)
            kept = (x>cfg.V_low)*(x<cfg.V_high)*(y>cfg.dV_low)*(y<cfg.dV_high)

            print(f"CH\tKEPT\tstd(V)\tstd(dVdt)")
            for iCh, k in enumerate(kept): print(f"{iCh}\t{k}\t{np.around(s_v[iCh],decimals=2)}\t{np.around(s_dv[iCh],decimals=2)}")




##############################################################################
#
#   REFERENCING
#
##############################################################################


def referencing_wrap(subs=cfg.subs,RefMethod=cfg.RefMethod):

    print(f"\n======== Applying reference ======================================\n")

    # ITERATE SUBJECT LIST =============================================================================
    TimeStart = time.time()
    for iSub, sub in enumerate(subs): 
        cf.display_progress(f"{sub}",iSub,len(subs),TimeStart)

        # add paths to parameters
        SP = cf.sub_params(sub)

        # load raw
        raw = mne.io.read_raw_fif(SP['RawFile'], preload=True)        
        raw.drop_channels(SP['ChInfo']['bad'])   

        RefRaw, bads = referencing(raw,RefMethod)

        # update bad channel list and save referenced epochs
        SP['ChInfo']['bad'] += bads
        json.dump(SP['ChInfo'], open(SP['ChInfoFile'],'w'))
        FileName = os.path.join(SP['DerivativesPath'],f"{sub}_raw.fif")
        RefRaw.save(FileName, overwrite=True)

        # move to next subject ===============================================

    cf.print_d(f"{cfg.RefMethod} reference done!")
    print(f"\n")




def referencing(raw,RefMethod):
    """ Subtract reference, computed with different methods
        Discard sensors (eg extremes of probes in laplacian)

    Parameters
    ----------
    raw : mne raw
    RefMethod : str

    Returns
    -------
    RefRaw :  mne raw
        referenced, channels discarded if necessary
    """

    data = raw.get_data(picks='all')

    if RefMethod == 'cmr':
        reference_data = np.tile(np.median(data,axis=0),(len(raw.ch_names),1))
        bads = []

    elif RefMethod == 'car':
        reference_data = np.tile(np.mean(data,axis=0),(1,len(raw.ch_names)))
        bads = []

    elif RefMethod == 'laplacian':
        reference_data, bads = laplacian_ref(raw)       

    elif RefMethod == 'bipolar':
        reference_data, bads = bipolar_ref(data,raw.ch_names) 
        
    elif RefMethod == 'none' or referencing == None: 
        reference_data = 0
        bads = []
    else: sys.exit(f'\n{referencing} referencing not recognised  ¯\_(´･_･`)_/¯ ')
    
    # referenced data
    RefRaw = mne.io.RawArray(data-reference_data, raw.info.copy())

    return RefRaw, bads


'''
def laplacian_ref(data, ChNames):

    ChProbes =  [''.join(filter(lambda x: not x.isdigit(), ChName)) for ChName in ChNames]
    bads = []

    # create a dummy raw that contains the references
    reference_data = np.zeros_like(data)

    for iCh, ChName in enumerate(ChNames[:-1]):     
        reference_data[iCh] = np.mean([data[iCh-1], data[iCh+1]], 0)
        if ChProbes[iCh -1]!=ChProbes[iCh +1]: bads +=[ChName]

    return reference_data, bads
'''


def laplacian_ref(raw):

    data = raw.get_data()
    ChNames = raw.ch_names
    PositionsDict = raw.get_montage().get_positions()['ch_pos']
    coords = np.array([PositionsDict[ChName] for ChName in ChNames])

    # create an empty raw to store the references
    reference_data = np.zeros_like(data)

    # adjacency matrix
    distance = sc.spatial.distance_matrix(coords,coords)
    np.fill_diagonal(distance,distance.max())           
    fnmd = np.nanmedian(np.min(distance,axis=0))  # first neighbor median distance
    adjacency = distance < cfg.s_adj*fnmd

    bads = []
    for iCh, ChName in enumerate(ChNames[:-1]):     
        CloseNeighbors = adjacency[iCh] 
        if np.sum(CloseNeighbors) < 2 : bads += [ChName]
        else: reference_data[iCh] = np.mean(data[CloseNeighbors], axis = 0)

    return reference_data, bads


def bipolar_ref(data, ChNames):

    ChProbes = [''.join(filter(lambda x: not x.isdigit(), ChName)) for ChName in ChNames]
    bads = []

    # create a dummy raw that contains the references
    reference_data = np.zeros_like(data)

    for iCh, ChName in enumerate(ChNames[:-1]):     
        reference_data[iCh] = data[iCh+1]
        if ChProbes[iCh]!=ChProbes[iCh +1]: bads +=[ChName]


    return reference_data, bads
    
                 

################## TIME FREQ TOOLS ##########################################
#     .----.                                .---.
#    '---,  `.____________________________.'  _  `.
#         )   ____________________________   <_>  :
#    .---'  .'                            `.     .'
#     `----'                                `---'                                                 
#############################################################################




def raw2TFR_wrap(subs=cfg.subs, bands=cfg.bands,skip=cfg.skip):
    """ Loop over subjects and bands and make and save
        mne raw objects with band power.

    """

    print(f"\n======== mne.raw 2 mne.rawTFR ====================================\n")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    TimeStart = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((iSub, sub), (iBand, band)) in enumerate(iterator): 

        cf.display_progress(f"{sub}, {band['name']} power", i, len(iterator), TimeStart) 


        # load subject data and raw file only once per subject
        if iBand==0:
            # add paths to parameters
            SP = cf.sub_params(sub)
            # load raw
            fname = os.path.join(SP['DerivativesPath'],f"{sub}_raw.fif")
            raw = mne.io.read_raw_fif(fname, preload=True)

        # skip if file already exists
        fname = os.path.join(SP['DerivativesPath'],f"{sub}_band-{band['name']}_raw.fif")
        if os.path.isfile(fname) and skip: continue
            
        RawBand = raw2TFR(raw,band)

        # save
        RawBand.save(fname, picks='all', overwrite=True)

    cf.print_d(f"done! elapsed time {int((time.time() - TimeStart)/60.)} min")
    print(f"\n")




def epochs2TFR_wrap(subs = cfg.subs, bands = cfg.bands, EpochsSetups = cfg.EpochsSetups):
    """ Loop over subjects and bands and make and save
        mne epochs objects with band power.

    """


    print(f"\n======== mne.epochs 2 mne.epochsTFR ==============================\n")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    TimeStart = time.time()
    iterator =  list(product(enumerate(subs),enumerate(EpochsSetups),enumerate(bands)))
    for i, ((iSub, sub), (iEp, EpochsSetup), (iBand, band)) in enumerate(iterator): 

        cf.display_progress(f"{sub}, {band['name']} power", i, len(iterator),TimeStart) 

        # load subject data and raw file only once per subject
        if iBand==0:
            # add paths to parameters
            SP = cf.sub_params(sub)
            # load epochs
            EpochsFile = os.path.join(SP['DerivativesPath'],f"{sub}_epochs-{EpochsSetup['name']}-epo.fif")
            epochs = mne.read_epochs(EpochsFile)
            
        EpochsBand = raw2TFR(epochs,band)

        # save
        fname = os.path.join(SP['DerivativesPath'],f"{sub}_band-{band['name']}_epochs-{EpochsSetup['name']}-epo.fif")
        EpochsBand.save(fname, overwrite=True)


    cf.print_d(f"done! elapsed time {int((time.time() - TimeStart)/60.)} min")
    print(f"\n")





def raw2TFR(raw,band):
    """ make band power raw using different methods method

    Parameters
    ----------
    raw : mne raw
    band : dict
        dictionary with band info (including TFR method)

    Returns
    -------
    RawBand : mne raw 
    """

    # if fmax > fNyquist lower fmax and print warning
    if band['fmax']!= None and band['fmax'] >= cfg.srate/2:
        band['fmax'] = cfg.srate/2 - 1;
        print(f"fmax larger than Nyquist freq., reduced to {band['fmax']}")

    # for broad band files just save again
    if band['method']== None or band['method']== 'none': 
        RawBand = raw.copy()

    elif band['method']== 'hilbert':
        # apply filter
        RawBand = raw.copy().filter(band['fmin'],band['fmax'],phase='zero',picks='all')
        # compute envelope
        RawBand.apply_hilbert(picks='all', envelope=True)

    elif band['method']== 'complex':
        # apply filter
        RawBand = raw.copy().filter(band['fmin'],band['fmax'],phase='zero',picks='all')
        # get complex hilbert
        RawBand.apply_hilbert(picks='all', envelope=False)

    elif band['method']== 'filter':
        # apply filter
        RawBand = raw.copy().filter(band['fmin'],band['fmax'],phase='zero',picks='all')

    elif band['method']== 'wavelet':
        RawBand = wavelet(raw.copy(),band)
    else: sys.exit(f"\n{band['method']} method not recognised  ¯\_(´･_･`)_/¯ ")
    

    return RawBand





def wavelet(raw,band):
    """ extra function for wavelet method
       with compatibility for mne epochs object instead of raw

    Parameters
    ----------
    raw : mne raw (or mne epochs)
    band : dict
        dictionary with band info

    Returns
    -------
    RawBand : mne raw 
    """

    # check that all the necessary info is in the band
    if not set(['nf', 'norm']).issubset(set(band.keys())):
        sys.exit("Band parameters for wavelet method not found")

    # make linearly or logspaced list of freqs for morlets
    freqs = np.linspace(band['fmin'],band['fmax'],band['nf'])
  
    # compute wavelets   (extra "epoch" dimension in morlet argument if input is raw)
    if cfg.input_format == 'raw': data = np.array([raw.get_data()])
    if cfg.input_format == 'epochs': data = raw.get_data()
    power = mne.time_frequency.tfr_array_morlet(data,cfg.srate,freqs, n_cycles=freqs / 2., output = 'power')
    
    # normalize across freqs
    if band['norm'] == 'std': 
        # std acrross time AND epochs, add epochs and time axis to broadcast
        power = power/power.std(axis=(0,-1))[np.newaxis,:,:,np.newaxis]
    elif band['norm'] == 'freq': 
        # add epoch, channel and time axis
        power = power*freqs[np.newaxis,np.newaxis,:,np.newaxis]

    # average across frequency axis (-2) and make raw / epochs
    if cfg.input_format == 'raw': RawBand = mne.io.RawArray(power[0].mean(axis=-2), raw.info)
                #                                              ^ in raw there is only one long "epoch"
    if cfg.input_format == 'epochs': RawBand = mne.EpochsArray(power.mean(axis=-2), raw.info, events=raw.events, metadata=raw.metadata,tmin=raw.tmin)

    return RawBand
    




#############################################################################
#
#             ADD ROIs
#
#          _---~~~~-_.
#        _(        )   )
#      ,   ) -~~- ( ,-' )_
#     (  `-,_..`., )-- '_,)
#    ( ` _)  (  -~( -_ `,  }
#    (_-  _  ~_-~~~~`,  ,' )
#      `~ -^(    __;-,((()))
#           ~~~~ {_ -_(())
#                 `\  }
#                   { }
#
#
#############################################################################


def add_ROI_from_atlas(subs=cfg.subs, bands=cfg.bands):
    """ Add a virtual channels with the average of channels
        inside ROIs from atlas

    Parameters
    ----------
    ROI : list of str or int
        list of channels to include by name or index
        optionally ROI can be 'all'
    ROIname : str
        name of the virtual channel
    """

    print(f"\n======== Add ROIs from atlas =====================================\n")


    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    TimeStart=time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((iSub, sub), (iBand, band)) in enumerate(iterator): 

        # load subject paths 
        SP = cf.sub_params(sub)

        # load raw
        fname = os.path.join(SP['DerivativesPath'],f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(fname, preload=True)

        # get list non ROI channels only
        ChNames = [ChName for ChName in raw.ch_names if 'ROI' not in ChName]
        coords = cf.get_coords(SP,picks=ChNames)
        labels = cf.get_ROI_labels(coords)

        for label in list(set(labels)):
            ROI = [ChNames[j] for j, l in enumerate(labels) if l==label ]
            if len(ROI)>0: # avoid adding empoty sets
                # ROI center of mass
                ROIcoords = cf.get_coords(SP,picks=ROI).mean(axis=0)
                ROIname = f"{str(len(SP['ChInfo']['ChNames'])+1).zfill(3)}-ROI-{label}"
                # add virtual channel to raw
                raw = add_ROI(raw,ROI,ROIname,ROIcoords)       
                # update subject info
                SP['ChInfo']['ChNames'] += [ROIname]
                SP['ChInfo']['coords'] += [ROIcoords.tolist()]
                cf.display_progress(f"{raw.ch_names[-1]} ({len(ROI)} channels) added to {sub} {band['name']}",i,len(iterator),TimeStart)

        # save
        json.dump(SP['ChInfo'], open(SP['ChInfoFile'],'w'))
        raw.save(fname, picks='all', overwrite=True)
 


def add_ROI(raw, ROI, ROIname, ROIcoords):
    """ Add a virtual channel with the average of other channels

    Parameters
    ----------
    raw : mne raw object to append
    ROI : list of str or int
        list of channels to include by name or index
        optionally ROI can be 'all'
    ROIname : str
        name of the virtual channel
    """



    # pick only real channels, do not include other ROIs
    if ROI == 'all': ROI = [ChName for ChName in raw.ch_names if ChName[0].isdigit()]



    ROIdata = np.array([raw.copy().get_data(picks=ROI).mean(axis=0)])
    # Create MNE info
    ROIinfo = mne.create_info([ROIname], sfreq=cfg.srate, ch_types='eeg')
    # Create the Raw object
    ROIraw = mne.io.RawArray(ROIdata,ROIinfo)

    # mne abbreviates names, use mne version
    ROIname = ROIraw.ch_names[0]
    # remove ROI if it already exists 
    if ROIname in raw.ch_names: raw.drop_channels([ROIname])

    # create montage for ROI
    montage = mne.channels.make_dig_montage({ROIname:ROIcoords})
    ROIraw.set_montage(montage)

    return raw.add_channels([ROIraw],force_update_info=True)






def clear_ROIs(subs=cfg.subs, bands=cfg.bands):
    """ 
    """

    print(f"\n======== Clear ROIs  =============================================\n")


    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    TimeStart = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((iSub, sub), (iBand, band)) in enumerate(iterator): 

        # load subject paths 
        SP = cf.sub_params(sub)

        # load raw
        fname = os.path.join(SP['DerivativesPath'],f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(fname, preload=True)

        # get list non ROI channels only
        ChNames = [ChName for ChName in raw.ch_names if ChName[:3]=='ROI']
        raw.drop_channels(ChNames)
        raw.save(fname, picks='all', overwrite=True)
 
        cf.display_progress(f"{sub} {band['name']}, {len(ChNames)} ROIs removed",i,len(iterator),TimeStart)

 



##############################################################################
#
#   MAKE RAW FROM SOURCE
#
##############################################################################


def source2raw(subs=cfg.subs):

    from neo.io import BlackrockIO

    TimeStart = time.time()
    # ITERATE SUBJECT LIST =============================================================================
    for iSub, sub in enumerate(subs): 

        print(f"\n--- {sub} ----------------------------------------------------\n")
        # add paths to parameters
        SP = cf.sub_params(sub)
        cf.display_progress(f"Loading source data, {sub}",iSub,len(subs),TimeStart)
        ieeg = []
        ttl = []
        # 2 files per session
        for i, source_file in enumerate(SP['source_files']):
            reader = BlackrockIO(filename=source_file)
            blks = reader.read(lazy=False)

            ttl_ =np.squeeze(1.*(np.diff(1.*(np.array(blks[0].segments[-1].analogsignals[0]).T[0,:] > 4000.))>0)).astype(np.int8)
            ieeg_ = np.array(blks[0].segments[-1].analogsignals[1],dtype=np.int16).T


            # start five seconds before first fixation
            t0 = np.argmax(ttl_>0.5)- int(5*cfg.source_srate)
            ttl += [ttl_[t0:]]
            ieeg += [ieeg_[:,t0:]]

        # loop over sessions
        ieeg_full = []
        ttl_full = []
        for i in range(0,len(ttl),2):
            m = min(len(ttl[i]),len(ttl[i+1]))
            ieeg_session = np.vstack((ieeg[i][:,:m],ieeg[i+1][:,:m]))
                        
            ieeg_full += [ieeg_session]
            ttl_full += [ttl[i][:m]]

        # concatenate sessions
        ieeg = np.concatenate(ieeg_full,axis=-1)
        ttl = np.concatenate(ttl_full)
 
       
        print(f"\n{ieeg.shape[0]} channels, {int(len(ttl)/cfg.source_srate)} seconds, {np.sum(ttl)} events, {round(ieeg.nbytes/1e9,2)} GB of data loaded")
        cf.print_d("saving...")

        # save one ieeg, one ttl file per subject
        fname = os.path.join(SP['RawPath'],f"{sub}_ttl.csv")
        ttl.tofile(fname,sep=',')
        


        if len(SP['ChNames'])!=ieeg.shape[0]: dp = fix_channels(dp, len(ieeg))

        cf.display_progress(f"Making MNE Raw, {sub} ",iSub,len(subs),TimeStart)

        # Create MNE info
        info = mne.create_info(SP['ChNames'], sfreq=cfg.source_srate, ch_types='eeg')
        # Finally, create the Raw object
        raw = mne.io.RawArray(ieeg, info)


        montage = mne.channels.make_dig_montage(dict(zip(SP['ChNames'],SP['coords'])))
        raw.set_montage(montage)

        #print(raw.get_montage().get_positions()['ch_pos']['001-AH1'])
         


        # downsample for memory
        if cfg.srate >= raw.info['sfreq']: 
            print(f"Error: Original sampling freq smaller than or equal to target sampling freq ({raw.info['sfreq']} Hz <={cfg.srate} Hz)")
        else:
            # resample raw
            cf.display_progress(f"Resampling from {cfg.source_srate} to {cfg.srate}, {sub}",iSub,len(subs),TimeStart)
            raw.resample(cfg.srate)


        cf.display_progress(f"Applying notch filtering at {cfg.landline_noise} Hz, and 4 harmonics, {sub}" ,iSub,len(subs),TimeStart)
        raw.notch_filter(cfg.landline_noise*np.arange(1,5), filter_length='auto', phase='zero',picks='all')      

        cf.print_d(f"Saving mne raw files for {sub}")
        fname = os.path.join(SP['RawPath'],f"{sub}_raw.fif")
        raw.save(fname, picks='all', overwrite=True)

        print(' ')



################################################################
# DUMMY CHANNELS FOR MISSING FILE
#
def fix_channels(dp, l):

    L = len(SP['ChNames'])

    if L>l: sys.exit('Number of channel labels larger than data') 

    print(f"\nCreating dummy channel info for {SP['sub']}")
    SP['ChNames'] += [f'{str(i+1).zfill(3)}-EL{i+1}' for i in range(L,l)]
    SP['coords'] += [[np.nan,np.nan,np.nan]]*(l-L)
    electrodes = {'names':[ch[4:] for ch in SP['ChNames']],'coords':SP['coords']}
    with open(os.path.join(SP['RawPath'],  SP['sub'] + '_electrodes.json'), 'w') as fp:
        json.dump(electrodes, fp)

    return dp


