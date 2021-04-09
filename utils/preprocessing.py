# import packages
import numpy as np
import pandas as pd
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
#   REJECTION
#
##############################################################################

def rejection_wrap(subs=cfg.subs, figures=True):
    """ Identify bad channels early in the processing and discard them
        completely from raw files.

        Rejection based on std of V and dVdt
    """

    print("\n======== rejection ==============================================\n")
    print(f"SUB\tTOTAL\tKEPT\tREJECTED")
    for i_sub, sub in enumerate(subs): 
       
        dp = cf.sub_params(sub)
        mneraw_file = os.path.join(dp['raw_path'],f"{sub}_raw.fif")
        raw = mne.io.read_raw_fif(mneraw_file,preload=True)

        kept, stats, thresholds = rejection(raw)

        if figures: rejection_figure(raw,sub,dp['figures_path'],kept,stats,thresholds)

        # mark bad channels in raw.info
        #raw.info['bads'] = np.array(raw.ch_names)[~kept]

        raw.drop_channels(np.array(raw.ch_names)[~kept])
        mneraw_file = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
        raw.save(mneraw_file, picks='all', overwrite=True)

        print(f"{dp['sub']}\t{len(dp['ch_names'])}\t{np.sum(kept)}\t{np.sum(~kept)}\t{int(100*np.sum(~kept)/len(kept))}%")



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
    s_v = np.std(data,axis=-1)
    s_dv = np.std(np.diff(data,axis=-1),axis=-1)
    slopes = fit_power_law(raw)
    #slopes = np.random.rand(len(s_v))

    # use robust scaler to get relative values
    x = np.squeeze(RobustScaler().fit_transform(s_v.reshape(-1, 1)))
    y = np.squeeze(RobustScaler().fit_transform(s_dv.reshape(-1, 1)))
    z = np.squeeze(RobustScaler().fit_transform(slopes.reshape(-1, 1)))

    # get thresholds in original scale
    thresholds = [
         np.squeeze(RobustScaler().fit(s_v.reshape(-1, 1)).inverse_transform([[cfg.V_low],[cfg.V_high]])),
         np.squeeze(RobustScaler().fit(s_dv.reshape(-1, 1)).inverse_transform([[cfg.dV_low],[cfg.dV_high]])),
         np.squeeze(RobustScaler().fit(slopes.reshape(-1, 1)).inverse_transform([[cfg.s_low],[cfg.s_high]]))
                  ]

    # identify good channels (those within the thresholds)
    kept = (x>cfg.V_low)*(x<cfg.V_high)*(y>cfg.dV_low)*(y<cfg.dV_high)*(z>cfg.s_low)*(z<cfg.s_high)
    
    # mark bad channels in raw.info
    #raw.info['bads'] = np.array(raw.ch_names)[~kept]

    return kept, np.array([s_v,s_dv,slopes]), thresholds


def fit_power_law(raw):
    """ Estimate exponential slope of power spectral density
        
    """

    psds, f = mne.time_frequency.psd_welch(raw,fmin=10,fmax=200,n_fft=512)

    slopes = np.array([])
    for i in range(len(raw.ch_names)):
        if np.sum(psds[i]>0) == 0:      # for channels with 0 var
            slopes = np.append(slopes,0)
        else:
            res = linregress(np.log(f),np.log(psds[i]))
            slopes = np.append(slopes,res.slope)
        
            

    return slopes


def rejection_figure(raw,sub,path,kept,stats,thresholds,save=True,plot=False):
    """ Make rejection summary figure for single subject
        
    """

    #create figure
    fig, ax = plt.subplots(2,1,figsize=(7,7))
    fig.suptitle(f"{sub}, rejected {np.sum(~kept)}/{len(kept)}, {int(100*np.sum(~kept)/len(kept))}%",fontsize = 15)

    # rejected channel positions
    pl.brain_plot(ax[0],cf.get_coords(raw),1.*(~kept),mask=kept,mode='symmetric',interval=[0,1])

    # scatter with rejection summary
    pl.scatter_plot(ax[1],stats[0],stats[1],stats[2],colorbar='s',mask=kept,
                    ylabel='std(dVdt)',xlabel='std(V)', vlines = thresholds[0],hlines=thresholds[1])

    # save figure
    fig_name = os.path.join(cf.check_path([path,'rejection']),f"{sub}.pdf")
    if save: fig.savefig(fig_name, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()

    

def rejection_epochs(subs=cfg.subs,ep_setups=cfg.ep_setups):
    """ Identify bad channels early in the processing and discard them
        completely from epochs files.

        Rejection based on std of V and dVdt
    """

    print("\n======== rejection ==============================================\n")
    for i_sub, sub in enumerate(subs): 
       
        for i_ep, ep_setup in enumerate(ep_setups):
            dp = cf.sub_params(sub)
            # load epochs
            epochs_file = os.path.join(dp['derivatives_path'],f"{sub}_epochs-{ep_setup['name']}-epo.fif")
            epochs = mne.read_epochs(epochs_file)

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
            for i_ch, k in enumerate(kept): print(f"{i_ch}\t{k}\t{np.around(s_v[i_ch],decimals=2)}\t{np.around(s_dv[i_ch],decimals=2)}")




##############################################################################
#
#   REFERENCING
#
##############################################################################


def referencing_wrap(subs=cfg.subs,ref_method=cfg.ref_method):

    print(f"\n======== referencing ================================================\n")

    # ITERATE SUBJECT LIST =============================================================================
    time_start = time.time()
    for i_sub, sub in enumerate(subs): 
        cf.display_progress(f"{sub}",i_sub,len(subs),time_start)

        # add paths to parameters
        dp = cf.sub_params(sub)

        # load raw
        #fname = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
        fname = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
        raw = mne.io.read_raw_fif(fname, preload=True)
        
        ref_raw = referencing(raw,ref_method)

        # save only selection of channels 
        fname = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
        ref_raw.save(fname, picks='all', overwrite=True)

        # move to next subject ===============================================

    cf.print_d(f"{cfg.ref_method} reference done!")
    print(f"\n")




def referencing(raw,ref_method):
    """ Subtract reference, computed with different methods
        Discard sensors (eg extremes of probes in laplacian)

    Parameters
    ----------
    raw : mne raw
    ref_method : str

    Returns
    -------
    ref_raw :  mne raw
        referenced, channels discarded if necessary
    """

    data = raw.get_data(picks='all')

    if ref_method == 'cmr':
        reference_data = np.tile(np.median(data,axis=0),(len(raw.ch_names),1))
        bads = []

    elif ref_method == 'car':
        reference_data = np.tile(np.mean(data,axis=0),(1,len(raw.ch_names)))
        bads = []

    elif ref_method == 'laplacian':
        reference_data, bads = laplacian_ref(data,raw.ch_names)       

    elif ref_method == 'bipolar':
        reference_data, bads = bipolar_ref(data,raw.ch_names) 
        
    elif ref_method == 'none' or referencing == None: pass

    else: sys.exit(f'\n{referencing} referencing not recognised  ¯\_(´･_･`)_/¯ ')
    
    # referenced data
    ref_raw = mne.io.RawArray(data-reference_data, raw.info.copy())
    ref_raw.info['bads'] += bads
    ref_raw.drop_channels(bads)

    return ref_raw


def laplacian_ref(data, ch_names):

    ch_probes =  [''.join(filter(lambda x: not x.isdigit(), ch_name)) for ch_name in ch_names]
    bads = []

    # create a dummy raw that contains the references
    reference_data = np.zeros_like(data)

    for i_ch, ch_name in enumerate(ch_names[:-1]):     
        reference_data[i_ch] = np.mean([data[i_ch-1], data[i_ch+1]], 0)
        if ch_probes[i_ch -1]!=ch_probes[i_ch +1]: bads +=[ch_name]

    return reference_data, bads


def bipolar_ref(data, ch_names):

    ch_probes = [''.join(filter(lambda x: not x.isdigit(), ch_name)) for ch_name in ch_names]
    bads = []

    # create a dummy raw that contains the references
    reference_data = np.zeros_like(data)

    for i_ch, ch_name in enumerate(ch_names[:-1]):     
        reference_data[i_ch] = data[i_ch+1]
        if ch_probes[i_ch]!=ch_probes[i_ch +1]: bads +=[ch_name]


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


    print(f"\n======== mne.raw 2 mne.rawTFR ===================================\n")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 

        cf.display_progress(f"{sub}, {band['name']} power", i, len(iterator),time_start) 


        # load subject data and raw file only once per subject
        if i_band==0:
            # add paths to parameters
            dp = cf.sub_params(sub)
            # load raw
            fname = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
            raw = mne.io.read_raw_fif(fname, preload=True)

        # skip if file already exists
        fname = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_raw.fif")
        if os.path.isfile(fname) and skip: continue
            
        raw_band = raw2TFR(raw,band)

        # save
        raw_band.save(fname, picks='all', overwrite=True)


    cf.print_d(f"done! elapsed time {int((time.time() - time_start)/60.)} min")
    print(f"\n")




def epochs2TFR_wrap(subs=cfg.subs, bands=cfg.bands,ep_setups=cfg.ep_setups):
    """ Loop over subjects and bands and make and save
        mne epochs objects with band power.

    """


    print(f"\n======== mne.epochs 2 mne.epochsTFR ===================================\n")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time()
    iterator =  list(product(enumerate(subs),enumerate(ep_setups),enumerate(bands)))
    for i, ((i_sub, sub), (i_ep, ep_setup), (i_band, band)) in enumerate(iterator): 

        cf.display_progress(f"{sub}, {band['name']} power", i, len(iterator),time_start) 

        # load subject data and raw file only once per subject
        if i_band==0:
            # add paths to parameters
            dp = cf.sub_params(sub)
            # load epochs
            epochs_file = os.path.join(dp['derivatives_path'],f"{sub}_epochs-{ep_setup['name']}-epo.fif")
            epochs = mne.read_epochs(epochs_file)
            
        epochs_band = raw2TFR(epochs,band)

        # save
        fname = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_epochs-{ep_setup['name']}-epo.fif")
        epochs_band.save(fname, overwrite=True)


    cf.print_d(f"done! elapsed time {int((time.time() - time_start)/60.)} min")
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
    raw_band : mne raw 
    """

    # if fmax > fNyquist lower fmax and print warning
    if band['fmax']!= None and band['fmax'] >= cfg.srate/2:
        band['fmax'] = cfg.srate/2 - 1;
        print(f"fmax larger than Nyquist freq., reduced to {band['fmax']}")

    # for broad band files just save again
    if band['method']== None or band['method']== 'none': 
        raw_band = raw.copy()

    elif band['method']== 'hilbert':
        # apply filter
        raw_band = raw.copy().filter(band['fmin'],band['fmax'],phase='zero',picks='all')
        # compute envelope
        raw_band.apply_hilbert(picks='all', envelope=True)

    elif band['method']== 'complex':
        # apply filter
        raw_band = raw.copy().filter(band['fmin'],band['fmax'],phase='zero',picks='all')
        # get complex hilbert
        raw_band.apply_hilbert(picks='all', envelope=False)
        # get phase
        #raw_band.apply_function(np.angle,picks='all',channel_wise=True)

    elif band['method']== 'filter':
        # apply filter
        raw_band = raw.copy().filter(band['fmin'],band['fmax'],phase='zero',picks='all')

    elif band['method']== 'wavelet':
        raw_band = wavelet(raw.copy(),band)
    else: sys.exit(f"\n{band['method']} method not recognised  ¯\_(´･_･`)_/¯ ")
    

    return raw_band





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
    raw_band : mne raw 
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
    if cfg.input_format == 'raw': raw_band = mne.io.RawArray(power[0].mean(axis=-2), raw.info)
                #                                              ^ in raw there is only one long "epoch"
    if cfg.input_format == 'epochs': raw_band = mne.EpochsArray(power.mean(axis=-2), raw.info, events=raw.events, metadata=raw.metadata,tmin=raw.tmin)

    return raw_band
    




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

    print(f"\n======== Add ROIs from atlas ===============================\n")


    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start=time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 

        # load subject paths 
        dp = cf.sub_params(sub)

        # load raw
        fname = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(fname, preload=True)

        # get list non ROI channels only
        ch_names = [ch_name for ch_name in raw.ch_names if ch_name[:3]!='ROI']
        coords = cf.get_coords(raw,picks=ch_names)
        labels = cf.get_ROI_labels(coords)

        for label in list(set(labels)):
            ROI = [ch_names[j] for j, l in enumerate(labels) if l==label ]
            if len(ROI)>0: # avoid adding empoty sets
                raw = add_ROI(raw,ROI,f"ROI-{label}")       
                cf.display_progress(f"{raw.ch_names[-1]} ({len(ROI)} channels) added to {sub} {band['name']}",i,len(iterator),time_start)
        raw.save(fname, picks='all', overwrite=True)
 


def add_ROI(raw, ROI, ROIname):
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
    if ROI == 'all': ROI = [ch_name for ch_name in raw.ch_names if ch_name[0].isdigit()]



    ROIdata = np.array([raw.copy().get_data(picks=ROI).mean(axis=0)])
    # Create MNE info
    ROIinfo = mne.create_info([ROIname], sfreq=cfg.srate, ch_types='eeg')
    # Create the Raw object
    ROIraw = mne.io.RawArray(ROIdata,ROIinfo)

    # ROI center of mass
    ROIcoords = cf.get_coords(raw,picks=ROI).mean(axis=0)

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

    print(f"\n======== Clear ROIs  ===================================\n")


    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 

        # load subject paths 
        dp = cf.sub_params(sub)

        # load raw
        fname = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(fname, preload=True)

        # get list non ROI channels only
        ch_names = [ch_name for ch_name in raw.ch_names if ch_name[:3]=='ROI']
        raw.drop_channels(ch_names)
        raw.save(fname, picks='all', overwrite=True)
 
        cf.display_progress(f"{sub} {band['name']}, {len(ch_names)} ROIs removed",i,len(iterator),time_start)

 



##############################################################################
#
#   MAKE RAW FROM SOURCE
#
##############################################################################


def source2raw(subs=cfg.subs):

    from neo.io import BlackrockIO

    time_start = time.time()
    # ITERATE SUBJECT LIST =============================================================================
    for i_sub, sub in enumerate(subs): 

        print(f"\n--- {sub} -----------------------------------\n")
        # add paths to parameters
        dp = cf.sub_params(sub)
        cf.display_progress(f"Loading source data, {sub}",i_sub,len(subs),time_start)
        ieeg = []
        ttl = []
        # 2 files per session
        for i, source_file in enumerate(dp['source_files']):
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
        fname = os.path.join(dp['raw_path'],f"{sub}_ttl.csv")
        ttl.tofile(fname,sep=',')
        


        if len(dp['ch_names'])!=ieeg.shape[0]: dp = fix_channels(dp, len(ieeg))

        cf.display_progress(f"Making MNE Raw, {sub} ",i_sub,len(subs),time_start)

        # Create MNE info
        info = mne.create_info(dp['ch_names'], sfreq=cfg.source_srate, ch_types='eeg')
        # Finally, create the Raw object
        raw = mne.io.RawArray(ieeg, info)


        montage = mne.channels.make_dig_montage(dict(zip(dp['ch_names'],dp['coords'])))
        raw.set_montage(montage)

        #print(raw.get_montage().get_positions()['ch_pos']['001-AH1'])
         


        # downsample for memory
        if cfg.srate >= raw.info['sfreq']: 
            print(f"Error: Original sampling freq smaller than or equal to target sampling freq ({raw.info['sfreq']} Hz <={cfg.srate} Hz)")
        else:
            # resample raw
            cf.display_progress(f"Resampling from {cfg.source_srate} to {cfg.srate}, {sub}",i_sub,len(subs),time_start)
            raw.resample(cfg.srate)


        cf.display_progress(f"Applying notch filtering at {cfg.landline_noise} Hz, and 4 harmonics, {sub}" ,i_sub,len(subs),time_start)
        raw.notch_filter(cfg.landline_noise*np.arange(1,5), filter_length='auto', phase='zero',picks='all')      

        cf.print_d(f"Saving mne raw files for {sub}")
        fname = os.path.join(dp['raw_path'],f"{sub}_raw.fif")
        raw.save(fname, picks='all', overwrite=True)

        print(' ')




def csv2raw(subs=cfg.subs):

    print(f"\n======== csv 2 mne.raw ================================================\n")

    # ITERATE SUBJECT LIST =============================================================================
    # check time to monitor progress
    time_start = time.time()
    for i_sub, sub in enumerate(subs): 

        # add paths to parameters
        dp = cf.sub_params(sub)

        cf.display_progress(f"{sub} Loading file...",i_sub,len(subs),time_start)

        # GET RAW DATA ---------------------------------------------------------------------------
        # Read the CSV file as a NumPy array
        #data = np.loadtxt(dp['raw_file'], delimiter=',',dtype=np.int16)
        chunk = pd.read_csv(dp['raw_file'],chunksize=1,dtype=np.int16)
        data = pd.concat(chunk)


        if len(dp['ch_names'])!=len(data): dp = fix_channels(dp, len(data))
        cf.display_progress(f"{sub} Data loaded",i_sub,len(subs),time_start)

        # Create MNE info
        info = mne.create_info(dp['ch_names'], sfreq=cfg.source_srate, ch_types='eeg')
        # Finally, create the Raw object
        raw = mne.io.RawArray(data, info)

        # downsample for memory
        if cfg.srate >= raw.info['sfreq']: 
            sys.exit(f"Error: Original sampling freq smaller than or equal to target sampling freq ({raw.info['sfreq']} Hz <={cfg.srate} Hz)")
        else:
            # resample raw
            cf.display_progress(f"{sub} Resamoling from {cfg.source_srate} to {cfg.srate}",i_sub,len(subs),time_start)
            raw.resample(cfg.srate)

        cf.display_progress(f"{sub} Applying notch filtering at {cfg.landline_noise} Hz, and 4 harmonics",i_sub,len(subs),time_start)
        raw.notch_filter(cfg.landline_noise*np.arange(1,5), filter_length='auto', phase='zero',picks='all')      

        cf.print_d(f"Saving mne raw files for {sub}")
        fname = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
        raw.save(fname, picks='all', overwrite=True)





################################################################
# DUMMY CHANNELS FOR MISSING FILE
#
def fix_channels(dp, l):

    L = len(dp['ch_names'])

    if L>l: sys.exit('Number of channel labels larger than data') 

    print(f"\nCreating dummy channel info for {dp['sub']}")
    dp['ch_names'] += [f'{str(i+1).zfill(3)}-EL{i+1}' for i in range(L,l)]
    dp['coords'] += [[np.nan,np.nan,np.nan]]*(l-L)
    electrodes = {'names':[ch[4:] for ch in dp['ch_names']],'coords':dp['coords']}
    with open(os.path.join(dp['raw_path'],  dp['sub'] + '_electrodes.json'), 'w') as fp:
        json.dump(electrodes, fp)

    return dp


