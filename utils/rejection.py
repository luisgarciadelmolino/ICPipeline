# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   CHANNEL REJECTION
#   ~~~~~~~~~~~~~~~~~
#
#   channel_rejection
#   drop_bad_epochs
#   spike_detection
#   channel_statistics
#
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from itertools import product

import utils.common_functions as cf
import static_parameters as sp
import utils.plots as pl




def simple_rejection(subs=sp.subject_list)
    """ Identify bad channels early in the processing and discard them
        completely from raw files.

        Rejection based on std and kurtosis
    """

    print("\n======== rejection ==============================================\n")
    time_start = time.time() 
    for i_sub, sub in enumerate(subs): 
        cf.display_progress(f"{sub}", i_sub, len(subs),time_start) 

        dp = cf.sub_params(sub)
        mneraw_file = os.path.join(dp['derivatives_path'],f"{sub}_raw.fif")
        raw = mne.io.read_raw_fif(mneraw_file,preload=True)

        data = raw.get_data()
        s = cf.clip_n_rescale(np.std(data,axis=-1), c = 0, zscore = False)
        k = cf.clip_n_rescale(kurtosis(data,axis=-1), c = 0, zscore = False)
    
        dx = 0.1
        x = np.arange(-5,5,dx)
        hs, _ = np.histogram(s,x)
        hk, _ = np.histogram(k,x)
        plt.plot(x[:-1]+dx/2,hs,'r')
        plt.plot(x[:-1]+dx/2,hk,'b')
        plt.show()


def rejection(subs=sp.subject_list, ep_setups=sp.epoch_setups):
    """
    """

    print("\n======== rejection ==============================================\n")


    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time() 
    iterator =  list(product(enumerate(subs), enumerate(ep_setups)))
    for i, ((i_sub, sub), (i_ep, ep_setup)) in enumerate(iterator): 
        cf.display_progress(f"{sub}, {ep_setup['level']}", i, len(iterator),time_start) 


        if i_ep == 0: 
            # load subject params
            dp = cf.sub_params(sub)
            rejected={}

        rejected[ep_setup['level']]={}

        K, S = [], []

        time_start = time.time()
        for i_ch, ch_name in enumerate(dp['ch_names']): 
            try: 
                fname = os.path.join(dp['derivatives_path'],f"{sub}_ch-{ch_name}_band-broad_rejection_level-{ep_setup['level']}_epo.fif")            
                dy = np.diff(np.squeeze(mne.read_epochs(fname, preload=True).get_data()),axis=-1)
            except: 
                rejected[ep_setup['level']][ch_name] = []
                continue

            if kurtosis(dy,axis=None)>10.:
                rejected[ep_setup['level']][ch_name] = [True]*len(dy)
            else:
                k, s = kurtosis(dy,axis=-1), skew(dy,axis=-1)            
                rejected[ep_setup['level']][ch_name] = ((np.abs(s)>2) | (k>20)).tolist()
        
        # save when done
        if i_ep == len(ep_setups)-1:
            fname = os.path.join(dp['derivatives_path'],f"{sub}_rejection.json")       
            json.dump(rejected, open(fname, 'w'))





def rejection_fig_single_ch(sub, ch_name, ep_setups=sp.epoch_setups, plot = True, save = False, temp=False):
    """ 
    """

    dp = cf.sub_params(sub)

    # make figure
    fig, ax = plt.subplots(5,len(ep_setups),figsize=(5*len(ep_setups),10), 
                               gridspec_kw={'height_ratios': [2,1,1,1,1]})    
    
    # add extra dimension to ax if it is missing
    if len(ep_setups) == 1: ax = np.array([ax]).T

    fig.suptitle(f"{sub} {ch_name}\n{dp['gyri'][dp['ch_names'].index(ch_name)]}")
    pl.clear_axes(ax[0,:])    
    pl.channel_position_plot(ax[0,0],dp['coords'],dp['ch_names'].index(ch_name))

    for i_ep, ep_setup in enumerate(ep_setups):
        try: 
            fname = os.path.join(dp['derivatives_path'],f"{sub}_ch-{ch_name}_band-broad_level-{ep_setup['level']}_epo.fif")            
            epochs = mne.read_epochs(fname, preload=True).decimate(10)
        except:
            plot = False
            save = False
            continue

        rejected = np.array(dp['rejected'][ep_setup['level']][ch_name])

        y = np.squeeze(epochs.get_data())
        t = epochs.times

        if np.sum(~rejected) > 1:
            title = f"Kept {np.sum(~rejected)}/{len(rejected)}, {int(100*np.sum(~rejected)/len(rejected))}%"
            pl.imshow_plot(ax[1,i_ep],y[~rejected], title = title,vmin = -1, vmax = 1 ,xlims = t,
                             vlines = ep_setup['xticks'], cmap='RdBu_r',colorbar = '' +'IQR'*(i_ep==len(ep_setups)-1))
            pl.trace_plot(ax[2,i_ep],t,y[~rejected],vlines = ep_setup['xticks'],plot_sem=True)
        else: 
            pl.clear_axes(ax[1:3,i_ep])    

        if np.sum(rejected) > 1:
            title = f"Rejected {np.sum(rejected)}/{len(rejected)}, {int(100*np.sum(rejected)/len(rejected))}%"
            pl.imshow_plot(ax[3,i_ep],y[rejected], title = title ,vmin = -1, vmax = 1, xlims = t,
                             vlines = ep_setup['xticks'], cmap='RdBu_r',colorbar = '' +'IQR'*(i_ep==len(ep_setups)-1))
            pl.trace_plot(ax[4,i_ep],t,y[rejected],vlines = ep_setup['xticks'],plot_sem=True)
        else: 
            pl.clear_axes(ax[3:5,i_ep])  

    ax[-1,0].set_xlabel('t (s)')
    fig.subplots_adjust(wspace=0.3, hspace=0.5)  #left=0.05, bottom=0.05, right=0.95, top=0.98, 

    if plot:    # show figure and wait for keyboard input to close
        plt.show()
         
    if save: 
        figures_path = cf.check_path(['..','Figures' + sp.out,'rejection'])
        fig_name = os.path.join(figures_path,f"{sub}_{ch_name}{'temp'*temp}.pdf")
        fig.savefig(fig_name, format='pdf') 
    
    plt.close()  




def rejection_fig_single_ch_wrap(subs=sp.subject_list, ep_setups=sp.epoch_setups):
    """ 
    """

    for i_sub, sub in enumerate(subs): 
        dp = cf.sub_params(sub)
        for i_ch, ch_name in enumerate(dp['ch_names']): 
            cf.print_d(f"{sub}, {ch_name}")
 
            rejection_fig_single_ch(sub, ch_name, ep_setups=sp.epoch_setups, plot = False, save = True, temp = True)

        figures_path = cf.check_path(['..','Figures' + sp.out,'rejection'])
        cf.concatenate_pdfs(figures_path,'temp',f"{sub}_rejection.pdf", remove=True)



def rejection_figure_statistics(subs=sp.subject_list, ep_setups=sp.epoch_setups, plot = True, save = True, temp=False):

    # make figure
    fig, ax = plt.subplots(len(subs),len(ep_setups),figsize=(5*len(ep_setups),10))
    
    # add extra dimension to ax if it is missing
    if len(ep_setups) == 1: ax = np.array([ax]).T

    fig.suptitle(f"Rejection summary")

    iterator =  list(product(enumerate(subs), enumerate(ep_setups)))
    for i, ((i_sub, sub), (i_ep, ep_setup)) in enumerate(iterator): 
        dp = cf.sub_params(sub)
        rejected = np.array([dp['rejected'][ep_setup['level']][ch_name] for ch_name in dp['ch_names']])

        pl.brain_plot(ax[i_sub,i_ep],dp['coords'],np.mean(rejected,axis=-1),title=sub, 
                        colorbar = '' + (i_sub==len(subs)-1)*'rejected', mode='interval',interval=[0,1])        


    if plot:    # show figure and wait for keyboard input to close
        plt.show()
         
    if save: 
        figures_path = cf.check_path(['..','Figures' + sp.out,'rejection'])
        fig_name = os.path.join(figures_path,f"rejection_summary{'temp'*temp}.pdf")
        fig.savefig(fig_name, format='pdf') 
    
    plt.close()  



'''
def rejection_figure(sub, ch_name, raw, metadata, events, ep_setups=sp.epoch_setups, plot = True, save = False, temp=False):
    """ Same as subject overview but for one single channel
    

    Parameters
    ----------
    key :  str
        metadata key to plot separate traces by metadata value
    """


    figures_path = cf.check_path(['..','Figures' + sp.out,'rejection'])


    # load subject params
    dp = cf.sub_params(sub)
    if ch_name not in dp['ch_names']: sys.exit(f"{sub} {ch_name} not found  (o_0) ")

    # make figure
    fig, ax = plt.subplots(6,len(ep_setups),figsize=(5*len(ep_setups),12))#, 
#                            gridspec_kw={'width_ratios': [e['tmax'] - e['tmin'] for e in ep_setups]})    
    # add extra dimension to ax if it is missing
    if len(ep_setups) == 1: ax = np.array([ax]).T

    fig.suptitle(f"{sub} {ch_name}\n{dp['gyri'][dp['ch_names'].index(ch_name)]}")

    
    pl.clear_axes(ax[0,:])    
    pl.channel_position_plot(ax[0,0],dp['coords'],dp['ch_names'].index(ch_name))

    # LOOP OVEREPOCH SETUPS -------------------------------------------------------
    for i_ep, ep_setup in enumerate(ep_setups): 

        # filter metadata according to epoch_setup
        epo_metadata = metadata[metadata['eventtype'].isin(ep_setup['event_type'])]
        # get event labels for given epoch_setup
        event_id = list(set(epo_metadata['number']))
        epo_events = events[np.isin(events[:,-1],event_id),:]

        epochs = mne.Epochs(raw, epo_events, event_id=event_id, tmin=ep_setup['tmin'], tmax=ep_setup['tmax'], picks='all', metadata = epo_metadata, preload=True, baseline=None,reject_by_annotation=False, verbose=False)


        y = np.squeeze(epochs.get_data())
        dy = np.diff(y,axis=-1)
        dy = (dy - np.mean(dy))/dy.std()
        t = epochs.times
        #k = kurtosis(np.diff(y,axis=-1),axis=-1)
        k = np.std(dy,axis=-1)
        #k = y.std(axis=-1)-1 
        k = np.squeeze(RobustScaler().fit_transform(k.reshape(-1, 1)))
        rejected = np.abs(k)>sp.k_threshold
        #rejected=k>sp.k_threshold

        dx = 0.1
        x = np.arange(-5,5,dx)
        h, _ = np.histogram(k,x,density=True)
        h2, _ = np.histogram(dy,x,density=True)
        x = x[:-1]
        ax[1,i_ep].bar(x+0.5*dx,h,width=dx)
        ax[1,i_ep].bar(x[x>sp.k_threshold]+0.5*dx,h[x>sp.k_threshold],width=dx,color='firebrick')
        ax[1,i_ep].plot(x+0.5*dx,h2,'k')
        ax[1,i_ep].set_xlabel(r'$\kappa$')
        ax[1,0].set_ylabel('# epochs')

        ax[1,i_ep].spines['right'].set_visible(False)
        ax[1,i_ep].spines['top'].set_visible(False)
        ax[1,i_ep].tick_params(axis='both', which='both', size = 0, labelsize=7)



        y = (y - np.median(y))/np.subtract(*np.percentile(y, [75, 25]))

        # make linearly or logspaced list of freqs for morlets
        freqs = np.linspace(1,140,20)
        # compute wavelets
        i_t = int(0.5*sp.srate)
        power = np.squeeze(mne.time_frequency.tfr_array_morlet(np.array(epochs.get_data()),sp.srate, freqs, n_cycles=freqs / 2., output = 'power'))[:,:,i_t:-i_t]
        power = (power - power.mean(axis=(0,-1))[np.newaxis,:,np.newaxis])/power.std(axis=(0,-1))[np.newaxis,:,np.newaxis]


        if np.sum(~rejected) > 1:
            title = f"Kept {np.sum(~rejected)}/{len(rejected)}, {int(100*np.sum(~rejected)/len(rejected))}%"
            pl.imshow_plot(ax[2,i_ep],y[~rejected,i_t:-i_t], title = title,vmin = -2, vmax = 2 ,xlims = t[i_t:-i_t],
                             vlines = ep_setup['xticks'], cmap='RdBu_r',colorbar = '' +'IQR'*(i_ep==len(ep_setups)-1))

            pl.imshow_plot(ax[3,i_ep],np.mean(power[~rejected],axis=0),xlims=t[i_t:-i_t], 
                              ylims=freqs,vlines = ep_setup['xticks'], cmap='RdBu_r',colorbar = '' +'5% - 95%'*(i_ep==len(ep_setups)-1))
        else: pl.clear_axes(ax[2:4,i_ep])    

        if np.sum(rejected) > 1:
            title = f"Rejected {np.sum(rejected)}/{len(rejected)}, {int(100*np.sum(rejected)/len(rejected))}%"
            pl.imshow_plot(ax[4,i_ep],y[rejected,i_t:-i_t], title = title ,vmin = -2, vmax = 2, xlims = t[i_t:-i_t],
                             vlines = ep_setup['xticks'], cmap='RdBu_r',colorbar = '' +'IQR'*(i_ep==len(ep_setups)-1))

            pl.imshow_plot(ax[5,i_ep],np.mean(power[rejected],axis=0), xlims=t[i_t:-i_t], 
                              ylims=freqs,vlines = ep_setup['xticks'], cmap='RdBu_r',colorbar = '' +'5% - 95%'*(i_ep==len(ep_setups)-1))

        else: pl.clear_axes(ax[4:6,i_ep])  

    ax[-1,0].set_xlabel('t (s)')
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.98, wspace=0.3, hspace=0.5)

    if plot:    # show figure and wait for keyboard input to close
        plt.show()
         
    if save: 
        fig_name = os.path.join(figures_path,f"{sub}_{ch_name}{'temp'*temp}.pdf")
        fig.savefig(fig_name, format='pdf') 
    
    plt.close()  
'''





















####################################################################################################
#   read channel rejection
#   look for rejection dictionary for sub and epoch given
#   if file does not exist call compute channel rejection 
#
def read_channel_rejection(P):

    # if epoch and sub are defined get rejection for specific epoch and sub
    if 'epoch' in P.keys() and 'sub' in P.keys(): 
        P = sub_params(P)
        rejected_file = os.path.join(P['derivatives_path'] ,  P['sub'] + '_level-' + P['epoch']['level'] + '_rejected.json')
        if os.path.isfile(rejected_file):
            with open(rejected_file,'r') as json_file: P['ch'] = json.load(json_file)
        else:
            print('Rejection file not found, computing  rejection')
            compute_channel_rejection(P)
            with open(rejected_file,'r') as json_file: P['ch'] = json.load(json_file)
    # if epoch or subject are not defined loop over all and save file
    else:
        sys.exit('Subject or epoch tag not specified')



####################################################################################################
#   Channel rejection
#   loop over subjects and epochs
#   compute channel rejection for each pair sub - epoch and save to file
#
def compute_channel_rejection(P):

    print('Channel rejection' + 50*'=') 

    for P['sub'] in P['subject_list']:

        print('\n' + P['sub'] + 50*'-') 

        for P['epoch'] in P['epochs']:
            # add paths to parameters
            P = sub_params(P)

            ch ={ 'rejected' : [], }   


            print('\nChannel rejection '+ P['epoch']['level'] + 50*'-')
            print('Channel    \t\trej.\t\tProgress')
        

            # get list of time domain files
            epoch_file_list = glob.glob(P['derivatives_path'] +'/*'+P['epoch']['level']+'_td_epo.fif')
            epoch_file_list.sort()

            if P['save_one_file_per_channel'] : num_channels = len(epoch_file_list)

            percent_rejected = []

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      
            # do annalysis file by file
            #
            for i,filename in enumerate(epoch_file_list): 

                epochs = mne.read_epochs(filename).decimate(2)   
                tmin,tmax=epochs.times[0],epochs.times[-1]
                # compute wavelets
                f = np.arange(5,101,5)
                tfrepochs = mne.time_frequency.tfr_morlet(epochs, freqs=f, n_cycles=f*0.5, use_fft=True,return_itc=False, n_jobs=1,picks='all',average=False,output='complex').crop(tmin=tmin + P['crop'], tmax=tmax - P['crop'])
                t = tfrepochs.times 

                amplitude = np.abs(np.squeeze(tfrepochs.data))
                amplitude = amplitude/np.median(amplitude,axis=(0,-1))[np.newaxis,:,np.newaxis]

                srate = t[1]-t[0]
                timewindowsize = 0.2 # in seconds

                good_epochs = epochs.__len__()*[True]


                for j in range(tfrepochs.__len__()):
                    # loop over time windows
                    t1 = 0
                    t2 = t1 + int(timewindowsize/srate)
                    while t2<amplitude.shape[-1]:
                        w3 = amplitude[j,:,t1:t2]            
                        if np.max(w3)>4:
                            corr = np.corrcoef(w3)
                            corr = corr[np.triu_indices_from(corr,k=1)]
                            if kurtosis(corr)>0: good_epochs[j] = False
                        if good_epochs[j] == False:  break
                        t1 = t2
                        t2 = t1 + int(timewindowsize/srate)

                bad_epochs = [not e for e in good_epochs]
                ch[epochs.ch_names[0]] = bad_epochs

                print(f'{epochs.ch_names[0]}\t\t{str(np.sum(bad_epochs))}/{str(len(bad_epochs))}\t\t({str(i+1)}/{str(num_channels)})')

                percent_rejected += [100.*np.sum(bad_epochs)/len(bad_epochs)]

            rejected_file = os.path.join(P['derivatives_path'] , P['sub'] + '_level-' + P['epoch']['level'] + '_rejected.json')
            with open(rejected_file,'w') as json_file: json.dump(ch,json_file, indent=2)

            if P['plot']: 
                bins = np.arange(0,100)
                h,b = np.histogram(np.array(percent_rejected),bins)
                plt.plot(b[:-1],h)
                plt.show()


            print('\rRejected (' + str(int(100.*len(ch['rejected'])/num_channels)) +'%):' + 20*' ' )
            print(ch['rejected'])    
            print( 50*'-' + '\n' )




