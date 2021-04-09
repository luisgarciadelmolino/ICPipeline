# import packages
import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time
from nilearn import plotting 
from itertools import product
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# import local functions
import utils.preprocessing as pr
import config as cfg
import utils.common_functions as cf
import utils.plots as pl


def PCT(ep_setup,subs=cfg.subs, bands=cfg.bands,alpha=cfg.alpha,picks=None,EOIlabel='PCT'):
    """ Run Permutation Cluster Test on epochs
    looping over subjects and bands

    either 1 sample test if len(queries['list']) == 1 
    or 2 sample tests if len(queries['list']) == 2 
        
    Parameters
    ----------
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    subs : list of str
        subject list
    bands : list of dict

    Outputs
    -------
    df : dataframe 
        with columns 'sub', 'band', 'ch_name', 'coords', 
        'p' (min p of all clusters found), 'tw' (time window when the minimal cluster was found) 
        stored as csv @ Data/derivatives/sub-XX/sub-XX_PCT-{queries['tag']}.csv
        
    """

    # from scipy.spatial import distance_matrix
    from scipy import sparse

    print("\n======== permutation cluster test ======================================")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 

        if i_band==0:
            # load subject params
            dp = cf.sub_params(sub)

        # load epochs
        epochs = cf.load_epochs(sub,band,ep_setup,picks=picks)

        sample = epochs.get_data()
        # standardize data (zscore channelwise)
        sample = (sample-sample.mean(axis=(0,-1))[np.newaxis,:,np.newaxis])/sample.std(axis=(0,-1))[np.newaxis,:,np.newaxis]
        sample= np.swapaxes(sample,-1,-2)
  
        # dummy adjacency matrix to run all channels at one as independent tests
        adjacency = sparse.csr_matrix(np.eye(sample.shape[-1]))
        # perform PCT
        T_obs, clusters, p_values, H0 = mne.stats.permutation_cluster_1samp_test(sample, n_permutations=cfg.n_permutations, out_type='mask',adjacency=adjacency)           

        # Create mask with only non significant data
        mask = np.ones_like(T_obs)   
        for c, p_val in zip(clusters, p_values):
            if p_val <= alpha:
                mask[c] = np.nan*T_obs[c]

        # use maks to identify significant channels
        mask_channels = np.sum(np.isnan(mask.T),axis=-1)>0
        significant = [ch for j, ch in enumerate(epochs.ch_names) if mask_channels[j]]
        
        # .... save to json (try saving on existing file or make a new one) ..........................
        EOIs_file = os.path.join(dp['derivatives_path'],f"{sub}_EOIs.json")
        try: 
            EOIs = json.load(open(EOIs_file))
            EOIs[f"EOI-{EOIlabel}-{band['name']}"]  = significant
        except: 
            EOIs = {f"EOI-{EOIlabel}-{band['name']}"  : significant}
        json.dump(EOIs, open(EOIs_file,'w'))
        # ............................................................................................


        cf.display_progress(f"{sub} {band['name']} {len(significant)}/{len(epochs.ch_names)} significant channels",i,len(iterator),time_start)

        # avg activity across epochs
        avg = sample.mean(axis=0)  
        avg = (avg-avg.mean())/avg.std()
        vmin, vmax = np.percentile(avg,[15,85])

        # figure ------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(2,1,figsize=(5,12),gridspec_kw={'height_ratios': [1, 5]})        
        fig.suptitle(f"PCT {sub} {band['name']} ({len(significant)}/{len(epochs.ch_names)})")

        # in case there are no significant channels
        try: pl.channel_position_plot(ax[0],cf.get_coords(epochs,picks=significant))
        except: pass

        pl.imshow_plot(ax[1], avg.T, title = 'avg across epochs', xlabel = 't (s)', ylabel = 'channels',  
                vmin = vmin, vmax = vmax, xlims = epochs.times, ylims = range(len(epochs.ch_names)),
                yticks = [j for j in range(len(epochs.ch_names)) if mask_channels[j]], yticklabels = significant,
                colorbar = 'dB (zscored)',cmap='RdBu_r')

        ax[1].imshow(mask.T,
                   extent=[epochs.times[0], epochs.times[-1], 0, len(epochs.ch_names)],
                   aspect='auto', origin='lower', cmap='gray',vmin=-1,vmax=0,alpha=0.4,interpolation=None)

        # save figure
        fig_path = cf.check_path(['..','Figures','PCT'+cfg.out])
        fig_name = os.path.join(fig_path,f"{sub}_{band['name']}_temp.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
        plt.close()

    # concatenate single channel figures for sub
    fig_path = cf.check_path(['..','Figures','PCT'+cfg.out])
    fig_name = f"PCT-{ep_setup['name']}.pdf"
    cf.concatenate_pdfs(fig_path,f'sub*temp',fig_name, remove=True)


      

# ====================================================================================
# T TEST
def ttest(ep_setups,split=None,subs=cfg.subs, bands=cfg.bands,alpha=cfg.alpha,picks=None,EOIlabel='ttest'):
    """ Run ttest on epochs looping over subjects and bands

    Use split to run tests on different subsets of the data
        
    Parameters
    ----------
    ep_setups : list of dict
    split : str or None
        separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test independently for each word position
    subs : list of str
        subject list
    bands : list of dict

    Outputs
    -------
   
    """
    print("\n======== t-test =================================================")

    from scipy.stats import ttest_ind

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 

        if i_band==0:
            # load subject params
            dp = cf.sub_params(sub)

        # colect data
        epochs = []
        for ep_setup in ep_setups: epochs += [cf.load_epochs(sub,band,ep_setup,picks=picks)]
                       
        # one single test         
        if split== None:
            values = None
            samples = [[e.get_data().mean(axis=-1) for e in epochs]]
        
        # multiple tests
        elif isinstance(split,str):
            samples=[]
            values = sorted(list(set(epochs[0].metadata[split])))
            for value in values:
                # update queries with split
                if isinstance(value,str): query = f"{split} == '{value}'"
                else: query = f"{split} == {value}"
                samples += [[e[query].get_data().mean(axis=-1) for e in epochs]]
 
        # sample = [n_tests (f.ex. one test per sentlength), 2 (samples to be compared), n_observations (n_epochs), n_channels]
       
        # perform tests
        p_values = []
        print(len(samples))
        for s1, s2 in samples:    
            # s1, s2 = [n_observations (n_epochs), n_channels]
            # skip if epochs are empty
            # if len(sample[0])*len(sample[1]) == 0: continue
            p_values += [ttest_ind(s1,s2,equal_var=False,alternative='greater')[1] ]

        p_values = np.array(p_values)
        positive_tests = np.sum(p_values<alpha,axis=0)
        mask_channels = (positive_tests>0)
        significant = [ch for j, ch in enumerate(epochs[0].ch_names) if mask_channels[j]]

        # .... save to json (try saving on existing file or make a new one) ..........................
        EOIs_file = os.path.join(dp['derivatives_path'],f"{sub}_EOIs.json")
        try: 
            EOIs = json.load(open(EOIs_file))
            EOIs[f"EOI-{EOIlabel}-{band['name']}"]  = significant
        except: 
            EOIs = {f"EOI-{EOIlabel}-{band['name']}"  : significant}
        json.dump(EOIs, open(EOIs_file,'w'))
        # ............................................................................................

        
        cf.display_progress(f"{sub} {band['name']} {len(significant)}/{len(mask_channels)} significant channels",i,len(iterator),time_start)

        # FIGURE ---------------------------------------------------------------------------------------------

        # mean and sem across epochs
        m = np.array([[s1.mean(axis=0),s2.mean(axis=0)] for s1, s2 in samples])  
        s = np.array([[s1.std(axis=0)/np.sqrt(len(s1)),s2.std(axis=0)/np.sqrt(len(s2))] for s1, s2 in samples])  

        
        fig, ax = plt.subplots(2,1,figsize=(5,12),gridspec_kw={'height_ratios': [1, 5]})        
        fig.suptitle(f"T-test {sub} {band['name']} ({np.sum(mask_channels)}/{len(mask_channels)})")

        
        y = np.arange(len(mask_channels))
        if split == None: 
            pl.channel_position_plot(ax[0],cf.get_coords(epochs,picks=significant))

            x = m[0,0]-m[0,1]
            xerr=np.sqrt(0.5)*(s[0,0]+s[0,1])

            ax[1].errorbar(x,y,xerr=xerr,color='k',fmt='o', alpha=0.1)
            ax[1].errorbar(x[mask_channels],y[mask_channels],xerr=xerr[mask_channels],color='k',fmt='o')

        else:
            pl.brain_plot(ax[0],cf.get_coords(epochs[0]),positive_tests,colorbar='# rejected H0',mask=~mask_channels)

            colors = plt.get_cmap('rainbow')(np.linspace(0., 1.,  len(values)))
            for j,v in enumerate(values):
                mask = p_values[j]<alpha
                x = m[j,0]-m[j,1]
                xerr=np.sqrt(0.5)*(s[j,0]+s[j,1])

                ax[1].errorbar(x,y,xerr=xerr,color=colors[j],fmt='o',alpha=0.1)
                ax[1].errorbar(x[mask],y[mask],xerr=xerr[mask],color=colors[j],fmt='o',label=f"{v}")

            ax[1].legend(title=split,frameon=False)


        ax[1].set_xlabel(f"{ep_setups[0]['name']} - {ep_setups[1]['name']} dB")
        ax[1].set_ylabel('channel')

        ax[1].set_yticks( [j for j in range(len(mask_channels)) if mask_channels[j]])
        ax[1].set_yticklabels(significant)

        ax[1].axvline(0, linestyle='-', color='grey', linewidth=0.5)

        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].tick_params(axis='both', which='both', size = 0, labelsize=7)

        plt.show()




        # save figure
        fig_path = cf.check_path(['..','Figures','ttest'+cfg.out])
        fig_name = os.path.join(fig_path,f"{sub}_{band['name']}_temp.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
        plt.close()                            

    # concatenate single channel figures for sub
    fig_path = cf.check_path(['..','Figures','ttest'+cfg.out])
    fig_name = f"ttest-{ep_setups[0]['name']}_{setups[1]['name']}.pdf"
    cf.concatenate_pdfs(fig_path,f'sub*temp',fig_name, remove=True)




                            
'''
# ====================================================================================
# PERMUTATION CLUSTER TEST       
def PCT_wrap(model,subs=cfg.subs, bands=cfg.bands,key=None):
    """ Run Permutation Cluster Test on epochs
    looping over subjects and bands

    either 1 sample test if len(queries['list']) == 1 
    or 2 sample tests if len(queries['list']) == 2 
        
    Parameters
    ----------
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    subs : list of str
        subject list
    bands : list of dict

    Outputs
    -------
    df : dataframe 
        with columns 'sub', 'band', 'ch_name', 'coords', 
        'p' (min p of all clusters found), 'tw' (time window when the minimal cluster was found) 
        stored as csv @ Data/derivatives/sub-XX/sub-XX_PCT-{queries['tag']}.csv
        
    """

    print("\n======== permutation cluster test ======================================")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 

        if i_band==0:
            # load subject params
            dp = cf.sub_params(sub)

        # colect data in a dict / dataframe
        df = pd.DataFrame(columns=['sub','ch_name','coords','band','p','tw'])  

        # raw file
        mneraw_file = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_raw.fif")
        raw = mne.io.read_raw_fif(mneraw_file,preload=True)

        # iterate over channels ---------------------------------------------------
        time_start = time.time() 
        for i_ch, ch_name in enumerate(dp['ch_names']):
            cf.display_progress(f"{sub}, {band['name']} {ch_name}", i_ch, len(dp['ch_names']),time_start) 
            coords = dp['coords'][dp['ch_names'].index(ch_name)]

            # file for a given channel might not exist
            try:
                epochs = cf.load_epochs(dp, raw.copy().pick_channels([ch_name]), model['ep_setup'])
            except: 
                continue

            # crop epochs
            if len(tw)>1: epochs.crop(tw[0],tw[-1])                                 

            # permutation statistics for 1 sample
            if len(model['queries'])==1:
                # extract sample using query
                sample = epochs[model['queries'][0]].get_data()[:,0,:] 
                # remove the mean
                sample -= sample.mean()
                # perform PCT
                t_clust, clusters, p_values, H0 = mne.stats.permutation_cluster_1samp_test(sample, n_permutations=cfg.n_permutations, out_type='mask')           

            # permutation statistics for 2 sample
            elif len(model['queries'])==2:
                # extract samples using query
                sample = [epochs[q].get_data()[:,0,:] for q in model['queries']]
                t_clust, clusters, p_values, H0 = mne.stats.permutation_cluster_test(sample, n_permutations=cfg.n_permutations, out_type='mask')           

            # save single channel results to df
            if len(p_values) == 0 : 
                p_value, twdf = 1., [0,0]
            else:
                p_value = np.min(p_values)
                i_c = np.argmin(p_values)
                twdf = [epochs.times[clusters[i_c][0].start], epochs.times[clusters[i_c][0].stop - 1]]
                
            df = df.append({ 'sub': sub,
                        'ch_name' : ch_name,
                        'coords' : coords,
                        'band' : band['name'],
                        'p' : p_value,
                        'tw' : twdf}, ignore_index=True)
            # go to next channel ------------------------------------------

        # save dataframe
        filename = os.path.join(dp['derivatives_path'], f"{sub}_band-{band['name']}_PCT-{model['tag']}.csv")
        df.to_csv(filename) 
        # go to next band / subject ======================================================
    



def PCT_figures_single_channel(sub,ch_name,i_ch,band,coords,queries,sample,times,tw,p):
    """Plot single channel figure for permutation test

    figure contains channel position panel, sample average panel 
    and statistics panel

    Parameters
    ----------
    sub : str
        'sub-XX' 
    ch_name : str
        channel name
    i_ch : int
        channel index in epochs
    i_band : int
        index of band in cfg.bands
    queries : list of str
        cotaining 1 or two queries to compare
    times : np.array
        time points from epochs
    sample : list of mne epochs

    """


    fig, ax = plt.subplots(2,1,figsize=(5,3),sharex=True)        
    fig.suptitle(f"PCT {sub} {ch_name} {band['name']}")

    pl.channel_position_plot(ax[0],[coords],0)
    if len(sample)==1:
        pl.trace_plot(ax[1],times, sample[0], ylabel='avg power', xticks=np.arange(0,1.2*cfg.soa,0.5*cfg.soa), plot_sem = True)
    elif len(sample)==2:
        pl.trace_plot(ax[1],times, sample[0], ylabel='avg across epochs', xticks=np.arange(0,1.2*cfg.soa,0.5*cfg.soa), plot_sem = True, color='blue',label=queries[0])
        pl.trace_plot(ax[1],times, sample[1], ylabel='avg across epochs', xlabel='t (s)', xticks=np.arange(0,1.2*cfg.soa,0.5*cfg.soa), plot_sem = True, color='green',label=queries[1],legend=True)
        ax[1].legend(frameon=False,loc =(0.7,0.8),fontsize=8)  


    ax[1].text(tw[0],ax[1].get_ylim()[1],f"p={p}",fontsize=7)
    ax[1].axvspan(tw[0], tw[1], color='gray', alpha=0.3,lw=0)
      


    PCT_path = cf.check_path(['..','Figures','PCT'+ cfg.out])
    fig_name = os.path.join(PCT_path,f"{sub}_{ch_name}_{band['name']}_temp.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    plt.close()








# ====================================================================================
# T TEST
def ttest(queries,tw=[],split=None,subs=cfg.subs, bands=cfg.bands):
    """ Run ttest on epochs looping over subjects and bands

    Use split to run tests on different subsets of the data
        
    Parameters
    ----------
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    tw : list of float
        time window where to run the test in seconds, 
        f.ex. [-0.1,0.5] (from -100ms to 500ms)
        if tw = [] use full epochs
    split : int or str
        if int divide the time window into 'split' chunks and run one test in each
        if string separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test for each word position
    subs : list of str
        subject list
    bands : list of dict

    Outputs
    -------
    df : dataframe 
        with columns 'sub', 'band', 'ch_name', 'coords', 'p' (min p of all clusters found), 
        'tw' (time window when the minimal cluster was found), 'split' (metadata[split] value)
        stored as csv @ Data/derivatives/sub-XX/sub-XX_ttest-{queries['tag']}.csv
        
    """
    print("\n======== t-test =================================================")

    from scipy.stats import ttest_ind

    # iterate over subjects =================================================
    for sub in subs:
        # load subject params
        dp = cf.sub_params(sub)
          
        # colect data in a dict / dataframe (one p value for each time window)
        df = pd.DataFrame(columns=['sub','ch_name','coords','band','p','tw','means','sem'])
          
        # iterate over bands -----------------------------------------------
        for i_band, band in enumerate(bands):
            cf.print_d(f"Colletcting {band['name']} band data for {sub}")

            # load epochs            
            fname = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_level-word_epo.fif")
            epochs = mne.read_epochs(fname, preload=True)

            # crop epochs
            if len(tw)>1: epochs.crop(tw[0],tw[-1])                     
                                
                                 
            # colect data from epochs
            times = epochs.times   
            ch_names = epochs.ch_names  
    
            # iterate channels --------------------------------------------
            time_start = time.time()
            for i_ch, ch_name in enumerate(ch_names):
                cf.display_progress(f"{sub} {band['name']} {ch_name}", i_ch, len(ch_names), time_start)
                coords = dp['coords'][dp['ch_names'].index(ch_name)]
                
                if isinstance(split,int):     
                    # num of time windows                
                    l = int(len(times)/split)  
                    # loop time windows
                    for i in range(split):
                        # extract samples using query
                        t1, t2 = i*l, min((i+1)*l,len(times)-1)      # time window min and max
                        sample = [epochs[q].get_data()[:,i_ch,t1:t2].mean(axis=-1) for q in queries['mne']]
                        # skip if epochs are empty
                        if len(sample[0])*len(sample[1]) == 0: continue
                        p = ttest_ind(sample[0],sample[1],equal_var=False)[1] 
                        twdf = [round(times[t1],3),round(times[t2],3)]
                        means = [s.mean() for s in sample]
                        sem = [s.std()/np.sqrt(s.shape[0]) for s in sample]
                        dic = { 'sub': sub, 'ch_name' : ch_name, 'coords' : coords, 
                               'band' : band['name'], 'p':p, 'tw':twdf, 'means':means, 'sem': sem}
                        df = df.append(dic, ignore_index=True)
                                    
                if isinstance(split,str): 
                    # loop over values of metadata column
                    for value in list(set(epochs.metadata[split])):
                        # update queries with split
                        if isinstance(value,str): q2 = f' and {split} == "{value}"'
                        else: q2 = f' and {split} == {value}'
                        sample = [epochs[q + q2].get_data()[:,i_ch,:].mean(axis=-1) for q in queries['mne']]
                        # skip if epochs are empty
                        if len(sample[0])*len(sample[1]) == 0: continue
                        p = ttest_ind(sample[0],sample[1],equal_var=False)[1] 
                        twdf = [times[0],times[-1]]
                        means = [s.mean() for s in sample]
                        sem = [s.std()/np.sqrt(s.__len__()) for s in sample]
                        dic = { 'sub': sub, 'ch_name' : ch_name, 'coords' : coords, 
                               'band' : band['name'], 'p':p, 'tw':twdf, f'{split}': value, 'means':means, 'sem': sem}
                        df = df.append(dic, ignore_index=True)
                                    
                                    
                                    
                # go to next channel ------------------------------------------

        # fdr correction
        df['p_fdr'] = mne.stats.fdr_correction(df['p'],cfg.alpha)[1]


        # save dataframe
        filename = os.path.join(dp['derivatives_path'], f"{sub}_ttest-{queries['tag']}.csv")
        df.to_csv(filename) 
                            
                            
                            
                            



def ttest_figures_single_channel(sub,ch_name,i_ch, band,coords,queries,df,ps,split=None,alpha=cfg.alpha):
    """Plot single channel figure for permutation test

    figure contains channel position panel, sample average panel 
    and statistics panel

    Parameters
    ----------
    sub : str
        'sub-XX' 
    ch_name : str
        channel name
    i_ch : int
        channel index in epochs
    i_band : int
        index of band in cfg.bands
    queries : list of str
        cotaining 1 or two queries to compare
    times : np.array
        time points from epochs
    sample : list of mne epochs

    """


    fig, ax = plt.subplots(2,1,figsize=(5,3),sharex=True)        
    fig.suptitle(f"ttest {sub} {ch_name} {band['name']}")

    pl.channel_position_plot(ax[0],[coords],0)
    

    # width and positions of bars
    width = 0.3
    x = np.arange(len(ps))  
    
    means = np.array([m for m in df['means']]).T
    sem = np.array([m for m in df['sem']]).T


    ax[1].bar(x , means[0],yerr=sem[0], width= width,  label=queries[0], color = 'navy')
    ax[1].bar(x + width, means[1],yerr=sem[1], width=width,  label=queries[1], color = 'firebrick')

    if split == None:
        ax[1].set_xticks([])
    elif isinstance(split,int):
        ax[1].set_xticks(np.arange(len(ps)) + width*0.5)
        ax[1].set_xticklabels([f"{tw} s" for tw in df['tw']])
    elif isinstance(split,str):
        ax[1].set_xticks(np.arange(len(ps)) + width*0.5)
        ax[1].set_xticklabels([f"{s}" for s in df[split]])
        ax[1].set_xlabel(f"{split}")

    # asterisks for significance
    for i_bar, p in enumerate(ps):
        if p<alpha: ax[1].text(i_bar + 0.5*width,ax[1].get_ylim()[1],'*',fontsize=12)

    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].tick_params(axis='both', which='both', size = 0, labelsize=9)


    ax[1].legend(frameon = False, loc = (1,0.2),fontsize=9)

    fig.subplots_adjust(right=0.8)

    ttest_path = cf.check_path(['..','Figures','ttest'+ cfg.out])
    fig_name = os.path.join(ttest_path,f"{sub}_{ch_name}_{band['name']}_temp.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    plt.close()






# ====================================================================================
# COMMON
       
                                    

def single_channel_figures_wrap(key, test, queries,tw=[], split=None, alpha=cfg.alpha, subs=cfg.subs, bands=cfg.bands):
    """ Wrap for PCT figures, basically loops over bands and subs,
    collects epochs data and call PTC_figures_single_channel

    Parameters
    ----------
    thr : float
        threshold p value to plot 
    same as  PTC 
 
    """

    print(f"\n{test} figures for {queries['tag']}")

    df = cf.load_df(f"{test}-{queries['tag']}")

    # iterate over subjects =================================================
    for sub in subs:
        # load subject params
        dp = cf.sub_params(sub)

        # iterate over bands -----------------------------------------------
        for i_band, band in enumerate(bands):
            cf.print_d(f"Colletcting {band['name']} band data for {sub}")

            # load epochs            
            fname = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_level-word_epo.fif")
            epochs = mne.read_epochs(fname, preload=False)   
                                 
            # colect data from epochs          
            ch_names = epochs.ch_names  


            # iterate channels --------------------------------------------
            time_start = time.time()
            for i_ch, ch_name in enumerate(ch_names):
                cf.display_progress(f"{sub} {band['name']} {ch_name}", i_ch, len(ch_names), time_start)

                coords = dp['coords'][dp['ch_names'].index(ch_name)]

                df_s = df[(df['sub']==sub) & (df['ch_name'] == ch_name) & (df['band'] == band['name'])]
                
                # skip if not significant
                
                if np.min(df_s[key].values)> alpha: continue
                if test == 'PCT':              
                    p, twdf = df_s['p'].iloc[0], df_s['tw'].iloc[0]
                        
                    # load and crop epochs
                    print(' ')
                    cf.print_d('load data')
                    epochs.load_data()
                    if len(tw)>1: epochs.crop(tw[0],tw[-1])          

                    # extract samples using query
                    sample = [epochs[q].get_data()[:,i_ch,:] for q in queries['mne']]
                    cf.print_d('fig')
                                               
                    PCT_figures_single_channel(sub, ch_name, i_ch, band, coords, queries['list'], sample, epochs.times, twdf, p)
                
                if test =='ttest':
                    ttest_figures_single_channel(sub, ch_name,i_ch, band, coords, queries['list'], df_s, df_s[key],split=split)

                # go to next channel ------------------------------------------

        # concatenate single channel figures for sub
        path = cf.check_path(['..','Figures',test+ cfg.out])
        fig_name = f"{sub}_{test}-{queries['tag']}.pdf"
        cf.concatenate_pdfs(path,f'{sub}*temp',fig_name, remove=True)
            
            # go to next band -------------------------------------------------- 
        # go to next subject ======================================================


           
      
def test_maps(model,test,subs=cfg.subs,bands=cfg.bands,alpha=cfg.alpha,fdr=False, plot = True, save = True):
    """ Plot brain maps test results

    Parameters
    ----------
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """


    print(f"\nBand comparison brain map for test {model['tag']}")

    df = cf.load_df(f"{test}-{model['tag']}", subs=subs, bands=bands)
 
    fig, ax = plt.subplots(len(bands),1, figsize=(10,4*len(bands)))
    if len(bands)==1: ax = np.array([ax])

    fig.suptitle(fr"{test} {model['tag']}, $p\leq {alpha}$",fontsize=20)

    for i_band, band in enumerate(bands):
        df_band = df[df['band']==band['name']]

        if fdr: df_band = cf.fdr_correction(df_band,'p')
    

        if len(df_band)==0: continue
        # extract data to plot
        x = -np.log10(df_band['p'])
            
        title=f"Significant {np.sum(df_band['p']<alpha)}/{len(df_band)} {int(100*np.sum(df_band['p']<alpha)/len(df_band))}%"

        pl.brain_plot(ax[i_band],cf.eval_coords(df_band['coords']),x, title=title,
                       ylabel=band['name'], mask = (df_band['p']>alpha),colorbar='' + (i_band==(len(bands)-1))*'$-\log_{10}(p)$',
               	         mode='interval',interval=[1.5,3])        

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8, wspace=0.3, hspace=0.4)
    if save: 
        fig_name = os.path.join(cf.check_path(['..','Figures', test+cfg.out]),f"brain_map_{test}_{model['tag']}.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()

'''











