# import packages
import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time
from nilearn import plotting 
from itertools import product
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy as sc

# import local functions
import utils.preprocessing as pr
import config as cfg
import utils.common_functions as cf
import utils.plots as pl




def get_samples(sub, band, EpochsSetups, split = None, picks = 'good', paired = False, avg = False):
    '''Get data from epochs conveniently structured

       sample = [n_tests (f.ex. one test per sentlength), 1 or 2 (samples to be compared), n_observations (n_epochs), n_channels]

    '''    

    # colect data
    epochs = [cf.load_epochs(sub, band, EpochsSetup, picks = picks) for EpochsSetup in EpochsSetups]

    # one single test         
    if split == None:
        values = None
        # avg acrross time dimension
        if avg : samples = [[e.get_data().mean(axis=-1) for e in epochs]]
        # do not avg across time
        if not avg : samples = [[e.get_data() for e in epochs]]       

    # multiple conditions
    elif isinstance(split,str):
        samples=[]
        # get categories from metadata values
        values = sorted(list(set(epochs[0].metadata[split])))
        for value in values:
            # update queries with split
            if isinstance(value,str): query = f"{split} == '{value}'"
            else: query = f"{split} == {value}"
            if avg: samples += [[e.get_data().mean(axis=-1) for e in epochs]]
            if not avg: samples += [[e.get_data() for e in epochs]]       

    # multiple time windows
    elif isinstance(split,list):
        samples=[]
        values = []
        for t1, t2 in split:                
            values +=[f"[{t1},{t2}]"]
            samples += [[e.copy().crop(e.tmin + t1,e.tmin + t2).get_data().mean(axis=-1) for e in epochs]]


    if paired:
        if len(epochs)!=2 or epochs[0].__len__()!=epochs[1].__len__():
            sys.exit('Paired samples require two epoch setups with the same number of observations.')
        samples =[s[0]-s[1] for s in samples]

    return samples, values  

      

# ==============================================================================
# T TEST
def ttest(EpochsSetups, EOIlabel, split = None, subs = cfg.subs, bands = cfg.bands, 
          picks = 'good', alpha = cfg.alpha, paired = False, PopMean = 0,
          alternative='greater', Bonferroni = False, fdr = True, figure = True):
    """ Run ttest on epochs looping over subjects and bands

    Use split to run tests on different subsets of the data
        
    Parameters
    ----------
    EpochsSetups : list of dict
    split : str or None
        separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test independently for each word position
    subs : list of str
        subject list
    bands : list of dict

    Outputs
    -------
   
    """
    print(f"\n======== t-test ==================================================\n  {EOIlabel}\n")

    from scipy.stats import ttest_ind, ttest_1samp

    # ITERATE SUBJECT LIST AND BAND LIST =======================================
    TimeStart = time.time()
    iterator =  list(product(enumerate(bands),enumerate(subs)))
    for i, ((iBand, band),(iSub, sub)) in enumerate(iterator): 

        cf.display_progress(f"{sub} {band['name']}",i,len(iterator),TimeStart)

        # load subject params
        SP = cf.sub_params(sub)

        if 'EOI' in picks and SP['ChInfo'][f"{picks}-{band['name']}"] == []: continue

        # colect data
        samples, values = get_samples(sub, band, EpochsSetups, split = split, picks = picks, 
                                      paired = paired, avg = True)

        # perform tests
        pValues = []
        # ---- 2 sample test --------
        if len(samples[0])==2:      
            for s1, s2 in samples:    
                # s1 = [n_observations (i.e. n_epochs), n_channels]
                pValues += [ttest_ind(s1-PopMean,s2,equal_var=False,alternative=alternative)[1] ]

        # ---- 1 sample test --------
        if len(samples[0])==1:     
            for s1 in samples:      
                pValues += [ttest_1samp(s1[0],PopMean,axis=0,alternative=alternative)[1] ]
        pValues = np.array(pValues)

        # Bonferroni correction
        if split != None and Bonferroni: pValues *= len(samples)
        # fdr correction
        if fdr: pValues = mne.stats.fdr_correction(pValues[:], cfg.alpha)[1].reshape(pValues.shape)
        RejectedNull = np.sum(pValues < alpha,axis=0)
        # boolean with True for significant chs
        SignificantMask = (RejectedNull > 0)  
        # List of significant ch names
        SignificantList = [ch for j, ch in enumerate(cf.get_picks(SP, picks = picks,band = band)) if SignificantMask[j]]

        # save to ChInfoFile
        SP['ChInfo'][f"{EOIlabel}-{band['name']}"]  = SignificantList
        json.dump(SP['ChInfo'], open(SP['ChInfoFile'],'w'))
        
        print(f"\r{sub} {band['name']} {len(SignificantList)}/{len(SignificantMask)} significant channels")

        if figure: ttest_figure_signle_sub(SP, band, EpochsSetups, split, samples, 
                                           values, SignificantMask, SignificantList, 
                                            PopMean, pValues, picks, alpha)

        # concatenate for each band
        if figure and iSub == len(subs)-1:
            #ttest_figure_all_subs()
            # concatenate figures
            FiguresPath = cf.check_path(['..','Figures','ttest'+cfg.out])
            FigureName = f"ttest-{EOIlabel}_band-{band['name']}.pdf"
            cf.concatenate_pdfs(FiguresPath,f'sub*temp',FigureName, remove=True)



# ==============================================================================
# T TEST FIGURE SINGLE SUB
def ttest_figure_signle_sub(SP, band, EpochsSetups, split, samples, values, SignificantMask, 
                            SignificantList, PopMean, pValues, picks, alpha):
    
    # mean and sem across epochs
    # ---- 2 sample test --------
    if len(samples[0]) == 2:
        m = np.array([[s1.mean(axis=0), s2.mean(axis=0)] for s1, s2 in samples])  
        s = np.array([[s1.std(axis=0)/np.sqrt(len(s1)), s2.std(axis=0)/np.sqrt(len(s2))] for s1, s2 in samples])  
        x = m[:,0] - m[:,1]
        xerr = np.sqrt(0.5)*(s[:,0] + s[:,1])
    # ---- 1 sample test --------
    if len(samples[0]) == 1:
        x = np.array([s1[0].mean(axis = 0) for s1 in samples])  
        xerr = np.array([s1[0].std(axis = 0)/np.sqrt(len(s1[0])) for s1 in samples])  
    
    # ---- Make Figure ----------------
    fig, ax = plt.subplots(2, 1, figsize=(5, 12), gridspec_kw = {'height_ratios' : [1, 5]})        

    title = f"T-test {SP['sub']} {band['name']} ({np.sum(SignificantMask)}/{len(SignificantMask)})"
    if 'sample' in EpochsSetups[0].keys(): title += f"\nsample: {EpochsSetups[0]['sample']}"
    fig.suptitle(title)

    # ---- plot glass brain ------------
    coords = np.array([SP['ChInfo']['coords'][SP['ChInfo']['ChNames'].index(ch)] for ch in  cf.get_picks(SP, picks = picks, band = band)])
    pl.channel_position_plot(ax[0], coords, mask = ~SignificantMask)

    # ---- scatter plot with mean and std ------
    y = np.arange(len(SignificantMask))
    if split == None: 
        ax[1].errorbar(x[0],y,xerr=xerr[0],color='k',fmt='o', alpha=0.1)
        ax[1].errorbar(x[0,SignificantMask],y[SignificantMask],
            xerr=xerr[0,SignificantMask],color='k',fmt='o')
    else:
        colors = plt.get_cmap('turbo')(np.linspace(0., 1.,  len(values)))
        for j,v in enumerate(values):
            mask = pValues[j]<alpha
            ax[1].errorbar(x[j],y,xerr=xerr[j],color=colors[j],fmt='o',alpha=0.1)
            ax[1].errorbar(x[j,mask],y[mask],xerr=xerr[j,mask],color=colors[j],fmt='o',label=f"{v}")
        if isinstance(split,list): title = 'time window'
        else: title = split
        ax[1].legend(title=title,frameon=False)

    # ---- decorate axes ------------
    ax[1].set_xlabel(f"{EpochsSetups[0]['name']} - {EpochsSetups[1]['name']} dB")
    ax[1].set_ylabel('channel')

    ax[1].set_yticks( [j for j in range(len(SignificantMask)) if SignificantMask[j]])
    ax[1].set_yticklabels(SignificantList)

    ax[1].axvline(PopMean, linestyle='-', color='grey', linewidth=0.5)

    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].tick_params(axis='both', which='both', size = 0, labelsize=7)

    # ---- save figure --------------
    FiguresPath = cf.check_path(['..','Figures','ttest'+cfg.out])
    FigureName = os.path.join(FiguresPath,f"{SP['sub']}_{band['name']}_temp.pdf")
    fig.savefig(FigureName, format='pdf', dpi=100) 
    plt.close()                            





'''

# ==============================================================================
# T TEST
def ttest(EpochsSetups, split = None, subs = cfg.subs, bands = cfg.bands, picks = None,
          alpha = cfg.alpha, EOIlabel = '', paired = False, PopMean = 0,
          alternative='greater',Bonferroni = False, fdr = True):
    """ Run ttest on epochs looping over subjects and bands

    Use split to run tests on different subsets of the data
        
    Parameters
    ----------
    EpochsSetups : list of dict
    split : str or None
        separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test independently for each word position
    subs : list of str
        subject list
    bands : list of dict

    Outputs
    -------
   
    """
    print(f"\n======== t-test {EOIlabel} =======================================\n")


    from scipy.stats import ttest_ind, ttest_1samp

    # ITERATE SUBJECT LIST AND BAND LIST =======================================
    TimeStart = time.time()
    iterator =  list(product(enumerate(bands),enumerate(subs)))
    for i, ((iBand, band),(iSub, sub)) in enumerate(iterator): 

        # load subject params
        dp = cf.sub_params(sub)

        # colect data
        epochs = []
        for EpochsSetup in EpochsSetups: epochs += [cf.load_epochs(sub, band, EpochsSetup, picks = picks)]
                       
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
                samples += [[e.copy()[query].get_data().mean(axis=-1) for e in epochs]]                

        # multiple tests
        elif isinstance(split,list):
            samples=[]
            values = []
            for t1, t2 in split:                
                values +=[f"[{t1},{t2}]"]
                samples += [[e.copy().crop(e.tmin + t1,e.tmin + t2).get_data().mean(axis=-1) for e in epochs]]
              

        if paired:
            if len(epochs)!=2 or epochs[0].__len__()!=epochs[1].__len__():
                sys.exit('Paired samples require two epoch setups with the same number of observations.')
            samples =[[s[0]-s[1]] for s in samples]


        # sample = [n_tests (f.ex. one test per sentlength), 2 (samples to be compared), n_observations (n_epochs), n_channels]
       
        # perform tests
        pValues = []
        if len(samples[0])==2:      # 2 sample test
            for s1, s2 in samples:    
                # s1 = [n_observations (i.e. n_epochs), n_channels]
                pValues += [ttest_ind(s1-PopMean,s2,equal_var=False,alternative=alternative)[1] ]
        if len(samples[0])==1:      # 1 sample test
            for s1 in samples:      
                pValues += [ttest_1samp(s1[0],PopMean,axis=0,alternative=alternative)[1] ]

        pValues = np.array(pValues)
        # Bonferroni correction
        if split != None and Bonferroni: pValues *= len(samples)
        # fdr correction
        if fdr: pValues = mne.stats.fdr_correction(pValues[:], cfg.alpha)[1].reshape(pValues.shape)
        RejectedNull = np.sum(pValues < alpha,axis=0)
        # boolean with True for significant chs
        SignificantMask = (RejectedNull > 0)  
        # List of significant ch names
        SignificantList = [ch for j, ch in enumerate(epochs[0].ch_names) if SignificantMask[j]]


        # .... save to json (try saving on existing file or make a new one) ..........................
        EOIs_file = os.path.join(dp['DerivativesPath'],f"{sub}_EOIs.json")
        try: 
            EOIs = json.load(open(EOIs_file))
            EOIs[f"EOI-{EOIlabel}-{band['name']}"]  = SignificantList
        except: 
            EOIs = {f"EOI-{EOIlabel}-{band['name']}"  : SignificantList}
        json.dump(EOIs, open(EOIs_file,'w'))
        # ............................................................................................

        
        cf.display_progress(f"{sub} {band['name']} {len(SignificantList)}/{len(SignificantMask)} significant channels",i,len(iterator),TimeStart)

        # FIGURE ---------------------------------------------------------------------------------------------

        # mean and sem across epochs
        if len(samples[0])==2:
            m = np.array([[s1.mean(axis=0),s2.mean(axis=0)] for s1, s2 in samples])  
            s = np.array([[s1.std(axis=0)/np.sqrt(len(s1)),s2.std(axis=0)/np.sqrt(len(s2))] for s1, s2 in samples])  
            x = m[:,0]-m[:,1]
            xerr=np.sqrt(0.5)*(s[:,0]+s[:,1])

 
        if len(samples[0])==1:
            x = np.array([s1[0].mean(axis=0) for s1 in samples])  
            xerr = np.array([s1[0].std(axis=0)/np.sqrt(len(s1[0])) for s1 in samples])  
        
        fig, ax = plt.subplots(2,1,figsize=(5,12),gridspec_kw={'height_ratios': [1, 5]})        

        title = f"T-test {sub} {band['name']} ({np.sum(SignificantMask)}/{len(SignificantMask)})"
        if 'sample' in EpochsSetups[0].keys(): title += f"\nsample: {EpochsSetups[0]['sample']}"
        fig.suptitle(title)


        pl.channel_position_plot(ax[0],cf.get_coords(epochs[0]),mask=~SignificantMask)

        
        y = np.arange(len(SignificantMask))
        if split == None: 

            ax[1].errorbar(x[0],y,xerr=xerr[0],color='k',fmt='o', alpha=0.1)
            ax[1].errorbar(x[0,SignificantMask],y[SignificantMask],
                xerr=xerr[0,SignificantMask],color='k',fmt='o')

        else:

            colors = plt.get_cmap('turbo')(np.linspace(0., 1.,  len(values)))
            for j,v in enumerate(values):
                mask = pValues[j]<alpha

                ax[1].errorbar(x[j],y,xerr=xerr[j],color=colors[j],fmt='o',alpha=0.1)
                ax[1].errorbar(x[j,mask],y[mask],xerr=xerr[j,mask],color=colors[j],fmt='o',label=f"{v}")
            if isinstance(split,list): title = 'time window'
            else: title = split
            ax[1].legend(title=title,frameon=False)


        ax[1].set_xlabel(f"{EpochsSetups[0]['name']} - {EpochsSetups[1]['name']} dB")
        ax[1].set_ylabel('channel')

        ax[1].set_yticks( [j for j in range(len(SignificantMask)) if SignificantMask[j]])
        ax[1].set_yticklabels(SignificantList)

        ax[1].axvline(PopMean, linestyle='-', color='grey', linewidth=0.5)

        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].tick_params(axis='both', which='both', size = 0, labelsize=7)

        # save figure
        FiguresPath = cf.check_path(['..','Figures','ttest'+cfg.out])
        FigureName = os.path.join(FiguresPath,f"{sub}_{band['name']}_temp.pdf")
        fig.savefig(FigureName, format='pdf', dpi=100) 
        plt.close()                            

        # concatenate for each band
        if iSub == len(subs)-1:
            # concatenate figures
            FiguresPath = cf.check_path(['..','Figures','ttest'+cfg.out])
            FigureName = f"ttest-{EOIlabel}_band-{band['name']}.pdf"
            cf.concatenate_pdfs(FiguresPath,f'sub*temp',FigureName, remove=True)








def PCT(EpochsSetup,subs=cfg.subs, bands=cfg.bands,alpha=cfg.alpha,picks=None,EOIlabel='PCT'):
    """ Run Permutation Cluster Test on epochs
    looping over subjects and bands
    """

    print("\n======== permutation cluster test ======================================")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    TimeStart = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((iSub, sub), (iBand, band)) in enumerate(iterator): 

        if iBand==0:
            # load subject params
            dp = cf.sub_params(sub)

        # load epochs
        epochs = cf.load_epochs(sub,band,EpochsSetup,picks=picks)

        sample = epochs.get_data()
        # standardize data (zscore channelwise)
        sample = (sample-sample.mean(axis=(0,-1))[np.newaxis,:,np.newaxis])/sample.std(axis=(0,-1))[np.newaxis,:,np.newaxis]
        #sample = (sample-sample.mean(axis=(-1))[:,:,np.newaxis])/sample.std(axis=(-1))[:,:,np.newaxis]
        sample= np.swapaxes(sample,-1,-2)
  
        # dummy adjacency matrix to run all channels at one as independent tests
        #adjacency = sc.sparse.csr_matrix(np.eye(sample.shape[-1]))

        # adjacency matrix
        distance = sc.spatial.distance_matrix(cf.get_coords(epochs),cf.get_coords(epochs))
        fnmd = np.nanmedian(np.sort(distance,axis=0)[1,:])
        adjacency = sc.sparse.csr_matrix(distance<cfg.s_adj*fnmd)

        # perform PCT
        T_obs, clusters, pValues, H0 = mne.stats.permutation_cluster_1samp_test(sample, n_permutations=cfg.n_permutations, out_type='mask',adjacency=adjacency,max_step=int(cfg.t_adj*cfg.srate),check_disjoint=True)           

        # Create mask with only non SignificantList data
        mask = np.ones_like(T_obs)   
        for c, p_val in zip(clusters, pValues):
            if p_val <= alpha:
                mask[c] = np.nan*T_obs[c]

        # use maks to identify SignificantList channels
        SignificantMask = np.sum(np.isnan(mask.T),axis=-1)>0
        SignificantList = [ch for j, ch in enumerate(epochs.ch_names) if SignificantMask[j]]
        
        # .... save to json (try saving on existing file or make a new one) ..........................
        EOIs_file = os.path.join(dp['DerivativesPath'],f"{sub}_EOIs.json")
        try: 
            EOIs = json.load(open(EOIs_file))
            EOIs[f"EOI-{EOIlabel}-{band['name']}"]  = SignificantList
        except: 
            EOIs = {f"EOI-{EOIlabel}-{band['name']}"  : SignificantList}
        json.dump(EOIs, open(EOIs_file,'w'))
        # ............................................................................................


        cf.display_progress(f"{sub} {band['name']} {len(SignificantList)}/{len(epochs.ch_names)} SignificantList channels",i,len(iterator),TimeStart)

        # avg activity across epochs
        avg = sample.mean(axis=0)  
        avg = (avg-avg.mean())/avg.std()
        vmin, vmax = np.percentile(avg,[15,85])

        # figure ------------------------------------------------------------------------------
        fig, ax = plt.subplots(2,1,figsize=(5,12),gridspec_kw={'height_ratios': [1, 5]})        
        fig.suptitle(f"PCT {sub} {band['name']} ({len(SignificantList)}/{len(epochs.ch_names)})")

        pl.channel_position_plot(ax[0],cf.get_coords(epochs),mask=~SignificantMask)
        pl.imshow_plot(ax[1], avg.T, title = 'avg across epochs', xlabel = 't (s)', ylabel = 'channels',  
                vmin = vmin, vmax = vmax, xlims = epochs.times, ylims = range(len(epochs.ch_names)),
                yticks = [j for j in range(len(epochs.ch_names)) if SignificantMask[j]], yticklabels = SignificantList,
                colorbar = 'dB (zscored)',cmap='RdBu_r')

        ax[1].imshow(mask.T,
                   extent=[epochs.times[0], epochs.times[-1], 0, len(epochs.ch_names)],
                   aspect='auto', origin='lower', cmap='gray',vmin=-1,vmax=0,alpha=0.4,interpolation=None)

        # save figure
        FiguresPath = cf.check_path(['..','Figures','PCT'+cfg.out])
        FigureName = os.path.join(FiguresPath,f"{sub}_{band['name']}_temp.pdf")
        fig.savefig(FigureName, format='pdf', dpi=100) 
        plt.close()

    # concatenate single channel figures for sub
    FiguresPath = cf.check_path(['..','Figures','PCT'+cfg.out])
    FigureName = f"PCT-{EpochsSetup['name']}.pdf"
    cf.concatenate_pdfs(FiguresPath,f'sub*temp',FigureName, remove=True)






'''










