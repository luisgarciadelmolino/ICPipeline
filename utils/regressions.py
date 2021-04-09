import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time, scipy
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting  
from sklearn import linear_model 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from itertools import product
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config as cfg
import utils.common_functions as cf
import utils.plots as pl
import utils.preprocessing as pr



def regression(x, y):
    """ Perform RidgeCV regression with outer CV
      
    Parameters
    ----------
    y : np array  
        array of responses
        (n_samples,) or (n_samples, n_targets)
    x : np array
        array of predictor values
        (n_samples, n_features)
   
    Returns
    -------
    scores : np array
        (n_targets)
    ps : np array
        (n_targets)
    scores_cv : np array
        (n_splits,n_targets)
    """

    # cross-validation fold
    cv = KFold(n_splits=cfg.n_splits, random_state=0, shuffle=True)

    # standardize data
    y = StandardScaler().fit_transform(y)
    x = StandardScaler().fit_transform(x)

    scores_cv = []
    # Compute scores for each cross-validation fold
    for train, test in cv.split(x, y):
        model = linear_model.RidgeCV(alphas=(0.01,0.1,1.,10.))#,100.,1000.))
        # Fit
        model.fit(x[train], y[train])

        # compute scores from predictions
        u = ((model.predict(x[test])-y[test])**2).sum(axis=0)
        v = ((y[test]-y[test].mean(axis=0))**2).sum(axis=0)
        scores_cv += [1 - u/v]

    scores_cv = np.array(scores_cv)

    # avg and p value across outer cv
    scores = scores_cv.mean(axis=0)
    ps = stats.ttest_1samp(scores_cv, 0, axis=0, alternative='greater')[1]

    return scores, ps, scores_cv





def stepwise_regression(epochs,predictors,sw_predictors,alpha=cfg.alpha):
    """         
    Parameters
    ----------
    epochs : mne epochs  
    predictors : list of string
        predictors for full model
    sw_predictors : list of string
        predictors to remove, has to be contained in predictors

    Returns
    -------
    scores : np array
        (n_channels,n_timepoints,1 + n_sw_predictors)
    ps : np array
        (n_channels,n_timepoints,1 + n_sw_predictors)
    """

    # data for regressions -------------------------------------------------
    # predictor values 
    x = np.array(epochs.metadata[predictors])
    # y true values, all channels, all timepoints at once
    # dims (n_epochs, n_ch x n_timpoints) = (n_samples, n_targets)
    y = epochs.get_data().reshape(epochs.__len__(),-1)

    # indices of features for stepwise
    idx = [predictors.index(p) for p in sw_predictors]
    
    # containers for results (one row for full regression plus one per sw feature)
    # dims (1 + n_sw_predictors, n_ch x n_timepoints)
    scores = np.zeros((1 + len(idx),y.shape[-1]))
    ps = np.ones((1 + len(idx),y.shape[-1]))

    # run full regression --------------------------------------------------
    scores[0], ps[0], scores_cv_full = regression(x,y)

    # fdr correction, include only p<1
    if cfg.fdr: ps[0,ps[0]<1] = mne.stats.fdr_correction(ps[0,ps[0]<1],alpha)[1]

    # indices of significant regressions
    s = (ps[0]<alpha)*(scores[0]>0)

    # if there are no significant regressions skip ablated models
    if all(~s): 
        scores = scores.reshape(1 + len(idx),len(epochs.ch_names),len(epochs.times))
        ps = ps.reshape(1 + len(idx),len(epochs.ch_names),len(epochs.times))
        return scores, ps
        
    # run ablated regressions (only for significant full regression)
    for i, i_f in enumerate(idx):
        # remove feature
        x_abl = np.delete(x, i_f, axis=1)
        _, _, scores_cv = regression(x_abl,y[:,s])

        # feature differential score for each cv fold
        dif_scores = scores_cv_full[:,s] - scores_cv

        # avg and p value across cv folds
        scores[i+1,s] = dif_scores.mean(axis=0)
        ps[i+1,s] = stats.ttest_1samp(dif_scores, 0, axis=0, alternative='greater')[1]

        # fdr correction
        if cfg.fdr: ps[i+1] = mne.stats.fdr_correction(ps[i+1],alpha)[1]


    scores = scores.reshape(1 + len(idx),len(epochs.ch_names),len(epochs.times))
    ps = ps.reshape(1 + len(idx),len(epochs.ch_names),len(epochs.times))

    return scores, ps




def regression_wrap(model,subs=cfg.subs, bands=cfg.bands, alpha=cfg.alpha):

    print("\n======== regressions ===================================================\n")

    # ITERATE SUBJECT LIST AND BAND LIST ===============================================
    time_start = time.time()
    iterator =  list(product(enumerate(subs),enumerate(bands)))
    for i, ((i_sub, sub), (i_band, band)) in enumerate(iterator): 
    
        dp = cf.sub_params(sub)

        cf.display_progress(f"{sub} {band['name']}", i, len(iterator),time_start) 

        # load epochs
        epochs = cf.load_epochs(sub,band,model['ep_setup'])[model['query']]

        scores, ps = stepwise_regression(epochs,model['predictors'],model['predictors_stepwise'])

        # save data into dataframe
        l = len(epochs.ch_names)
        coords = cf.get_coords(epochs)
        data = np.array([l*[sub], epochs.ch_names,coords[:,0],coords[:,1],coords[:,2],l*[band['name']]]).T
        df = pd.DataFrame(data,columns=['sub','ch_name','x','y','z','band'])#.astype('object')

        # for each sw_predictor save time point with max significant score
        for i_p, p in enumerate([''] + model['predictors_stepwise']):
            idx = np.squeeze((scores*(ps<alpha))[i_p].argmax(axis=-1))
            df['r2'+p] = [scores[i_p,i_ch,idx[i_ch]] for i_ch in range(l)]
            df['p'+p] = [ps[i_p,i_ch,idx[i_ch]] for i_ch in range(l)]
            df['t'+p] = [epochs.times[idx[i_ch]] for i_ch in range(l)]

        file_name = os.path.join(dp['derivatives_path'],f"{sub}_band-{band['name']}_reg-{model['tag']}.csv")
        df.to_csv(file_name)



################################################################################
#
#   FIGURES
#
################################################################################



def reg_statistics(model,predictors=[''],subs=cfg.subs,bands=cfg.bands,alpha=cfg.alpha,filter_tag='', plot = True, save = True):

    # load data 
    df = cf.load_df(f"reg_{model['tag']}",bands=bands,subs=subs)


    L = len(df)
    # filter out regressions with negative score
    df = df.query(f'r2>0  and p<{alpha}')

    if predictors == 'all': predictors= [''] + model['predictors']
    keys = ['p'+ p for p in predictors]
    conditions = ['r2'+ p for p in predictors]

    fig, ax = plt.subplots(len(keys),3, figsize=(12,3*len(keys)), gridspec_kw={'width_ratios': [1, 2, 0.5]})
    if len(keys)==1: ax=np.array([ax])
 

    fig.suptitle(f"Regression stats summary",fontsize=15)

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(bands)))

    for i, (k, c) in enumerate(zip(keys,conditions)):   

        # scatter r2 vs p -------------------------------------------------------------
        for i_band, band in enumerate(bands):
            ax[i,0].scatter(df[df['band']==band['name']][k],df[df['band']==band['name']][c],c=[colors[i_band]],alpha=0.3)
        ax[i,0].set_xlabel(k)
        ax[i,0].set_ylabel(c)      
        ax[i,0].axvline(alpha,color='k',linewidth=0.5)            
        ax[i,0].axhline(0,color='k',linewidth=0.5)            
        ax[i,0].spines['right'].set_visible(False)
        ax[i,0].spines['top'].set_visible(False)
        ax[i,0].tick_params(axis='both', which='both', size = 0, labelsize=9)       
        
        # number of significant channels ----------------------------------------------
        n = []

        for sub in subs:

            # load subject params
            dp = cf.sub_params(sub)

            # significant channels for subjec-band pair
            n_sub = []
        
            # iterate over bands -----------------------------------------------
            for i_band, band in enumerate(bands):
                    n_sub += [np.sum(np.array(df[(df['sub']==sub) & (df['band']==band['name'])][k])<alpha)]
                # go to next band -------------------------------------------------- 
            n += [n_sub]
            # go to next subject ======================================================

        n = np.array(n).T

        # width and positions of bars
        width = 0.8/len(bands)
        x = np.arange(len(subs))  

        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(bands)))

        for i_band, band in enumerate(bands):    
            ax[i,1].bar(x + i_band*width, n[i_band], width, color = colors[i_band])
            ax[i,2].bar(i_band*width, np.sum(n[i_band]), width, label=band['name'], color = colors[i_band])


        ax[i,1].set_ylabel("# significant channels")
        ax[i,1].set_xticks(np.arange(len(subs)) + 0.4)
        ax[i,1].set_xticklabels(subs,rotation=45)
        ax[i,1].spines['right'].set_visible(False)
        ax[i,1].spines['top'].set_visible(False)
        ax[i,1].tick_params(axis='both', which='both', size = 0, labelsize=9)

        ax[i,2].set_xticks([0.4])
        ax[i,2].set_xlim([-0.2,1])
        ax[i,2].set_xticklabels(['Total'])
        ax[i,2].spines['right'].set_visible(False)
        ax[i,2].spines['top'].set_visible(False)
        ax[i,2].tick_params(axis='both', which='both', size = 0, labelsize=9)

        if i>0: L = len(df)
        ax[i,2].set_title(f"{np.sum(n)}/{L}  {int(100*np.sum(n)/L)}%")        



    ax[0,2].legend(frameon = False, loc = (1,0.2),fontsize=9)
    fig.subplots_adjust(left=0.1,right=0.9,wspace=0.3, hspace=0.5)


    # save figure
    fig_name = os.path.join(cf.check_path(['..','Figures'+cfg.out, f"reg_{model['tag']}"]),f"stats_reg_{model['tag']}.pdf")
    if save: fig.savefig(fig_name, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()





def reg_maps(model,predictors=[''],subs=cfg.subs,bands=cfg.bands,alpha=cfg.alpha,filter_tag = '',fdr=False, plot = True, save = True, concat = True,time=True):
    """ Plot brain maps for variables (p_value, r2, ...) 
        stored in files for each band

    Parameters
    ----------
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nBand comparison brain map for reg {model['tag']}")

    if predictors == 'all': predictors= [''] + model['predictors']


    # load data 
    df = cf.load_df(f"reg_{model['tag']}",bands=bands,subs=subs)
    # get total number of channels
    L = len(df[df['band']==bands[0]['name']])
    # filter out regressions with negative score
    df = df.query(f'r2>0  and p<{alpha}')

    if time:
        conditions, keys = [], []
        for p in predictors:
            conditions += 2*['p'+ p]
            keys += ['r2'+ p, 't'+p]
    else:
        conditions = ['p'+ p for p in predictors]
        keys = ['r2'+ p for p in predictors]

    # loop over figures
    for i, (k, c) in enumerate(zip(keys,conditions)): 

        print(i, k)
        fig, ax = plt.subplots(len(bands),2, figsize=(12,3*len(bands)),gridspec_kw={
                           'width_ratios': [3, 1]})
        if len(bands)==1: ax=np.array([ax])
        fig.suptitle(fr"reg {model['tag']}, {k}, {c} $\leq {alpha}$",fontsize=20)

        for i_band, band in enumerate(bands):
            df_band = df[df['band']==band['name']]

            if len(df_band)==0: continue
            # extract data to plot
            x = df_band[k]
            coords = np.array([c for c in df_band['coords'].values])
            pl.brain_plot(ax[i_band,0],coords, x, ylabel=band['name'], mask = (df_band[c]>alpha), colorbar='' + (i_band==(len(bands)-1))*k, mode='interval', interval=[max(0,np.min(df[k])), np.max(df[k])]) 


            s = df_band[df_band[c]<alpha][k]
            x = np.linspace(0,np.max(df[k]),20)
            h, x = np.histogram(s,bins=x,density=False)
            ax[i_band,1].bar(x[:-1]+.5*(x[1]-x[0]),h,width=.8*(x[1]-x[0]))
            ax[i_band,1].set_xlabel(k)
            ax[i_band,1].spines['right'].set_visible(False)
            ax[i_band,1].spines['top'].set_visible(False)
            ax[i_band,1].tick_params(axis='both', which='both', size = 0, labelsize=9)
            if k!='r2': L = len(df)
            ax[i_band,1].set_title(f"{np.sum(h)}/{L}  {int(100*np.sum(h)/L)}%")        


        fig.subplots_adjust(left=0.1,right=0.9, hspace=0.5)
        if save: 
            fig_name = os.path.join(cf.check_path(['..','Figures'+cfg.out, f"reg_{model['tag']}"]),f"brain_map_reg_{model['tag']}_{str(i).zfill(2)}{'_temp'*concat}.pdf")
            fig.savefig(fig_name, format='pdf', dpi=100) 
        if plot: plt.show()
        else: plt.close()
    if save and concat:
        cf.concatenate_pdfs(cf.check_path(['..','Figures'+cfg.out, f"reg_{model['tag']}"]), 'temp', f"brain_map_reg-{model['tag']}.pdf", remove=True)









def reg_single_channel_wrap(model,subs=cfg.subs,bands=cfg.bands,alpha=cfg.alpha, name=''):

    # load data 
    df = cf.load_df(f"reg_{model['tag']}",bands=bands,subs=subs)
    # filter out not significant
    df = df.query(f'p<{alpha}')

    time_start = time.time()
    for i in range( len(df)):
        cf.display_progress(f"{df.iloc[i]['sub']} {df.iloc[i]['ch_name']} {df.iloc[i]['band']}", i, len(df),time_start) 
        for predictor in model['predictors_stepwise']:
            if df.iloc[i]['p'+predictor]<alpha  and df.iloc[i]['r2'+predictor]>0:
                band = cfg.bands[[b['name'] for b in cfg.bands].index(df.iloc[i]['band'])]  
                reg_single_channel(df.iloc[i]['sub'],df.iloc[i]['ch_name'],band,model, plot = False)
    cf.concatenate_pdfs(cf.check_path(['..','Figures'+cfg.out, f"reg_{model['tag']}"]), 'temp', f"single_channel_reg-{model['tag']}{name}.pdf", remove=True)



def reg_single_channel(sub,ch_name,band,model,alpha=cfg.alpha, plot = True, save = True,temp=True,n=4):


    # load subject params
    dp = cf.sub_params(sub)

    # load epochs
    all_epochs = cf.load_epochs(sub,band,model['ep_setup'],picks=[ch_name]).decimate(5)
    epochs = all_epochs[model['query']]

    # regression
    scores, ps = stepwise_regression(epochs,model['predictors'],model['predictors_stepwise'])

    predictors = model['predictors_stepwise']

    # FIGURE --------------------------------------------------------
    fig, ax = plt.subplots(len(predictors)+1,3, figsize=(15,3*(len(predictors)+1)))
    fig.suptitle(f"{sub} {ch_name} {band['name']}\n{model['query']}",fontsize=15)

    # ch_position
    pl.channel_position_plot(ax[0,2],cf.get_coords(epochs,picks=[ch_name]))

    # channel summary   
    pl.channel_summary_plot(ax[0,1],sub,ch_name,model['tag'],model['predictors'])


    t_sig = pl.score_plot(ax[0,0],scores,ps,epochs.times,model['predictors_stepwise'],vlines = [dp['soa']*x for x in model['ep_setup']['xticks']],xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']])
    ax[1,0].set_title(model['query'])

    
    if np.sum(np.array([len(t_sig[pr]) == 2 for pr in predictors])) == 0: 
        plt.close()
        return

    for i_p, predictor in enumerate(predictors):

        predictor_values = epochs.metadata[predictor].values
        N=min(n,len(set(predictor_values)))
        predictor_thresholds = np.linspace(np.nanmin(predictor_values), np.nanmax(predictor_values), N+1) 
        #predictor_thresholds = np.percentile(predictor_values, np.linspace(0,100,N+1))
        colors = plt.get_cmap('autumn')(np.linspace(0., 1.,  N))

        for i in range(N):

            y = np.squeeze(epochs[f'{predictor}>={predictor_thresholds[i]} and {predictor}<={predictor_thresholds[i+1]}'].get_data(picks=ch_name))
            label = f"({round(predictor_thresholds[i])},{round(predictor_thresholds[i+1])})"
            
            pl.trace_plot(ax[i_p+1,0], epochs.times, y, ylabel='zscored power',title =predictor, xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']],
                                   vlines = [dp['soa']*x for x in model['ep_setup']['xticks']], color = colors[i],label=label,legend=True,plot_sem=True,xlabel='t (s)')

        if len(t_sig[predictor])==2: 
            ax[i_p+1,0].axvspan(t_sig[predictor][0]-cfg.smooth*0.5,t_sig[predictor][1]+cfg.smooth*0.5,color='gray',alpha=0.3,lw=0)
            pl.response_plot(ax[i_p+1,1], ch_name, all_epochs.copy(), predictor, tmin=t_sig[predictor][0]-cfg.smooth*0.5 ,tmax=t_sig[predictor][1]+cfg.smooth*0.5)
        else: pl.clear_axes(np.array([ax[i_p+1,1]]))

        values = all_epochs.metadata[predictor].values
        order =np.argsort(values)
        yticks = [j for j, x in enumerate(np.diff(values[order])!=0) if x]
        yticklabels = [str(values[order][idx]) for idx in yticks]

        m = np.squeeze(all_epochs.copy().get_data(picks=ch_name))
        m = (m-m.mean())/m.std()
        vmin = np.percentile(m,15)
        vmax = np.percentile(m,85)

        pl.imshow_plot(ax[i_p+1,2], m,vmin=vmin,vmax=vmax, ylabel = predictor, xlims  = all_epochs.times,
                           yticks = yticks, yticklabels = yticklabels, colorbar = 'z-score',cmap='RdBu_r',order=order)







    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
    if save: 
        fig_name = os.path.join(cf.check_path(['..','Figures'+cfg.out,  f"reg_{model['tag']}"]),f"{sub}_{ch_name}_{band['name']}_temp.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()



def reg_single_channel2(sub,ch_name,band_name,model,extra_band,predictor2,predictors='all',alpha=cfg.alpha, plot = True, save = False,temp=False,n=4,twsize=cfg.tws):


    # load subject params
    dp = cf.sub_params(sub)

    if predictors=='all': predictors=model['predictors']
    ep_setup=model['ep_setup']

    # raw file
    mneraw_file = os.path.join(dp['derivatives_path'],f"{sub}_band-{band_name}_raw.fif")
    raw = mne.io.read_raw_fif(mneraw_file,preload=True)
    all_epochs = cf.load_epochs(dp,raw.copy().pick_channels([ch_name]),model['ep_setup'])
    epochs= all_epochs[model['query']]

    # get predictor values 
    X = np.array(epochs.metadata[model['predictors']])
    X[np.isinf(X)] = -15.5

    # get responses averaged over time windows
    tmin,tmax = model['ep_setup']['tmin'],model['ep_setup']['tmax']- twsize
    tws = [[t,t+cfg.tws] for t in np.arange(tmin,tmax ,(tmax - tmin)/100.)]
    y = []
    t = []
    for t1, t2 in tws:
        # epoch average
        e = epochs.copy()
        y += [np.squeeze(e.crop(t1,t2).get_data(picks=ch_name).mean(axis=-1))]
        t += [0.5*(t1+t2)]
    y = np.swapaxes(np.array([y]).T,-1,-2)

    # compute regressions
    R = regression(X,y,t)



    # FIGURE --------------------------------------------------------

    fig, ax = plt.subplots(len(predictors)+2,3, figsize=(15,3*(len(predictors)+2)))

    fig.suptitle(f"{sub} {ch_name} {band_name}\n{model['predictors']}\n{model['query']}",fontsize=15)

    # ch_position
    #pl.clear_axes(ax[0,:])
    pl.channel_position_plot(ax[0,0],[dp['coords'][dp['ch_names'].index(ch_name)]],0)

    # channel summary   
    pl.channel_summary_plot(ax[0,1],sub,ch_name,model['tag'],model['predictors'])



    # raster plot
    y = cf.load_epochs(dp,raw.copy().pick_channels([ch_name]),model['ep_setup']).get_data(picks=ch_name)
    #y = (y - np.median(y))/np.subtract(*np.percentile(y, [75, 25]))
    y = (y - np.median(y))/y.std()
    pl.imshow_plot(ax[1,1], y[:,0,:],title='All trials', xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']],vlines = [dp['soa']*x for x in ep_setup['xticks']],ylabel='epoch', colorbar = 'zscore',vmin=-1,vmax=1,cmap='RdBu_r')

    # score plot
    t_sig = pl.score_plot(ax[1,0],R,model['predictors'],vlines = [dp['soa']*x for x in ep_setup['xticks']],xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']])
    ax[1,0].set_title(model['query'])

   
 
    if np.sum(np.array([len(t_sig[pr]) == 2 for pr in predictors])) == 0: 
        plt.close()
        return

    for i_p, predictor in enumerate(predictors):

        predictor_values = epochs.metadata[predictor].values
        predictor_values[np.isinf(predictor_values)] =-15.5
        N=min(n,len(set(predictor_values)))
        predictor_thresholds = np.linspace(np.nanmin(predictor_values), np.nanmax(predictor_values), N+1) 
        colors = plt.get_cmap('autumn')(np.linspace(0., 1.,  N))

        for i in range(N):

            y = np.squeeze(epochs[f'{predictor}>={predictor_thresholds[i]} and {predictor}<={predictor_thresholds[i+1]}'].get_data(picks=ch_name))
            label = f"({round(predictor_thresholds[i])},{round(predictor_thresholds[i+1])})"
            
            pl.trace_plot(ax[i_p+2,0], epochs.times, y, ylabel='zscored power',title =predictor, xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']],
                                   vlines = [dp['soa']*x for x in ep_setup['xticks']], color = colors[i],label=label,legend=True,plot_sem=True,xlabel='t (s)')

        if len(t_sig[predictor])==2: 
            ax[i_p+2,0].axvspan(t_sig[predictor][0],t_sig[predictor][1],color='gray',alpha=0.3,lw=0)
            pl.response_plot(ax[i_p+2,1], ch_name, all_epochs.copy(), predictor, tmin=t_sig[predictor][0] ,tmax=t_sig[predictor][1])
        else: pl.clear_axes(np.array([ax[i_p+2,1]]))


    # Third column

    order = np.argsort(np.squeeze(epochs.copy().crop(t_sig[predictor2][0] , t_sig[predictor2][1]).get_data(picks=ch_name).mean(axis=-1)))

    # raster plot
    y = epochs.get_data(picks=ch_name)
    y = (y - np.median(y))/y.std()
    pl.imshow_plot(ax[1,2], y[order,0,:],title=f"{band_name} {predictor2} ordered", xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']],vlines = [dp['soa']*x for x in ep_setup['xticks']],ylabel='epoch', colorbar = 'zscore',vmin=-1,vmax=1,cmap='RdBu_r')


    for i in range(len(extra_band)):
        # raw file
        mneraw_file = os.path.join(dp['derivatives_path'],f"{sub}_band-{extra_band[i]}_raw.fif")
        raw = mne.io.read_raw_fif(mneraw_file,preload=True)
        all_epochs = cf.load_epochs(dp,raw.copy().pick_channels([ch_name]),model['ep_setup'])
        epochs= all_epochs[model['query']]

        # raster plot
        y = epochs.get_data(picks=ch_name)
        y = (y - np.median(y))/y.std()
        pl.imshow_plot(ax[2+i,2], y[order,0,:],title=f"{extra_band[i]}", xlims=[model['ep_setup']['tmin'],model['ep_setup']['tmax']],vlines = [dp['soa']*x for x in ep_setup['xticks']],ylabel='epoch', colorbar = 'zscore',vmin=-1,vmax=1,cmap='RdBu_r')

    pl.clear_axes(np.array(ax[0,2]))


    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
    if save: 
        #i_band = [band['name'] for band in cfg.bands].index(band_name)
        fig_name = os.path.join(cf.check_path(['..','Figures'+cfg.out,  f"reg_{model['tag']}"]),f"{sub}_{ch_name}_{band_name}_temp.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()





def reg_clustering(model,predictors=[],subs=cfg.subs,bands=cfg.bands,alpha=cfg.alpha, name='',r2=1):

    # load data 
    df = cf.load_df(f"reg_{model['tag']}",bands=bands,subs=subs)

    if 	predictors=='all': predictors=model['predictors']

    #keys = ['r2','p','t'] + ['dr2_'+p,'p_'+p,'t_'+p for p in predictors]
    keys = ['r2']*r2 + ['dr2_'+p for p in predictors]
    ps = ['p']*r2 + ['p_'+p for p in predictors]
   
    for  i in range(len(keys)):
        X = []
        for band in bands:
            df_band=df.query(f"band =='{band['name']}'")

            scores = df_band[keys[i]].values
            p_value =  df_band[ps[i]].values
            p_value_full = df_band['p'].values

            X += [(p_value<alpha)*(p_value_full<alpha)]

            #x_ch += (scores*(p_values<alpha)*(p_values[0]<alpha)).tolist()
            #x_ch += (1.*(p_values<alpha)*(p_values[0]<alpha)).tolist()
            #x_ch += (-np.log10(p_values)*(p_values[0]<alpha)).tolist()

        X = np.array(X)

     
        
        # Standardize the data to have a mean of ~0 and a variance of 1
        X = StandardScaler().fit_transform(X.T).T

        #C = np.corrcoef(X)
        C = np.cov(X)

        U, S, Vh = scipy.linalg.svd(C)


        #idx = np.argsort(C.mean(axis=0))
        #idx = np.argsort(Vh[0,:])
        idx = [0,10,8,6,9,7,5,4,3,2,1]


        fig, ax = plt.subplots(1,2,figsize=(8,3))

        fig.suptitle(keys[i],fontsize=20)

        m = np.percentile(np.abs(C),90)
        ax[0].imshow(C,vmin=-m,vmax=m,cmap='RdBu_r')
        #ax[0].imshow(C)
        ax[0].set_yticks(range(len(bands)))
        ax[0].set_yticklabels([band['name'] for band in bands])
        ax[0].set_xticks(range(len(bands)))
        ax[0].set_xticklabels([band['name'] for band in bands],rotation=90)


        im = ax[1].imshow(C[idx,:][:,idx],vmin=-m,vmax=m,cmap='RdBu_r')
        #ax[1].imshow(C[idx,:][:,idx])
        ax[1].set_yticks(range(len(bands)))
        ax[1].set_yticklabels([bands[i]['name'] for i in idx])
        ax[1].set_xticks(range(len(bands)))
        ax[1].set_xticklabels([bands[i]['name'] for i in idx],rotation=90)

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10,length=0) 
        cbar.outline.set_visible(False)
        cbar.set_label('corr coeff', rotation=90,fontsize=10)

        for a in ax:
            a.spines['right'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.spines['top'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.tick_params(axis='both', which='both', size = 0, labelsize=10)


        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        plt.show()






'''
def reg_clustering(model,predictors=[],subs=cfg.subs,bands=cfg.bands,alpha=cfg.alpha, name=''):

    # load data 
    df = cf.load_df(f"reg_{model['tag']}",bands=bands,subs=subs)

    if 	predictors=='all': predictors=model['predictors']

    #keys = ['r2','p','t'] + ['dr2_'+p,'p_'+p,'t_'+p for p in predictors]
    keys = ['r2'] + ['dr2_'+p for p in predictors]
    ps = ['p'] + ['p_'+p for p in predictors]
   

    X = []
    # loop only over channels with significant regressions
    for ch in set(df.query(f'r2>0  and p<{alpha}')['ch_name']):
        x_ch = []
        for band in bands:
            scores = df.query(f"ch_name=='{ch}' and band =='{band['name']}'")[keys].values[0]
            p_values =  df.query(f"ch_name=='{ch}' and band =='{band['name']}'")[ps].values[0]
            #x_ch += (scores*(p_values<alpha)*(p_values[0]<alpha)).tolist()
            x_ch += (1.*(p_values<alpha)*(p_values[0]<alpha)).tolist()
            #x_ch += (-np.log10(p_values)*(p_values[0]<alpha)).tolist()


        X += [x_ch]
    X = np.array(X)


 
    
    # Standardize the data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(X)

    #C = np.corrcoef(X.T)
    C = np.cov(X.T)

    U, S, Vh = scipy.linalg.svd(C)

    idx = np.argsort(C.mean(axis=0))
    #idx = np.argsort(Vh[0,:])

    fig, ax = plt.subplots(1,2,figsize=(8,3))

    m = np.percentile(np.abs(C),80)
    ax[0].imshow(C,vmin=-m,vmax=m,cmap='RdBu_r')
    ax[0].set_yticks(range(len(bands)))
    ax[0].set_yticklabels([band['name'] for band in bands])
    ax[0].set_xticks(range(len(bands)))
    ax[0].set_xticklabels([band['name'] for band in bands],rotation=90)


    ax[1].imshow(C[idx,:][:,idx],vmin=-m,vmax=m,cmap='RdBu_r')
    ax[1].set_yticks(range(len(bands)))
    ax[1].set_yticklabels([bands[i]['name'] for i in idx])
    ax[1].set_xticks(range(len(bands)))
    ax[1].set_xticklabels([bands[i]['name'] for i in idx],rotation=90)
    plt.show()


    sys.exit()


    n_components = 7
    


    # PCA -----------------------------------------------------------------------------------
    print('PCA')

    # Create a PCA instance: pca
    pca = PCA(n_components=n_components)

    # fit pca model
    reduced_data = pca.fit_transform(X_std)

    print(reduced_data.shape)

    # Plot explained variances and principal components
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(bands)))

    fig, ax = plt.subplots(1,6, figsize=(24, 3))
    fig.suptitle(f"PCA")

    features = range(pca.n_components_)
    ax[0].bar(features, pca.explained_variance_ratio_)
    ax[0].set_xlabel('PCA features')
    ax[0].set_ylabel('variance (ratio)')
    ax[0].set_xticks(features)

    
    # plot vectors
    for i in range(4):
        v = pca.components_[i].reshape(11,-1)
        m = np.max(np.abs(v))
        ax[i+1].imshow(v,vmin=-m,vmax=m, cmap='RdBu_r')
        ax[i+1].set_xticks(range(len(keys)))
        ax[i+1].set_xticklabels(keys,rotation=90)
        ax[i+1].set_title(f'PC{i+1}')

    ax[1].set_yticks(range(len(bands)))
    ax[1].set_yticklabels([band['name'] for band in bands])  


    # plot scatter of 2 first components
    ax[-1].remove()
    ax[-1]=fig.add_subplot(1,6,6,projection='3d')
    ax[-1].scatter(reduced_data[:,0], reduced_data[:,1],reduced_data[:,2])
    ax[-1].set_ylabel('PC2')
    ax[-1].set_xlabel('PC1')
    ax[-1].set_zlabel('PC3')

    fig_name = os.path.join(cf.check_path(['..','Figures','reg_clustering']),f'PCA_summary.jpg')
    fig.savefig(fig_name, format='jpg', dpi=100)

    plt.close()


    fig, ax = plt.subplots(1,n_components, figsize=(4*n_components, 3), sharey= True, sharex = True)
    fig.suptitle(f"PCA kmeans")


    for dim in range(2,n_components):

        
        print(f'n dims {str(dim)}')    

        # run k means
        ks = range(2, n_components)
        inertias = []
        for k in ks:
            print(f'\tn clusters {str(k)}')

            # Create a KMeans instance with k clusters: model
            kmodel = KMeans(n_clusters=k)        
            # Fit model to samples
            kmodel.fit(reduced_data[:,:dim])
            
            # Append the inertia to the list of inertias
            inertias.append(kmodel.inertia_)

            labels = kmodel.predict(reduced_data[:,:dim])

            colors2 = plt.get_cmap('magma')(labels/k)

            fi, a = plt.subplots(1,k+1, figsize=(3*(k+1), 3))
            fi.suptitle(f"PCA kmeans {dim} dim {k} clusters")

            # plot scatter of 2 first components
            a[0].remove()
            a[0]=fi.add_subplot(1,k+1,1,projection='3d')
            a[0].scatter(reduced_data[:,0], reduced_data[:,1],reduced_data[:,2], color=colors2[:])
            a[0].set_ylabel('PC2')
            a[0].set_xlabel('PC1')
            a[0].set_zlabel('PC3')
    
            for i in range(k):
                v = X[labels==i,:].mean(axis=0).reshape(11,-1)
                m = np.max(np.abs(v))
                a[i+1].imshow(v,vmin=-m,vmax=m, cmap='RdBu_r')
                a[i+1].set_xticks(range(len(keys)))
                a[i+1].set_xticklabels(keys,rotation=90)
                a[i+1].set_title(f'#ch = {np.sum(labels==i)}')
                a[i+1].set_yticks(range(len(bands)))
                a[i+1].set_yticklabels([band['name'] for band in bands])  


            fig_name = os.path.join(cf.check_path(['..','Figures','reg_clustering']),f'PCA_kmeans_{dim}dim_{k}clusters.jpg')
            fi.savefig(fig_name, format='jpg', dpi=100)
            plt.close(fi)


        ax[dim].plot(ks, inertias, '-o', color='black')
        ax[dim].set_xlabel('number of clusters, k')
        ax[dim].set_ylabel('inertia')
        ax[dim].set_title(f'{str(dim)} components')
        ax[dim].set_xticks(ks)

    fig_name = os.path.join(cf.check_path(['..','Figures','reg_clustering']),f'PCA_kmeans.jpg')
    fig.savefig(fig_name, format='jpg', dpi=100)
    plt.close()








    
    # TSNE --------------------------------------------------------------------------

    #np.random.seed(42)
    #rndperm = np.random.permutation(X_std.shape[0])
    #N = 10000
    #df_subset = data.loc[rndperm[:N],:].copy()
    #data_subset = df_subset[feat_cols].values


    #-----------------------------------------------------------------------------
    fig, ax = plt.subplots(1,n_components, figsize=(4*n_components, 3), sharey= True, sharex = True)
    fig.suptitle(f"TSNE kmeans")


    for dim in range(2,4):

        print(f'n components {str(dim)}')

        time_start = time.time()
        tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
        reduced_data = tsne.fit_transform(X_std)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        print(reduced_data.shape)

        # run k means
        ks = range(2, n_components)
        inertias = []
        for k in ks:
            print(f'\tn clusters {str(k)}')

            # Create a KMeans instance with k clusters: model
            kmodel = KMeans(n_clusters=k)        
            # Fit model to samples
            kmodel.fit(reduced_data[:,:dim])
            
            # Append the inertia to the list of inertias
            inertias.append(kmodel.inertia_)

            labels = kmodel.predict(reduced_data[:,:dim])

            colors2 = plt.get_cmap('magma')(labels/k)

            fi, a = plt.subplots(1,k+1, figsize=(3*(k+1), 3))
            fi.suptitle(f"TSNE kmeans {dim} dim {k} clusters")

            # plot scatter of 2 first components
            a[0].scatter(reduced_data[:,0], reduced_data[:,1], color=colors2[:])
            a[0].set_ylabel('C2')
            a[0].set_xlabel('C1')

    
            for i in range(k):
                v = X[labels==i,:].mean(axis=0).reshape(11,-1)
                m = np.max(np.abs(v))
                a[i+1].imshow(v,vmin=-m,vmax=m, cmap='RdBu_r')
                a[i+1].set_xticks(range(len(keys)))
                a[i+1].set_xticklabels(keys,rotation=90)
                a[i+1].set_title(f'#ch = {np.sum(labels==i)}')
                a[i+1].set_yticks(range(len(bands)))
                a[i+1].set_yticklabels([band['name'] for band in bands])  


            fig_name = os.path.join(cf.check_path(['..','Figures','reg_clustering']),f'TSNE_kmeans_{dim}dim_{k}clusters.jpg')
            fi.savefig(fig_name, format='jpg', dpi=100)
            plt.close(fi)



        ax[dim].plot(ks, inertias, '-o', color='black')
        ax[dim].set_xlabel('number of clusters, k')
        ax[dim].set_ylabel('inertia')
        ax[dim].set_title(f'{str(dim)} components')
        ax[dim].set_xticks(ks)

        fig_name = os.path.join(cf.check_path(['..','Figures','reg_clustering']),f'TSNE_kmeans.jpg')
        fig.savefig(fig_name, format='jpg', dpi=100)


'''




























