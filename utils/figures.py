import sys, os, glob, csv, json, datetime
import warnings
import numpy as np
import mne
import matplotlib as mpl
from nilearn import plotting 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import static_parameters as sp
import utils.common_functions as cf
import utils.plots as pl
import utils.overview as ov


'''
# ====================================================================================================
# COMMON FIGURES
def band_comparison_brain(key, test, queries, split=None, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands):
    """ Plot brain maps for variables (p_value, r2, ...) 
        stored in files for each band

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    split : int or str
        if int divide the time window into 'split' chunks and run one test in each
        if string separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test for each word position
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nBand comparison brain map for {test} {queries['tag']}")

    df = cf.load_df(f"{test}-{queries['tag']}", subs=subs, bands=bands)

    if split==None: labels = ['']# no split
    elif isinstance(split,int): labels = list(np.unique(df['tw'].to_numpy())) # split by time windows
    elif isinstance(split,str): labels = list(np.unique(df[split].to_numpy())) # split by conditions

    num_plots = len(labels)

    fig, ax = plt.subplots(len(sp.bands),num_plots, figsize=(10*num_plots,3*len(sp.bands)))

    fig.suptitle(fr"{test} {queries['tag']}, $p \leq {alpha}$",fontsize=20)
    

    # threshold is given in units of p, has to be transformed
    if key[0]=='p': alpha = -np.log10(alpha)
 
    if len(sp.bands)==1: ax=ax[np.newaxis,:] 
    if split == None: ax=ax[:,np.newaxis]

    # loop over plots
    for i_plot, label in enumerate(labels):
        
        for i_band, band in enumerate(sp.bands):
            # filter df by band and split
            if split == None:
                df_band = df[df['band']==band['name']]
                title = ''
            elif isinstance(split,int):
                df_band = df[(df['band']==band['name']) & ([x==label for x in df['tw'].to_numpy()])]
                title = '' + (str(label) + ' s')*(i_band==0)
            elif isinstance(split,str):
                df_band = df[(df['band']==band['name']) & (df[split]==label)]
                title = '' + f"{split} = {label}"*(i_band==0) 


            # extract data to plot
            x = df_band[key]

            # for p values
            if key[0]=='p':
                # transform p values and threshold
                x = -np.log10(x)
                cbar_label = r'-log$_{10}(p)$'
                
            # get channel positions
            coords = np.array([ c for c in df_band['coords']])

            

            pl.brain_plot(ax[i_band,i_plot],coords,np.clip(x,alpha,3),title=title, 
                           ylabel='' + band['name']*(i_plot==0), colorbar = '' + (i_band==(len(bands)-1))*cbar_label, 
                            mask = (x<alpha),mode='interval',interval=[alpha,3])        


        
    #fig.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    # save figure
    fig_name = os.path.join(cf.check_path(['..','Figures', test+sp.out]),f"brain_map_{test}_{queries['tag']}.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    if sp.plot: plt.show()
    plt.close()
'''



# ====================================================================================================
# COMMON FIGURES
def band_comparison_brain(key, test, file_tag, split=None, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands):
    """ Plot brain maps for variables (p_value, r2, ...) 
        stored in files for each band

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    split : int or str
        if int divide the time window into 'split' chunks and run one test in each
        if string separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test for each word position
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nBand comparison brain map for {test} {file_tag}")

    df = cf.load_df(f"{test}-{file_tag}", subs=subs, bands=bands)

    if split==None: labels = ['']# no split
    elif isinstance(split,int): labels = list(np.unique(df['tw'].to_numpy())) # split by time windows
    elif isinstance(split,str): labels = list(np.unique(df[split].to_numpy())) # split by conditions

    num_plots = len(labels)
    

    # threshold is given in units of p, has to be transformed
    if key[0]=='p': alpha = -np.log10(alpha)
 
    if len(sp.bands)==1: ax=ax[np.newaxis,:] 

    # loop over plots
    for i_plot, label in enumerate(labels):


        fig, ax = plt.subplots(len(bands),1, figsize=(10,3*len(bands)))

        fig.suptitle(fr"{test} {file_tag}, $p \leq {alpha}$",fontsize=20)

        
        for i_band, band in enumerate(bands):
            # filter df by band and split
            if split == None:
                df_band = df[df['band']==band['name']]
                title = ''
                figtag = ''
            elif isinstance(split,int):
                df_band = df[(df['band']==band['name']) & ([x==label for x in df['tw'].to_numpy()])]
                title = '' + (str(label) + ' s')*(i_band==0)
                figtag=f'_tw{i_plot}'
            elif isinstance(split,str):
                df_band = df[(df['band']==band['name']) & (df[split]==label)]
                title = '' + f"{split} = {label}"*(i_band==0) 
                figtag = f"_{split}-{label}"

            # extract data to plot
            x = df_band[key]

            # for p values
            if key[0]=='p':
                # transform p values and threshold
                x = -np.log10(x)
                cbar_label = r'-log$_{10}(p)$'
                
            # get channel positions
            coords = np.array([ c for c in df_band['coords']])

            

            pl.brain_plot(ax[i_band],coords,np.clip(x,alpha,3),title=title, 
                           ylabel=band['name'], colorbar = '' + (i_band==(len(bands)-1))*cbar_label, 
                            mask = (x<alpha),mode='interval',interval=[alpha,3])        


        
        #fig.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

        # save figure
        fig_name = os.path.join(cf.check_path(['..','Figures', test+sp.out]),f"brain_map_{test}_{file_tag}{figtag}.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
        if sp.plot: plt.show()
        plt.close()




# ====================================================================================================
# COMMON FIGURES

def contrast(key1, key2, test1,test2, model1, model2, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands,plot=True,save=True):
    """ Plot brain maps for variables (p_value, r2, ...) 
        stored in files for each band

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
    model1, model2 : dict
        contains mne queries, file tag
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nContrast {test1} {model1['tag']} {test2} {model2['tag']}")


    df1 = cf.load_df(f"{test1}_{model1['tag']}", subs=subs, bands=bands)
    df2 = cf.load_df(f"{test2}_{model2['tag']}", subs=subs, bands=bands)

    fig, ax = plt.subplots(len(bands),1, figsize=(10,3*len(bands)))
    if len(bands)==1: ax = np.array([ax])

    fig.suptitle(fr"{test1} {key1} {model1['tag']} vs {test2} {key3} {model2['tag']}, $p \leq {alpha}$",fontsize=20)
        
    for i_band, band in enumerate(bands):

        df1_band = df1[df1['band']==band['name']]
        df2_band = df2[df2['band']==band['name']]

        mask1 = df1_band[key].values<alpha
        mask2 = df2_band[key].values<alpha

        # for regression results filter significant channels
        if test=='reg':
            mask1 *= (df1_band['r2'].values>0)*(df1_band['p'].values<alpha)
            mask2 *= (df2_band['r2'].values>0)*(df2_band['p'].values<alpha)

        x = 1.*mask1 + 2*mask2

        
        cb = ''
        if i_band == len(bands)-1: cb = [model1['tag'],model2['tag'],'both']

        pl.brain_plot(ax[i_band],cf.eval_coords(df1_band['coords']),x, ylabel=band['name'],
                        mask = (x<0.5),mode='contrast',colorbar=cb)        

    #fig.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    # save figure
    if save:
        fig_name = os.path.join(cf.check_path(['..','Figures', test1+sp.out]),f"contrast_{test1}-{model1['tag']}_{test2}-{model2['tag']}.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
    if plot: plt.show()
    else: plt.close()




'''
def band_comparison_contrast(test, tag, predictors, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands, fdr=True):
    """ Plot brain maps for variables (p_value, r2, ...) 
        stored in files for each band

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    split : int or str
        if int divide the time window into 'split' chunks and run one test in each
        if string separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test for each word position
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nBand comparison brain map for {test} {tag} {predictors[0]} {predictors[1]}")


    df = cf.load_df(f"{test}_{tag}", subs=subs, bands=bands)

    fig, ax = plt.subplots(len(sp.bands),1, figsize=(10,3*len(sp.bands)))

    fig.suptitle(fr"{test} {tag} Contrast {predictors[0]} {predictors[1]}, $p \leq {alpha}$",fontsize=20)

    for i_band, band in enumerate(bands):

        df_band = df[df['band']==band['name']]

        # get channel positions
        coords = np.array([ c for c in df_band['coords']])

        mask1 = df_band['p_'+predictors[0]+'_fdr'*fdr].values<alpha
        mask2 = df_band['p_'+predictors[1]+'_fdr'*fdr].values<alpha
        
        x = 1.*mask1 + 2*mask2

        
        cb = ''
        if i_band == len(bands)-1: cb =  predictors + ['both']

        pl.brain_plot(ax[i_band],coords,x, ylabel=band['name'],
                        mask = (x<0.5),mode='contrast',colorbar=cb)        


        
    #fig.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    # save figure
    fig_name = os.path.join(cf.check_path(['..','Figures', f"{test}_{tag}"]),f"contrast_{test}_{tag}_{predictors[0]}_{predictors[1]}{'_fdr'*fdr}.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    if sp.plot: plt.show()
    plt.close()

'''


def band_comparison_lateralization(key, test, tag, split=None, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands):
    """ Bar plots for proportion of significant channels on
        left right axis and frontal occipital axis

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
    tag : str
        test tag
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nLateralization figure for {test} {tag}")

    df = cf.load_df(f"{test}_{tag}", subs=subs, bands=bands)

    if split==None: labels = ['']# no split
    elif isinstance(split,int): labels = list(np.unique(df['tw'].to_numpy())) # split by time windows
    elif isinstance(split,str): labels = list(np.unique(df[split].to_numpy())) # split by conditions

    num_plots = len(labels)
    
    if len(sp.bands)==1: ax=ax[np.newaxis,:] 

    # loop over plots
    for i_plot, label in enumerate(labels):


        fig, ax = plt.subplots(len(bands),2, figsize=(5,3*len(bands)))

        fig.suptitle(fr"{test} {tag}, $p \leq {alpha}$",fontsize=20)

        # for axis limits
        x, y = 0, 0

        for i_band, band in enumerate(bands):
            # filter df by band and split
            if split == None:
                df_band = df[df['band']==band['name']]
                title = ''
                figtag = ''
            elif isinstance(split,int):
                df_band = df[(df['band']==band['name']) & ([x==label for x in df['tw'].to_numpy()])]
                title = '' + (str(label) + ' s')*(i_band==0)
                figtag=f'_tw{i_plot}'
            elif isinstance(split,str):
                df_band = df[(df['band']==band['name']) & (df[split]==label)]
                title = '' + f"{split} = {label}"*(i_band==0) 
                figtag = f"_{split}-{label}"

            df_sig = df_band[df_band[key]<=alpha]

            # get channel positions
            coords_all = np.array([ c for c in df_band['coords']])
            coords_sig = np.array([ c for c in df_sig['coords']])

            # skip if there aren't significant channels
            if len(coords_sig.shape)!=2: continue

            # left rigth
            h_all, bins = np.histogram(coords_all[:,0],bins=6,density=False)
            h_sig, bins = np.histogram(coords_sig[:,0],bins=bins,density=False)

            width = (bins[1]-bins[0])

            ax[i_band,0].bar(bins[:-1]+width/2,h_sig/h_all,width=width*0.9,color='firebrick')
            ax[i_band,0].set_xticks([bins[0],bins[-1]])
            ax[i_band,0].set_xticklabels(['L','R'])
            #ax[i_band,0].set_ylim([0,1])
            ax[i_band,0].spines['right'].set_visible(False)
            ax[i_band,0].spines['top'].set_visible(False)
            ax[i_band,0].tick_params(axis='both', which='both', size = 0, labelsize=9)
            
            y = max(y,np.max(h_sig/h_all))

            # frontal occipital
            h_all, bins = np.histogram(coords_all[:,1],bins=6,density=False)
            h_sig, bins = np.histogram(coords_sig[:,1],bins=bins,density=False)

            width = (bins[1]-bins[0])

            ax[i_band,1].barh(bins[:-1]+width/2,h_sig/h_all,height=width*0.9,color='firebrick')
            ax[i_band,1].set_yticks([bins[0],bins[-1]])
            ax[i_band,1].set_yticklabels(['O','F'])
            #ax[i_band,1].set_xlim([0,1])
            ax[i_band,1].spines['right'].set_visible(False)
            ax[i_band,1].spines['top'].set_visible(False)
            ax[i_band,1].tick_params(axis='both', which='both', size = 0, labelsize=9)
            
            x = max(x,np.max(h_sig/h_all))

        for i_band in range(len(bands)):
            ax[i_band,0].set_ylim([0,y])
            ax[i_band,1].set_xlim([0,x])

        fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.4, hspace=0.4)


        # save figure
        fig_name = os.path.join(cf.check_path(['..','Figures', f"{test}_{tag}"+sp.out]),f"lateralization_{test}_{tag}_{key}.pdf")
        fig.savefig(fig_name, format='pdf', dpi=100) 
        if sp.plot: plt.show()
        plt.close()






'''
def band_comparison_lateralization(key, test, queries, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands):
    """ Bar plots for proportion of significant channels on
        left right axis and frontal occipital axis

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
    queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f"\nLateralization figure for {test} {queries['tag']}")

    df = cf.load_df(f"{test}-{queries['tag']}", subs=subs, bands=bands)

    fig, ax = plt.subplots(len(sp.bands),2, figsize=(5,3*len(sp.bands)))
    if len(sp.bands)==1: ax=ax[np.newaxis,:] 

    fig.suptitle(fr"{test} {queries['tag']}, $p \leq {alpha}$",fontsize=20)
    
    x, y = 0, 0

    for i_band, band in enumerate(sp.bands):

        df_band = df[df['band']==band['name']]

        df_sig = df_band[df_band[key]<=alpha]

        # get channel positions
        coords_all = np.array([ c for c in df_band['coords']])
        coords_sig = np.array([ c for c in df_sig['coords']])

        # left rigth
        h_all, bins = np.histogram(coords_all[:,0],bins=6,density=False)
        h_sig, bins = np.histogram(coords_sig[:,0],bins=bins,density=False)

        width = (bins[1]-bins[0])

        ax[i_band,0].bar(bins[:-1]+width/2,h_sig/h_all,width=width*0.9,color='firebrick')
        ax[i_band,0].set_xticks([bins[0],bins[-1]])
        ax[i_band,0].set_xticklabels(['L','R'])
        #ax[i_band,0].set_ylim([0,1])
        ax[i_band,0].spines['right'].set_visible(False)
        ax[i_band,0].spines['top'].set_visible(False)
        ax[i_band,0].tick_params(axis='both', which='both', size = 0, labelsize=9)
        
        y = max(y,np.max(h_sig/h_all))

        # frontal occipital
        h_all, bins = np.histogram(coords_all[:,1],bins=6,density=False)
        h_sig, bins = np.histogram(coords_sig[:,1],bins=bins,density=False)

        width = (bins[1]-bins[0])

        ax[i_band,1].barh(bins[:-1]+width/2,h_sig/h_all,height=width*0.9,color='firebrick')
        ax[i_band,1].set_yticks([bins[0],bins[-1]])
        ax[i_band,1].set_yticklabels(['O','F'])
        #ax[i_band,1].set_xlim([0,1])
        ax[i_band,1].spines['right'].set_visible(False)
        ax[i_band,1].spines['top'].set_visible(False)
        ax[i_band,1].tick_params(axis='both', which='both', size = 0, labelsize=9)
        
        x = max(x,np.max(h_sig/h_all))

    for i_band in range(len(bands)):
        ax[i_band,0].set_ylim([0,y])
        ax[i_band,1].set_xlim([0,x])

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.4, hspace=0.4)


    # save figure
    fig_name = os.path.join(cf.check_path(['..','Figures', test+sp.out]),f"lateralization_{test}_{queries['tag']}.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    if sp.plot: plt.show()
    plt.close()
'''


def band_comparison_matrix(key, file_tag, path, title='', label='',
                            subs=sp.subject_list, bands=sp.bands,log10=False):
    """ Compare variables (p_value, r2, ...) between bands 
        with scatter plots

    Parameters
    ----------
    key : str
        name of the column to extract
    file_tag : str
        name of the figure will be brain_map_{file_tag}.pdf
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   

    """

    print(f'\nBand comparison matrix for {file_tag}')

    if len(bands)==1: sys.exit('At least 2 bands needed only 1 was passed')
    
    df = cf.load_df(file_tag,subs=subs,bands=bands)


    # fig 2: comparison accros bands ---------------------------------------------------------------------
    fig, ax = plt.subplots(len(bands)-1,len(bands)-1,figsize=(2.5*(len(bands)-1),2.5*(len(bands)-1)))    
    fig.suptitle(title ,fontsize=20)
    clear_axes(ax)   

    for i, band in enumerate(bands[:-1]):

        # data for first band
        x = df[df['band'] == band['name']][key]
        if log10: x = -np.log10(x)
               
        ax[-1,i].set_xlabel(band['name']) 
        ax[i,0].set_ylabel(bands[i+1]['name']) 

        # comparison across bands
        for j in range(i,len(bands)-1): 
            y = df[df['band'] == bands[j+1]['name']][key]
            if log10: y = -np.log10(y)
                        
            ax[j,i].scatter(x,y)
            #ax[j,i].hist2d(x,y,bins=np.linspace(min(x.min(),y.min()),max(x.max(),y.max()),10),cmap='Reds')
            
            
            # draw diagonal line
            ax[j,i].axis('equal')
            ax[j,i].plot([0, 1], [0, 1],'k--', transform=ax[j,i].transAxes)
            ax[j,i].spines['left'].set_visible(True)
            ax[j,i].spines['bottom'].set_visible(True)
            #ax[i,j].set_xlim([0,np.log10(sp.n_permutations)])
            #ax[i,j].set_ylim([0,np.log10(sp.n_permutations)])
            #ax[i,j].set_xticks([0,1,2])
            #ax[i,j].set_yticks([0,1,2])

    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

    # save figure
    fig_name = os.path.join(cf.check_path(path),f"matrix_{file_tag}.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    if sp.plot: plt.show()
    plt.close()





                                    
def num_significant_channels(key,test, tag, split=None, alpha=sp.alpha, subs=sp.subject_list, bands=sp.bands):
    """ Make a bar plot with number of significant channels
        per subject per band for some test or regression

    Parameters
    ----------
    key : str
        name of the column to extract
    test : str
        name of the test/regression that works as a label for files
        queries : dict
        generated with cf.compose_queries, contains mne queries, file tag, queries list
    split : int or str
        if int divide the time window into 'split' chunks and run one test in each
        if string separate epochs by values of metadata column 'split',
        f.ex. split = 'w_position' run the test for each word position
    alpha : float
        level of significance (leave out p>alpha)
    subs : list of str
        list of subjects to include f.e. [sub-01, sub-02]
    bands : list of dict
        list of bands to include   
    """

    print(f"\n{test} {tag} summary figure")

    df = cf.load_df(test + '_' + tag)


    if split==None: labels = ['']# no split
    elif isinstance(split,int): labels = list(np.unique(df['tw'].to_numpy())) # split by time windows
    elif isinstance(split,str): labels = list(np.unique(df[split].to_numpy())) # split by conditions

    num_plots = len(labels)



    fig, ax = plt.subplots(num_plots,1, figsize = (len(subs),3*num_plots),sharex=True)
    fig.suptitle(fr"{test} {tag}, {key} $\leq {alpha}$")


    if split == None: ax=np.array([ax])

    # loop over plots
    for i_plot, label in enumerate(labels):
      
        # number of significant channels
        n = []

        for sub in subs:

            # load subject params
            dp = cf.sub_params(sub)

            # significant channels for subjec-band pair
            n_sub = []
        
            # iterate over bands -----------------------------------------------
            for i_band, band in enumerate(bands):
                if split == None:
                    n_sub += [np.sum(np.array(df[(df['sub']==sub) & (df['band']==band['name'])][key])<alpha)]
                    title = ''
                elif isinstance(split,int):
                    n_sub += [np.sum(np.array(df[(df['sub']==sub) & (df['band']==band['name']) & ([x==label for x in df['tw'].to_numpy()])][key])<alpha)]
                    title = str(label) + ' s'
                elif isinstance(split,str):
                    n_sub += [np.sum(np.array(df[(df['sub']==sub) & (df['band']==band['name']) & (df[split]==label)][key])<alpha)]
                    title = f"{split} = {label}"
                # go to next band -------------------------------------------------- 
            n += [n_sub]
            # go to next subject ======================================================

        n = np.array(n).T

        # width and positions of bars
        width = 0.8/len(bands)
        x = np.arange(len(subs))  

        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(bands)))

        for i_band, band in enumerate(bands):    
            ax[i_plot].bar(x + i_band*width, n[i_band], width, label=band['name'], color = colors[i_band])

        ax[i_plot].set_ylabel("# significant channels")
        ax[i_plot].set_title(title)
        ax[i_plot].set_xticks(np.arange(len(subs)) + 0.4)
        ax[i_plot].set_xticklabels(subs)
        ax[i_plot].spines['right'].set_visible(False)
        ax[i_plot].spines['top'].set_visible(False)
        ax[i_plot].tick_params(axis='both', which='both', size = 0, labelsize=9)


    ax[i_plot].legend(frameon = False, loc = (1,0.2),fontsize=9)
    fig.subplots_adjust(right=0.8)


    # save figure
    fig_name = os.path.join(cf.check_path(['..','Figures', test+sp.out]),f"summary_{test}_{tag}_{key}.pdf")
    fig.savefig(fig_name, format='pdf', dpi=100) 
    if sp.plot: plt.show()
    plt.close()

                                    






def spike_detection(sub,ch_name,tw):

    dp = cf.sub_params(sub)

    # load epochs
    filename = os.path.join(dp['derivatives_path'], f'{sub}_raw.fif')
    raw = mne.io.read_raw_fif(filename, preload=True).crop(tmin=tw[0], tmax=tw[1])

    
    amp = np.array([raw.get_data(picks=ch_name)])
    t = np.linspace(tw[0],tw[1],amp.shape[-1])    

    # compute wavelets
    f = np.arange(5,150,2)
    power = np.squeeze(mne.time_frequency.tfr_array_morlet(amp,sp.srate,f, n_cycles=f / 2., output = 'power')[0])
    power = power/power.std(axis=-1)[:,np.newaxis]#*f[:,np.newaxis]
    amp= np.squeeze(amp)


    timewindowsize = 0.1 # s
    d = int(timewindowsize*sp.srate)
    C = []
    for i in range(len(t) - d):
        cf.print_d(f"{i/(len(t) - d)}")
        corr = np.corrcoef(power[:,i:i+d])
        corr = corr[np.triu_indices_from(corr,k=1)]    
        C +=[corr]
        
    C =np.array(C).T


    fig, ax = plt.subplots(4,1, figsize=(6, 6))
    fig.suptitle(f'{sub} {ch_name}')

    coords = dp['coords'][dp['ch_names'].index(ch_name)]
    pl.channel_position_plot(ax[0],[coords],0)

    pl.trace_plot(ax[1],t,amp, xlims = [tw[0]+0.5,tw[1]-0.5], ylabel='V (uV)')

    pl.imshow_plot(ax[2], power, title = '', ylabel = 'f (Hz)', xlims = [tw[0]+0.5,tw[1]-0.5], ylims = f,colorbar='power (std)')

    pl.trace_plot(ax[3],t[:-d],C, xlims = [tw[0]+0.5,tw[1]-0.5], ylabel='correlation',xlabel='t (s)',plot_std = True, plot_p595 = True, mode='avg')


    fig_name = os.path.join(cf.check_path(['..','Figures','spikes']),f'{sub}_{ch_name}_{int(tw[0])}.pdf')
    fig.savefig(fig_name, format='pdf')
    plt.close()













