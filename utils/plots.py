import sys, os, glob, csv, json, datetime, copy
import warnings
import numpy as np
import mne
import matplotlib as mpl
from nilearn import plotting 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config as cfg
import utils.common_functions as cf


# ====================================================================================================
# SINGLE PANEL 

def brain_plot(ax,coords,coefs,mode='minmax',colorbar='',title='',mask=[],interval=[],ylabel='',ROIs=[]):
    """Plot electrodes on brain with nilearn

    This function is basically a wrap for nilearn view_markers or connectome view.
    colors for markers can be customized throuhg variables 
    
    Parameters
    ----------
    coefs : list of floats 
        color of the markers according to coefs
    """
    
    # get colors from coefficients
    if mode == 'around_the_mean':
        m = coefs.nanmean()
        M = np.nanmax(np.abs(coefs - m))
        colors = mpl.cm.RdBu_r((coefs - m + M)/(2*M))
        norm=mpl.colors.Normalize(m-M,m+M)
        cmap = mpl.cm.RdBu_r

    elif mode == 'symmetric':
        M = np.nanmax(np.abs(coefs))
        colors = mpl.cm.RdBu_r((coefs + M)/(2*M))
        norm=mpl.colors.Normalize(-M, M)
        cmap = mpl.cm.RdBu_r
    elif mode == 'minmax':
        M, m = np.nanmax(coefs), np.nanmin(coefs)
        colors = mpl.cm.viridis((coefs - m)/(M - m))
        norm=mpl.colors.Normalize(m, M)
        cmap = mpl.cm.viridis
    elif mode == 'interval':
        if len(interval) != 2: sys.exit('Plot_brain: Specify an interval for colorbar')
        M, m = interval[1], interval[0]
        colors = mpl.cm.viridis((coefs - m)/(M - m))
        norm= mpl.colors.Normalize(m, M)
        cmap = mpl.cm.viridis

    elif mode == 'contrast':
        #color_labels = [[0,0,0,.5],[0,0,0.7,1],[0.9,0.1,0.1,1],[0.5,0,0.5,1]]
        #colors = np.array([ color_labels[int(c)] for c in coefs])
        #color_labels = color_labels[1:]

        colors = mpl.cm.viridis((coefs-1)/2)
        
    elif mode == 'ROIs':
        colors = mpl.cm.tab20(coefs/20)
        cmap = mpl.cm.tab20
        norm= mpl.colors.Normalize(0, 1)

    sizes = 50*np.ones(len(colors))
    if len(mask)>0 and np.sum(mask)>0:
        
        sizes[mask] = 0.5*np.ones(np.sum(mask))
        colors[mask] = np.array([[0.,0.,0.,.5]]*np.sum(mask))

    # nilearn plot
    n = len(coords)
    plotting.plot_connectome(np.zeros((n,n)),coords, node_color=colors.tolist(), node_size=sizes, display_mode='lyrz', axes=ax) 
    
    # standalone colorbar
    if colorbar != '':
        if mode=='contrast':
            colors = mpl.cm.viridis([0,.5,1])
            for i, label in enumerate(colorbar): ax.text(.90,-0.13*(i+1),label,color=colors[i],transform=ax.transAxes,fontsize=15) 
        else:
            divider = make_axes_locatable(ax)
            cbax = divider.append_axes("bottom", size="5%", pad='10%')
            cbar = mpl.colorbar.ColorbarBase(cbax, cmap=cmap,norm=norm,orientation="horizontal")
            cbar.ax.tick_params(labelsize=9,length=0) 
            cbar.outline.set_visible(False)
            cbax.set_xlabel(colorbar,fontsize=12)

    if mode == 'ROIs':
        divider = make_axes_locatable(ax)
        cbax = divider.append_axes("right", size="2%", pad='10%')
        cbar = mpl.colorbar.ColorbarBase(cbax, cmap=cmap,norm=norm,orientation="vertical", ticks=np.arange(len(ROIs))/20.+0.5/20.)
        cbar.ax.tick_params(labelsize=9,length=0) 
        cbar.ax.set_yticklabels([r[4:] for r in ROIs])
        cbar.outline.set_visible(False)
        cbax.set_xlabel(colorbar,fontsize=12)

    ax.set_title(title,fontsize=15)
    ax.text(-0.1,0.5,ylabel,rotation='vertical',va='center', transform=ax.transAxes,fontsize=15)


def scatter_plot(a,x,y,coefs,mode='minmax',colorbar='',title='',mask=[],interval=[],ylabel='',xlabel='', xlims = [], ylims = [], xticks = [], xticklabels = [], vlines = [], yticks = [], yticklabels = [], hlines=[]):
   
    # get colors from coefficients
    if mode == 'around_the_mean':
        m = coefs.nanmean()
        M = np.nanmax(np.abs(coefs - m))
        colors = mpl.cm.RdBu_r((coefs - m + M)/(2*M))
        norm=mpl.colors.Normalize(m-M,m+M)
        cmap = mpl.cm.RdBu_r

    elif mode == 'symmetric':
        M = np.nanmax(np.abs(coefs))
        colors = mpl.cm.RdBu_r((coefs + M)/(2*M))
        norm=mpl.colors.Normalize(-M, M)
        cmap = mpl.cm.RdBu_r
    elif mode == 'minmax':
        M, m = np.nanmax(coefs), np.nanmin(coefs)
        colors = mpl.cm.viridis((coefs - m)/(M - m))
        norm=mpl.colors.Normalize(m, M)
        cmap = mpl.cm.viridis
    elif mode == 'interval':
        if len(interval) != 2: sys.exit('Plot_brain: Specify an interval for colorbar')
        M, m = interval[1], interval[0]
        colors = mpl.cm.viridis((coefs - m)/(M - m))
        norm= mpl.colors.Normalize(m, M)
        cmap = mpl.cm.viridis

    elif mode == 'contrast':
        #color_labels = [[0,0,0,.5],[0,0,0.7,1],[0.9,0.1,0.1,1],[0.5,0,0.5,1]]
        #colors = np.array([ color_labels[int(c)] for c in coefs])
        #color_labels = color_labels[1:]
        colors = mpl.cm.viridis((coefs-1)/2)
        


    edgecolors='face'
    #for i, k in enumerate(mask): 
    #    print(i)
    #    print(k)
    if len(mask)>0 and np.sum(mask)>0: edgecolors = ['k'*int(~k) + 'none'*int(k) for k in mask]

    im = a.scatter(x,y,color=colors,edgecolors=edgecolors,s = 100)
    
    # standalone colorbar
    if colorbar != '':
        if mode=='contrast':
            colors = mpl.cm.viridis([0,.5,1])
            for i, label in enumerate(colorbar): ax.text(.90,-0.13*(i+1),label,color=colors[i],transform=a.transAxes,fontsize=15) 
        else:
            divider = make_axes_locatable(a)
            cbax = divider.append_axes("right", size="5%", pad='10%')
            cbar = mpl.colorbar.ColorbarBase(cbax, cmap=cmap,norm=norm)
            cbar.ax.tick_params(labelsize=9,length=0) 
            cbar.outline.set_visible(False)
            cbax.set_xlabel(colorbar,fontsize=12)


    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    a.set_title(title,fontsize=15)

    if xticks!=[]:
        a.set_xticks(xticks)
        for xl in xticks: a.axvline(xl, linestyle='-', color='grey', linewidth=0.5)
        if bool(xticklabels): a.set_xticklabels(xticklabels,rotation = 90)
    if yticks!=[]:
        a.set_yticks(yticks)
        for yl in yticks: a.axhline(yl, linestyle='-', color='grey', linewidth=0.5)
        if bool(yticklabels): a.set_yticklabels(yticklabels)
    if vlines!=[]: 
        for xl in vlines: a.axvline(xl, linestyle='-', color='grey', linewidth=0.5)
    if hlines!=[]: 
        for xl in hlines: a.axhline(xl, linestyle='-', color='grey', linewidth=0.5)
    if len(ylims)>1: a.set_ylim([ylims[0],ylims[-1]])
    if len(xlims)>1: a.set_xlim([xlims[0],xlims[-1]])

    a.spines['right'].set_visible(False)
    #a.spines['left'].set_visible(False)
    a.spines['top'].set_visible(False)
    #a.spines['bottom'].set_visible(False)
    a.tick_params(axis='both', which='both', size = 0, labelsize=7)


def channel_position_plot(ax,coords):
    """Plot electrode position on brain with nilearn

    Plot all electrodes in white and selected electrode in blue.
    Output jpeg with several projections

    Parameters
    ----------
    ax : pyplot ax
        ax where to plot 
    coords : list 
        coords of channel to plot
    """
    # nilearn plot
    plotting.plot_connectome(np.zeros((len(coords),len(coords))),coords, node_color=['firebrick'], display_mode='lyrz', axes=ax) 



def imshow_plot(a, m, title = '', xlabel = '', ylabel = '',  
                vmin = [], vmax = [], xlims = [], ylims = [],
                xticks = [], xticklabels = [], yticks = [], yticklabels = [],
                colorbar = '',vlines = [],cmap='RdBu_r',order=[]):
    """ plot a 2dim data using imshow """

                    
    if not bool(vmin): vmin = np.percentile(m,5)
    if not bool(vmax): vmax = np.percentile(m,95)
    if xlims==[]: xlims = range(m.shape[1])
    if ylims==[]: ylims = range(m.shape[0])

    if order!= []: m = m[order]

    im = a.imshow(m, extent=(xlims[0],xlims[-1],ylims[0],ylims[-1]), origin='lower', cmap=cmap, interpolation ='none', aspect='auto',vmin=vmin,vmax=vmax)

    if colorbar != '':
        divider = make_axes_locatable(a)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=5,length=0) 
        cbar.outline.set_visible(False)
        cbar.set_label(colorbar, rotation=90,fontsize=7)

    a.set_title(title)
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)

    a.spines['right'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.tick_params(axis='both', which='both', size = 0, labelsize=7)
    if len(xticks)!=0:
        a.set_xticks(xticks)
        #for x in xticks: a.axvline(x, linestyle='-', color='w', linewidth=0.5)
        if xticklabels!=[]: a.set_xticklabels(xticklabels,rotation = 90)
    if len(yticks)!=0:
        a.set_yticks(yticks)
        #for y in yticks: a.axhline(y, linestyle='-', color='w', linewidth=0.5)
        if yticklabels!=[]: a.set_yticklabels(yticklabels)

    if vlines!=[]: 
        for xl in vlines: a.axvline(xl, linestyle='-', color='w', linewidth=0.5)


def trace_plot(a, x, y, title = '', xlabel = '', ylabel = '', xlims = [], ylims = [],
                xticks = [], xticklabels = [], vlines = [], yticks = [], yticklabels = [], color = 'k', label='', 
                legend=False, plot_sem = False, plot_std = False, plot_p595 = False, mode='avg'):
    """ Plot time-series on an ax

    for multiple time-series options include 
    mode  = 'epohcs', 'avg' or 'both' for one line per epoch, one line for the avg or both
    plot_sem : shade area within standard error of the mean
    plot_std : shade area within +- 1 std of the mean
    plot_p595: dashed lines at 5 and 90 percentiles
    """



    # if y is a vector just plot y. If it is an array plot mean (and maybe sem and percentiles)
    if len(y.shape) == 1:
        if isinstance(y[0],complex): y = np.angle(y)
        a.plot(x,y,color=color,label=label)

    else:
        # get mean
        if isinstance(y[0,0],complex):   # for ITC
            y /= np.abs(y)
            m = np.abs(np.mean(y,axis=0))
        else:                       # for everything else
            y = (y-y.mean())/y.std()
            m = y.mean(axis=0) 


        if plot_sem:
            # plot standard error of the mean 
            sem = y.std(axis=0)/np.sqrt(y.shape[0])
            a.fill_between(x,m + sem,m - sem,color=color,alpha=0.2, lw = 0)
        if plot_std:
            # plot standard deviation around the mean 
            std = y.std(axis=0)
            a.fill_between(x,m + std,m - std, color=color, alpha=0.2, lw = 0)
        if plot_p595:
            # plot percentile 5 and 95
            a.plot(x,np.percentile(y,5,axis=0),color=color,ls='--', lw = 0.5)
            a.plot(x,np.percentile(y,95,axis=0),color=color,ls='--', lw = 0.5)

        if mode=='avg':
            # plot mean on top
            a.plot(x,m,color=color,label=label)
        elif mode == 'both':
            for i in range(y.shape[0]): a.plot(x,y[i], color=color,lw=0.2,alpha=0.1,zorder=0)
            # plot mean on top
            a.plot(x,m,color=color,label=label,lw=2)
        elif mode=='epochs':
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, 10))
            y = y/(5.*y.std())
            for i in range(y.shape[0]): a.plot(x,i + y[i], color=colors[i%10],lw=0.5)


    a.set_title(title)
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)

    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.tick_params(axis='both', which='both', size = 0, labelsize=7)

    if xticks!=[]:
        a.set_xticks(xticks)
        for xl in xticks: a.axvline(xl, linestyle='-', color='grey', linewidth=0.5)
        if bool(xticklabels): a.set_xticklabels(xticklabels,rotation = 90)
    if yticks!=[]:
        a.set_yticks(yticks)
        for yl in yticks: a.axhline(yl, linestyle='-', color='grey', linewidth=0.5)
        if bool(yticklabels): a.set_yticklabels(yticklabels)
    if vlines!=[]: 
        for xl in vlines: a.axvline(xl, linestyle='-', color='grey', linewidth=0.5)
    if xlims==[]: xlims = x
    a.set_xlim([xlims[0],xlims[-1]])
    if len(ylims)>1: a.set_ylim([ylims[0],ylims[-1]])

    if legend: a.legend(frameon=False,loc=(0.8,0.5),fontsize=7)



def response_plot(a,ch_name,epochs,predictor,tmin,tmax):

    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.tick_params(axis='both', which='both', size = 0, labelsize=7)
    a.set_xlabel(predictor)


    trialtypes = list(set(epochs.metadata['trialtype']))
    trialtypes.sort()
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(trialtypes)))
    epochs.crop(tmin,tmax)



    for i, trialtype in enumerate(trialtypes):
        x = epochs[f"trialtype == '{trialtype}'"].metadata[predictor].values
        x[np.isnan(x)] = np.nanmin(x)
        xs = list(set(x.round()))
        xs.sort()
       
        if len(xs)>20: xs = np.percentile(xs,np.linspace(0,100,10))

        ms = [] 
        ss = []
        X = []
        for x in xs:
            y = epochs[f"trialtype == '{trialtype}' and {predictor}>={x} and {predictor}<{x+1}"].get_data(picks=ch_name)
            # plot if there are at least 5 epochs 
            if y.shape[0] > 1:  
                Y = np.squeeze(y).mean(axis=-1)
                ms += [Y.mean()]
                ss += [Y.std()/np.sqrt(y.shape[0])] 
                X += [x]

        a.errorbar(X,ms,yerr=ss,color=colors[i],label=trialtype)

    a.set_title(f"[{round(tmin,3)},{round(tmax,3)}] s")

    a.legend(frameon=False,fontsize=7)


def score_plot(a,scores,ps,t,predictors,alpha=cfg.alpha,vlines=[],xlims = [], ylims = []):

    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.tick_params(axis='both', which='both', size = 0, labelsize=7)
    a.set_ylabel(r'$r^2$')

    scores = np.squeeze(scores)
    ps = np.squeeze(ps)

    a.plot(t,scores[0],color='k',lw=2)
    M = max(scores[0]) 
    a.scatter(t[ps[0]<alpha], 1.05*M*np.ones_like(t[ps[0]<alpha]),marker= 's',color='k')


    t_sig={'all':t[np.argmax(scores[0]*(ps[0]<alpha))]}

    #a.set_ylim([-0.05*max(scores[0]),1.5*max(scores[0])])

    a1 = a.twinx()
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(predictors)))
    #m = 0
    for j, predictor in enumerate(predictors):

        # score and p value
        a1.plot(t,scores[j+1],color=colors[j],lw=1)
        a.scatter(t[(ps[j+1]<alpha)*(ps[0]<alpha)], (1.1 +0.1*j)*M*np.ones_like(t[(ps[j+1]<alpha)*(ps[0]<alpha)]),marker= 's',color=colors[j],label=predictor)
        
        if min(ps[j+1])<alpha: t_sig[predictor]=significant_window(scores[j+1]*(ps[j+1]<alpha)*(ps[0]<alpha),t)
        else: t_sig[predictor]=[]
        #m = max(m,scores[j+1].max())

    a1.set_ylabel(r"$\Delta r^2$")
    a.legend(frameon=False,fontsize=7)
    #a1.set_ylim([-0.1*m,2*m])
    a1.spines['top'].set_visible(False)
    a1.tick_params(axis='both', which='both', size = 0, labelsize=7)

    if vlines!=[]: 
        for xl in vlines: a.axvline(xl, linestyle='-', color='grey', linewidth=0.5)

    if xlims==[]: xlims = x
    a.set_xlim([xlims[0],xlims[-1]])

    return t_sig



def significant_window(x,t):

    if x.max() == 0: return []

    # to avoid running out of the vector set extremes to 0
    x[0], x[-1] = 0, 0

    center = np.argmax(x)
    i_min, i_max = center, center
    while x[i_min]>0: i_min -=1
    while x[i_max]>0: i_max +=1

    return [t[i_min],t[i_max]]


def channel_summary_plot(a,sub,ch_name,tag,predictors,bands=cfg.bands,alpha=cfg.alpha):

    # load data 
    df = cf.load_df(f"reg_{tag}",bands=bands,subs=[sub])
    # filter out regressions with negative score
    df = df.query(f"p<{alpha} and ch_name=='{ch_name}'")


    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.tick_params(axis='both', which='both', size = 0, labelsize=7)
    a.set_yticks(range(len(bands)))
    a.set_yticklabels([band['name'] for band in bands])
    a.set_xticks(range(len(predictors)+1))
    a.set_xticklabels(['full'] + [p for p in predictors],rotation=90)

    T = np.zeros((len(bands),len(predictors)+1))


    for i_band, band in enumerate(bands):
        df_band = df.query(f"band=='{band['name']}'")
        #print(df_band)
        try:
            T[i_band,0] = df_band['t']
        except:
            T[i_band,:] = -1
            continue
        for i_p, pred in enumerate(predictors):
            
            #print(pred,df_band[f'p_{pred}'].values, df_band[f'p_{pred}'].values)
            T[i_band,i_p+1] = df_band[f't{pred}']
            if df_band.iloc[0][f'p{pred}']>alpha:
                 T[i_band,i_p+1] = -1
                 #print(df_band.iloc[0][f'p_{pred}']>alpha)





    palette = copy.copy(plt.cm.viridis)
    palette.set_under('w', 1.0)
    im = a.imshow(T,vmin=0,vmax=0.6,cmap=palette)
    
    divider = make_axes_locatable(a)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,length=0) 
    cbar.outline.set_visible(False)
    cbar.set_label('t (s)', rotation=90,fontsize=7)




def clear_axes(axes):
    """ Clear spines and ticks to have less cluttered plots """

    if len(axes.shape) == 1: ax = np.array([axes])
    if len(axes.shape) == 0: ax = np.array([[axes]])

    for a in axes.ravel():
        a.spines['right'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.tick_params(axis='both', which='both', size = 0, labelsize=7)
        a.set_xticks([])
        a.set_yticks([])


   









'''


       /\
      /  \
     / _o \
    / <(\  \
   /   />`D \
  '----------`  
UNDER CONSTRUCTION







def plot_regression(ax,data,predictors,response,title):
    """Plot univariate regression for several predictors

    Basically a wrap for seaborn regplot.
    It overlays univariate gregressions for several predictors

    Parameters
    ----------
    P : dict
        Dictionary with parameters (initially read from json parameters file)
    ax : pyplot axis 
        axis where to plot
    data : pandas df
        predictors and response data
    predictors : list of str
        names of the columns of data to be used as predictors
    response : str
        name of the column of data to be used as response
    title : str
        title of the subplot
    fname : str
        name of file, it should contain full path and end in '.jpeg'

    Returns
    -------
    ax : pyplot axis
    """
    
    colors = mpl.cm.cividis(np.linspace(0,1,len(predictors)))
    for i, predictor in enumerate(predictors) : sns.regplot(x=predictor,y = 'burst rate (Hz)',data=data,label=predictor,ax=ax,color=colors[i],scatter=True)
    #ax.legend(frameon=False)
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', size=0)


    return ax













'''

















