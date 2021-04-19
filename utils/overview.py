# import packages1
import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time
from nilearn import plotting 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from itertools import product


# import local functions
import config as cfg
import utils.common_functions as cf
import utils.plots as pl
import utils.preprocessing as pr


def subject_overview(subs = cfg.subs, bands = cfg.bands, EpochsSetups = cfg.EpochsSetups,
                        avg = True, TrialWise = True, key = None, picks = 'good'):
    """ Make one figure per channel with
    channel position and one evoked plot per band

    Parameters
    ----------
    key :  str
        metadata key to plot separate traces by metadata value
    """

    print("\n======== subject overview =========================================\n")

    # iterate over subjects
    for sub in subs:
        # load subject params
        SP = cf.sub_params(sub)

        # preload raw data for all bands
        epochs = []
        iterator =  product(enumerate(bands),enumerate(EpochsSetups))
        for i, ((iBand, band),(iEp, EpochsSetup)) in enumerate(iterator): 
            cf.print_d(f"{sub}, loading {band['name']} epochs...")
            # get list of channels
            PicksList = cf.get_picks(SP, picks, band = band)
            if PicksList == []: continue
            # epochs (phase epochs are complex and cannot be resampled)
            try: epochs += [cf.load_epochs(sub, band, EpochsSetup, picks = PicksList).resample(100)]
            except: epochs += [cf.load_epochs(sub, band, EpochsSetup, picks = PicksList)]

        # iterate over channels, one figure per channel
        TimeStart = time.time()
        for iCh, ChName in enumerate(PicksList): 
            cf.display_progress(f"{sub}, {ChName}", iCh, len(PicksList),TimeStart) 
    
            channel_overview(sub, ChName, bands, EpochsSetups, EpochsList=epochs, key=key, plot = False, save = True, temp=True)
            
        # concatenate figures subject wise
        FileName = f"{sub}_band-{band['name']}_{picks}_overview.pdf"
        FiguresPath = cf.check_path(['..', 'Figures', 'overview' + cfg.out])
        cf.concatenate_pdfs(FiguresPath, 'temp', FileName, remove = True)

    # concatenate for all subjects
    FileName = f"band-{band['name']}_{picks}_overview.pdf"
    FiguresPath = cf.check_path(['..', 'Figures', 'overview' + cfg.out])
    cf.concatenate_pdfs(FiguresPath, 'sub', FileName, remove = False)









def channel_overview(sub, ChName, bands, EpochsSetups, EpochsList=None, 
                     key=None, plot = True, save = False, temp=False):
    """ Same as subject overview but for one single channel
    

    Parameters
    ----------
    key :  str
        metadata key to plot separate traces by metadata value
    """

    # load subject params
    SP = cf.sub_params(sub)
    #if ChName not in SP['ChNames']: sys.exit(f"{sub} {ChName} not found  (o_0) ")

    # make figure
    fig, ax = plt.subplots(3,len(EpochsSetups),figsize=(5*len(EpochsSetups),4.5))  
         
    # add extra dimension if missing
    if len(EpochsSetups) == 1: ax = np.array([ax]).T

    title = f"{sub} {ChName}"
    if 'sample' in EpochsSetups[0].keys(): title += f"\nsample: {EpochsSetups[0]['sample']}"
    fig.suptitle(title)

    # plot channel cosition
    pl.channel_position_plot(ax[0,0],cf.get_coords(SP,picks=[ChName]))   
    pl.clear_axes(ax[0,:])   


    # LOOP OVER BANDS AND EPOCH SETUPS -----------------------------------------
    iterator =  product(enumerate(bands),enumerate(EpochsSetups))
    for i, ((iBand, band),(iEp, EpochsSetup)) in enumerate(iterator): 

        if EpochsList == None: epochs = cf.load_epochs(sub,band,EpochsSetup)
        else: epochs = EpochsList[iEp].copy()
        epochs = epochs.pick_channels([ChName])

 
        xticks = [x*SP['soa'] for x in EpochsSetup['xticks']]

        # avg PLOT -----------------------------------------------------------
        if band['method'] in [None,'filter']: ylabel = f"{band['name']} band\n"*(iEp==0) + f"\nV (z-scored)"
        elif band['method'] in ['complex']: ylabel = f"{band['name']} band\n"*(iEp==0) + f"\nITC"
        else: ylabel = f"{band['name']} power\n"*(iEp==0) + f"\ndB (z-scored)"

        # OPTION 1: plot all epochs together ----------------------------------
        if key == None:
            x = np.squeeze(epochs.get_data(picks=ChName))
            
            # plot avg trace
            pl.trace_plot(ax[2,iEp], epochs.times, x, ylabel = ylabel, 
                            vlines = [SP['soa']*x for x in EpochsSetup['xticks']], plot_sem = True)

        # OPTION 2: separate epochs by condition -------------------------------
        else:
            conditions=list(set(epochs.metadata[key]))
            conditions.sort()
            if isinstance(conditions[0],str):
                queries = [f"{key} =='{c}'" for c in conditions]
            elif len(conditions)>10: 
                conditions = np.linspace(min(conditions),max(conditions),5)
                queries = [f"{key} >={conditions[i]} and {key} <{conditions[i+1]}" for i in range(len(conditions)-1)]
            else:
                 queries = [f"{key} =={c}" for c in conditions]
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(conditions)))
            for i_q, query in enumerate(queries):
                x = np.squeeze(epochs.copy()[query].get_data(picks=ChName))
                # plot avg trace per condition
                pl.trace_plot(ax[2,iEp], epochs.times, x,  ylabel = '' + band['name']*(iEp==0), 
                   vlines = xticks, plot_sem = True, color=colors[i_q],label = conditions[i_q])

            # make legend
            if iEp==len(EpochsSetups)-1 and key!=None:
                ax[-1,-1].legend(frameon=False,title=key,loc=(1,0.2),fontsize = 7)  # 



        # TrialWise PLOT -----------------------------------------------------------
        if band['method'] in [None,'filter']: ylabel, cb = f"{band['name']} band", f"V (z-scored)"
        elif band['method'] in ['complex']: ylabel, cb =f"{band['name']} band", r"$\theta$"
        else: ylabel, cb = f"{band['name']} power", f"dB (z-scored)"

        try: order = EpochsSetup['order']
        except: order = key
        if order!=None:
            values = epochs.metadata[order].values
            idx =np.argsort(values)
            yticks = [j for j, x in enumerate(np.diff(values[idx])!=0) if x]
            yticklabels = [str(values[idx][y]) for y in yticks]
            ylabel=(iEp==0)*ylabel + f"\n\n{order}"
        else: 
            idx = []
            yticks = []
            yticklabels = []
            ylabel=(iEp==0)*ylabel + f"\n\nepoch"

        m = np.squeeze(epochs.copy().get_data(picks=ChName))
        if isinstance(m[0,0],complex): 
            m = np.angle(m)
            vmin, vmax = -np.pi, np.pi
        else:
            m = (m-m.mean())/m.std()
            vmin = np.percentile(m,15)
            vmax = np.percentile(m,85)

        pl.imshow_plot(ax[1,iEp], m,vmin=vmin,vmax=vmax, ylabel = ylabel, xlims  = epochs.times,
                                title = EpochsSetup['name'], yticks = yticks, yticklabels = yticklabels, vlines = xticks, colorbar = cb,cmap='jet',order=idx)


    ax[-1,0].set_xlabel('t (s)')
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    if save: 
        FiguresPath = cf.check_path(['..','Figures' , 'overview' + cfg.out ])
        fig_name = os.path.join(FiguresPath,f"{sub}_{ChName}{'_temp'*temp}.pdf")
        fig.savefig(fig_name, format='pdf') 
    if plot: plt.show()
    else: plt.close()  




def electrode_positions(subs=cfg.subs):
    """ Make one figure per channel with the position of the channel
        Concatenate for each subject
    """

    print("\n======== electrode positions ======================================\n")

    FiguresPath =cf.check_path(['..','Figures' , 'overview' + cfg.out ])

    # iterate over subjects
    for sub in subs:

        cf.print_d(f'Electrode positions for {sub}')

        # load subject params
        SP = cf.sub_params(sub)

        for iCh, ChName in enumerate(SP['ChNames']):

            fig, ax = plt.subplots(1,1,figsize=(6,2))    
            pl.clear_axes(np.array([ax]))    
            fig.suptitle(f"{sub} {ChName}")

            pl.channel_position_plot(ax,SP['coords'],iCh)

            fig_name = os.path.join(FiguresPath,f"elec_positions_{sub}_{ChName}_temp.pdf")
            fig.savefig(fig_name, format='pdf') 
            plt.close()

        cf.concatenate_pdfs(FiguresPath,'temp',f"elec_positions_{sub}.pdf", remove=True)



def metadata_overview(features, query=None, subs=cfg.subs, key=None, plot_subs = True, plot = True, save = True):

    
    print("\n======== metadata overview ========================================\n")

    
    # colect metadata 
    metadata_list = []
    for sub in cfg.subs: 
        SP = cf.sub_params(sub)
        sub_metadata = pd.read_csv(SP['metadata_file'])
        if query!=None: sub_metadata=sub_metadata.query(query)
        metadata_list += [sub_metadata]

    # make figure
    fig, ax = plt.subplots(1+len(subs)*plot_subs,len(features),figsize=(2*len(features),2*(1+len(subs)*plot_subs)))  
                           
    # add extra dimension if missing
    if not plot_subs: ax = np.array([ax])

    fig.suptitle(f"Metadata")

    metadata_list = [pd.concat(metadata_list)] + metadata_list*plot_subs
    labels = ['All subjects'] + subs*plot_subs


    iterator =  product(enumerate(metadata_list),enumerate(features))
    for (i_m, metadata),(i_f, feature) in iterator: 


        # OPTION 1: plot all epochs together
        if key == None:
            # plot all subs together
            data = metadata[feature].values
            x = np.arange(data.min()-1.5,data.max()+2)
            h, _ = np.histogram(data,x,density=True)
            pl.trace_plot(ax[i_m,i_f], x[:-1]+0.5, h, ylabel = '' + labels[i_m]*(i_f==0), title = '' + feature*(i_m==len(metadata_list)-1))
            ax[i_m,i_f].set_xticks(np.arange(data.min(),data.max()+2))

        
        # OPTION 2: separate epochs by condition
        else:
            conditions=list(set(metadata[key]))
            conditions.sort()
            queries = [f"{key} =='{c}'" for c in conditions]
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(conditions)))
            for i_q, query in enumerate(queries):
                data = metadata[feature].values
                x = np.arange(data.min()-1.5,data.max()+2)
                data_query = metadata.query(query)[feature].values
                h, _ = np.histogram(data_query,x,density=True)
                pl.trace_plot(ax[i_m,i_f], x[:-1]+0.5, h,  ylabel = '' + labels[i_m]*(i_f==0), xlabel = '' + feature*(i_m==len(metadata_list)-1) ,color=colors[i_q],label = conditions[i_q])               
            ax[i_m,i_f].set_xticks(np.arange(data.min(),data.max()+2))
    ax[0,-1].legend(frameon=False,loc=(0.5,0.5))


    fig.subplots_adjust(top=0.8,bottom=0.2,wspace=0.3, hspace=0.3)
    if save: 
        FiguresPath =cf.check_path(['..','Figures' , 'overview'+ cfg.out])
        fig_name = os.path.join(FiguresPath,f"metadata.pdf")
        fig.savefig(fig_name, format='pdf') 
    if plot: plt.show()
    else: plt.close()












'''
def channel_overview(mode ,sub, ChName, EpochsList=None, bands = cfg.bands, EpochsSetups=cfg.EpochsSetups,key=None, plot = True, save = False, temp=False):
    """ Same as subject overview but for one single channel
    

    Parameters
    ----------
    key :  str
        metadata key to plot separate traces by metadata value
    """

    # load subject params
    SP = cf.sub_params(sub)
    #if ChName not in SP['ChNames']: sys.exit(f"{sub} {ChName} not found  (o_0) ")

    # make figure
    fig, ax = plt.subplots(len(bands) + 1,len(EpochsSetups),figsize=(5*len(EpochsSetups),(1. +0.*(mode=='TrialWise'))*1.5*(len(bands) + 1)))  
         
    # add extra dimension if missing
    if len(EpochsSetups) == 1: ax = np.array([ax]).T

    fig.suptitle(f"{sub} {ChName}")#\n{SP['gyri'][SP['ChNames'].index(ChName)]}")
   
    pl.clear_axes(ax[0,:])   


    # LOOP OVER BANDS AND EPOCH SETUPS -----------------------------------------
    iterator =  product(enumerate(bands),enumerate(EpochsSetups))
    for i, ((iBand, band),(iEp, EpochsSetup)) in enumerate(iterator): 

        if EpochsList == None: epochs = cf.load_epochs(sub,band,EpochsSetup)
        else: epochs = EpochsList[i].copy()

        # this cannot be done outside the loop because positions are read from epochs
        if i==0:
            pl.channel_position_plot(ax[0,0],cf.get_coords(epochs,picks=[ChName]))
            #try: pl.channel_position_plot(ax[0,0],cf.get_coords(epochs,picks=[ChName]))
            #except: print('Channel positions not found')

        epochs = epochs.pick_channels([ChName])

        xticks = [x*SP['soa'] for x in EpochsSetup['xticks']]





        # TRACE PLOT -----------------------------------------------------------
        if mode =='avg':

            if band['method'] in [None,'filter']: ylabel = f"{band['name']} band\n"*(iEp==0) + f"\nV (z-scored)"
            elif band['method'] in ['complex']: ylabel = f"{band['name']} band\n"*(iEp==0) + f"\nITC"
            else: ylabel = f"{band['name']} power\n"*(iEp==0) + f"\ndB (z-scored)"

            # OPTION 1: plot all epochs together ----------------------------------
            if key == None:
                x = np.squeeze(epochs.get_data(picks=ChName))
                
                # plot avg trace
                pl.trace_plot(ax[iBand+1,iEp], epochs.times, x, ylabel = ylabel, 
                                title ='' + (iBand==0)*EpochsSetup['name'], vlines = [SP['soa']*x for x in EpochsSetup['xticks']], plot_sem = True)

            # OPTION 2: separate epochs by condition -------------------------------
            else:
                conditions=list(set(epochs.metadata[key]))
                conditions.sort()
                if isinstance(conditions[0],str):
                    queries = [f"{key} =='{c}'" for c in conditions]
                elif len(conditions)>10: 
                    conditions = np.linspace(min(conditions),max(conditions),5)
                    queries = [f"{key} >={conditions[i]} and {key} <{conditions[i+1]}" for i in range(len(conditions)-1)]
                else:
                     queries = [f"{key} =={c}" for c in conditions]
                colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(conditions)))
                for i_q, query in enumerate(queries):
                    x = np.squeeze(epochs.copy()[query].get_data(picks=ChName))
                    # plot avg trace per condition
                    pl.trace_plot(ax[iBand+1,iEp], epochs.times, x,  ylabel = '' + band['name']*(iEp==0), 
                       title = '' + EpochsSetup['name']*(iBand==0), vlines = xticks, 
                                    plot_sem = True, color=colors[i_q],label = conditions[i_q])

        # TrialWise PLOT -----------------------------------------------------------
        if mode == 'TrialWise':

            if band['method'] in [None,'filter']: ylabel, cb = f"{band['name']} band", f"V (z-scored)"
            elif band['method'] in ['complex']: ylabel, cb =f"{band['name']} band", r"$\theta$"
            else: ylabel, cb = f"{band['name']} power", f"dB (z-scored)"

            try: key = EpochsSetup['order']
            except: pass
            if key!=None:
                values = epochs.metadata[key].values
                order =np.argsort(values)
                yticks = [j for j, x in enumerate(np.diff(values[order])!=0) if x]
                yticklabels = [str(values[order][idx]) for idx in yticks]
                ylabel=(iEp==0)*ylabel + f"\n\n{key}"
            else: 
                order = []
                yticks = []
                yticklabels = []
                ylabel=(iEp==0)*ylabel + f"\n\nepoch"

            m = np.squeeze(epochs.copy().get_data(picks=ChName))
            if isinstance(m[0,0],complex): 
                m = np.angle(m)
                vmin, vmax = -np.pi, np.pi
            else:
                m = (m-m.mean())/m.std()
                vmin = np.percentile(m,15)
                vmax = np.percentile(m,85)

            pl.imshow_plot(ax[iBand+1,iEp], m,vmin=vmin,vmax=vmax, ylabel = ylabel, xlims  = epochs.times,
                                    title = '' + EpochsSetup['name']*(iBand==0), 
                        yticks = yticks, yticklabels = yticklabels, vlines = xticks,
                                colorbar = cb,cmap='jet',order=order)


    ax[-1,0].set_xlabel('t (s)')
    if key!=None and mode=='avg': ax[-1,0].legend(frameon=False,title=key)  # ,loc=(0.9,0.9)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    if save: 
        FiguresPath = cf.check_path(['..','Figures' , 'overview' + cfg.out ])
        fig_name = os.path.join(FiguresPath,f"{sub}_{ChName.replace('ROI','000')}_{mode}{'_temp'*temp}.pdf")
        fig.savefig(fig_name, format='pdf') 
    if plot: plt.show()
    else: plt.close()  





def ROI_positions(subs=cfg.subs):
    """ Make one figure per subject with the position of the ROIs
    """

    print("\n======== ROI positions ==============================================\n")

    FiguresPath =cf.check_path(['..','Figures' , 'overview'+ cfg.out])

    # iterate over subjects
    for sub in subs:

        cf.print_d(f'ROI positions for {sub}')

        # load subject params
        SP = cf.sub_params(sub)
        # load raw
        FileName = os.path.join(SP['DerivativesPath'],f"{sub}_raw.fif")
        raw = mne.io.read_raw_fif(FileName, preload=False)

        # chnames, coords, ROI labels
        ChNames = [ch for ch in rawChNames if ch[:3]!='ROI']
        coords = cf.get_coords(raw,ChNames)
        ROI_labels = cf.get_ROI_labels(coords)

        ROIs = sorted(list(set(ROI_labels)))
        print(ROIs)
        cmap = plt.get_cmap('tab20')(np.linspace(0, 1, len(ROIs)))
        colors = [cmap[ROIs.index(ROI_label)] for ROI_label in ROI_labels]


        view = plotting.view_markers(coords, marker_color=colors)
        FileName = os.path.join(cf.check_path(['..','Figures', 'overview' + cfg.out]),f"{sub}_ROIpositions.html")
        view.save_as_html(FileName)






def ROI_overview(ROIs, subs=cfg.subs, bands = cfg.bands, EpochsSetups=cfg.EpochsSetups,
                        avg=True, TrialWise=True, key=None):
    """ Make one figure per ROI with
    position and one evoked plot per band

    Parameters
    ----------
    key :  str
        metadata key to plot separate traces by metadata value
    """


    iterator1 =  list(product(enumerate(ROIs),enumerate(subs)))
    TimeStart = time.time()        
    for i1, ((i_ROI, ROI),(i_sub, sub)) in enumerate(iterator1): 
        cf.display_progress(f"{ROI}, {sub}", i1, len(iterator1),TimeStart) 

        SP = cf.sub_params(sub)

        # preload raw data for all bands
        epochs = []
        iterator =  product(enumerate(bands),enumerate(EpochsSetups))
        for i, ((iBand, band),(iEp, EpochsSetup)) in enumerate(iterator): 
            #cf.print_d(f"Loading {band['name']} epochs")
            # epochs
            try: epochs += [cf.load_epochs(sub,band,EpochsSetup).resample(100)]
            except: epochs += [cf.load_epochs(sub,band,EpochsSetup)]


        ROI_labels = cf.get_ROI_labels(cf.get_coords(epochs[0]))

        # skip subject if there are no channels in the ROI
        if ROI[4:] in ROI_labels: 
            if avg: channel_overview('avg', sub, ROI, EpochsList = epochs, bands = bands, EpochsSetups=EpochsSetups, key=key, plot = False, save = True, temp = True)
            if TrialWise:  channel_overview('TrialWise',sub, ROI, EpochsList = epochs, bands = bands, EpochsSetups=EpochsSetups, key=key, plot = False, save = True, temp = True)
            if avg: channel_overview('avg', sub, ROI, EpochsList = epochs, bands = bands, EpochsSetups=EpochsSetups, key=key, plot = False, save = True, temp = False)
            if TrialWise:  channel_overview('TrialWise',sub, ROI, EpochsList = epochs, bands = bands, EpochsSetups=EpochsSetups, key=key, plot = False, save = True, temp = False)

            ChNames = [ch for j, ch in enumerate(epochs[0].ch_names) if ROI_labels[j]==ROI[4:] and ch[:3]!='ROI']
            for ChName in ChNames:            
                if ChName not in epochs[0].ch_names: 
                    print(f"{ChName} not in {sub}")
                    continue

                if avg: channel_overview('avg', sub, ChName, EpochsList = epochs, bands = bands, EpochsSetups=EpochsSetups, key=key, plot = False, save = True, temp = True)
                if TrialWise:  channel_overview('TrialWise',sub, ChName, EpochsList = epochs, bands = bands, EpochsSetups=EpochsSetups, key=key, plot = False, save = True, temp = True)

        if i_sub == len(subs)-1:
            if key == None: FileName = f"{ROI}_overview.pdf"
            else: FileName = f"{ROI}_overview_{key}.pdf"
            FiguresPath = cf.check_path(['..','Figures', 'overview'  + cfg.out ])
            cf.concatenate_pdfs(FiguresPath,'temp',FileName, remove=True)

def ROI_overview2(ROIs, subs=cfg.subs, bands = cfg.bands, EpochsSetups=cfg.EpochsSetups):
    """ Make one figure per ROI with
    position and one evoked plot per band

    Parameters
    ----------
    key :  str
        metadata key to plot separate traces by metadata value
    """


    TimeStart=time.time()
    for i_sub, sub in enumerate(subs): 
        cf.display_progress(f"{sub}", i_sub, len(subs),TimeStart) 

        SP = cf.sub_params(sub)

        # preload raw data for all bands
        epochs = []
        iterator =  product(enumerate(bands),enumerate(EpochsSetups))
        for i, ((iBand, band),(iEp, EpochsSetup)) in enumerate(iterator): 
            # phase epochs cannot be resampled because they're complex
            try: epochs += [cf.load_epochs(sub,band,EpochsSetup).resample(100)]
            except: epochs += [cf.load_epochs(sub,band,EpochsSetup)]

 
        # get coords and colors
        coords = []
        coefs = np.array([])
        for i, ROI in enumerate(ROIs):
            c = cf.get_coords(epochs[0],picks=[ROI])
            coords +=c.tolist()
            coefs = np.append(coefs,len(c)*[i])

        if len(coefs)==0: continue 

        # make figure
        fig, ax = plt.subplots(len(bands) + 1,len(EpochsSetups),figsize=(6*len(EpochsSetups),2*(len(bands) + 1)))               
        # add extra dimension if missing
        if len(EpochsSetups) == 1: ax = np.array([ax]).T
        fig.suptitle(f"{sub}")
        pl.clear_axes(ax[0,:])   

        pl.brain_plot(ax[0,0],coords,coefs,mode='ROIs',colorbar='',title='',mask=[],interval=[],ylabel='',ROIs=ROIs)

        iterator =  product(enumerate(bands),enumerate(EpochsSetups))
        for i, ((iBand, band),(iEp, EpochsSetup)) in enumerate(iterator): 

            if band['method'] in [None,'filter']: ylabel = f"{band['name']} band\n"*(iEp==0) + f"\nV (z-scored)"
            elif band['method'] in ['complex']: ylabel = f"{band['name']} band\n"*(iEp==0) + f"\nITC"
            else: ylabel = f"{band['name']} power\n"*(iEp==0) + f"\ndB (z-scored)"

            for i_ROI, ROI in enumerate(ROIs):
                try: x = np.squeeze(epochs[i].get_data(picks=ROI))            
                except: continue
                # plot avg trace
                pl.trace_plot(ax[iBand+1,iEp], epochs[i].times, x, ylabel = ylabel, 
                                title ='' + (iBand==0)*EpochsSetup['name'], vlines = [SP['soa']*x for x in EpochsSetup['xticks']],color=plt.get_cmap('tab20')(i_ROI/20.))


        FiguresPath = cf.check_path(['..','Figures' , 'overview' + cfg.out ])
        fig_name = os.path.join(FiguresPath,f"{sub}_ROIs_temp.pdf")
        fig.savefig(fig_name, format='pdf') 


    FileName = f"ROI_overview_light.pdf"
    FiguresPath = cf.check_path(['..','Figures' , 'overview' + cfg.out ])
    cf.concatenate_pdfs(FiguresPath,'temp',FileName, remove=True)
        
'''







