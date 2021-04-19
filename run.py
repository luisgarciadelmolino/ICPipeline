# import packages
import numpy as np
import pandas as pd
import sys, os, glob, csv, json, mne, time

# import local functions
import utils.preprocessing as pr
import utils.common_functions as cf
import utils.overview as ov
import utils.inference as nf
import utils.regressions as rg
#import utils.figures as fg

import config as cfg

mne.set_log_level("CRITICAL")


# PREPROCESSING
#pr.source2raw()
#pr.ch_info()    
#pr.rejection_wrap()
#pr.referencing_wrap()
#pr.raw2TFR_wrap()
#pr.clear_ROIs()
#pr.add_ROI_from_atlas()



# OVERVIEW 


EpochsSetupsPlots = [  
    {
        "name" : "SentBegining",
        'key' : 'position', "values" : [ 1 ], "tmin" : -1.5, "tmax" : 8, 
        'baseline' : 'hpf', "xticks" : [n for n in range(6)], "smooth" : 0.02, "clip":0,"order" : "sentlength"
    }, 
    {
        "name" : "SentEnd",
        'key' : 'position_end', "values" : [ -1 ], "tmin" : -8, "tmax" : 1.5,
        'baseline' : 'hpf', "xticks": [-n for n in range(6)],"smooth":0.02,"clip":0, "order": "sentlength" 
    }, 
    {
        "name" : "word",
        'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : -0.25,"tmax" : .75,
        'baseline' : 'hpf', "xticks": [0.5*n for n in range(3)],"smooth":0.02,"clip":0,"order":"position"
    }
]



# TASK RESPONSIVE CHANNELS ##########################
# avg activation during word epochs > baseline

EpochsSetups =  [
    {   
        "name" : "word",
        'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5,
        #"sample":"trialtype=='sent_main'",
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
    {
        "name" : "baseline",
        'key' : "position" ,"values" :  [1], "tmin" : -1.25,"tmax" : -.75,
        #"sample":"trialtype=='sent_main'" , 
        'baseline' : 'hpf', "smooth":0.00,"clip":0
    },
]

#nf.ttest(EpochsSetups, EOIlabel = 'EOI-TaskResponsive', split = [[0.,0.3],[0.150,0.450],[0.2,0.5]])
#ov.subject_overview(EpochsSetups = EpochsSetupsPlots, picks = 'EOI-TaskResponsive', key='trialtype')



# ENG / JABBA ##########################
EpochsSetups =  [
    {
        "name" : "english",
        'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5,
        "sample":"trialtype=='sent_main' and func_word == 0", 
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
    {
        "name" : "jabba",
        'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5,
        "sample":"trialtype=='jabba' and func_word == 0",
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
]
nf.ttest(EpochsSetups,picks='EOI-TaskResponsive',EOIlabel='Jabba',alternative='two-sided')
ov.subject_overview(EpochsSetups=EpochsSetupsPlots,picks='EOI-Jabba',key='trialtype')




# RAMPING ##########################
EpochsSetups =  [
    {
        "name" : "LastWord",
        'key' : "position_end" ,"values" :  [-1], "tmin" : 0.,"tmax" : .5,
        "sample":"trialtype=='sent_main'", 
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
    {
        "name" : "SecondWord",
        'key' : "position" ,"values" :  [2], "tmin" : 0.,"tmax" : .5,
        "sample":"trialtype=='sent_main'",
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
]
nf.ttest(EpochsSetups,picks='EOI-TaskResponsive',EOIlabel='Ramping',paired=True)
ov.subject_overview(EpochsSetups=EpochsSetupsPlots,picks='EOI-Ramping',key='trialtype')



# INHIBITED CHANNELS ##########################
# avg activation during word epochs < baseline
EpochsSetups =  [
    {
        "name" : "word",
        'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5,
        #"sample":"trialtype=='sent_main'" , 
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
    {
        "name" : "baseline",
        'key' : "position" ,"values" :  [1], "tmin" : -1.25,"tmax" : -.75,
        #"sample":"trialtype=='sent_main'", 
        'baseline' : 'hpf', "smooth":0.0,"clip":0
    },
]

nf.ttest(EpochsSetups, EOIlabel = 'Inhibited', alternative = 'less')
ov.subject_overview(EpochsSetups = EpochsSetupsPlots, picks = 'EOI-Inhibited', key = 'trialtype')





'''
# Wlist / Sent ##########################
EpochsSetups =  [
    {"name" : "sentence","sample":"trialtype=='sent_loc'" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5, 'baseline' : 'hpf', "smooth":0.0,"clip":0},
    {"name" : "WordList","sample":"trialtype=='word_list'" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5, 'baseline' : 'hpf', "smooth":0.0,"clip":0},
]
nf.ttest(EpochsSetups, picks = 'EOI-TaskResponsive', EOIlabel = 'WordList', alternative = 'two-sided', split = [[0.,0.3],[0.150,0.450],[0.2,0.5]])
ov.subject_overview(EpochsSetups = EpochsSetupsPlots, picks = 'EOI-WordList', key = 'trialtype')
'''





'''
# RAMPING CHANNELS ###########################
# avg activation during  last word > first word
EpochsSetups =  [
    {"name" : "last-word","sample":"trialtype=='sent_main'" ,'key' : "position_end" ,"values" :  [-1], "tmin" : 0.,"tmax" : .5, 'baseline' : 'hpf', "smooth":0.0,"clip":0},
    {"name" : "first-word","sample":"trialtype=='sent_main'" ,'key' : "position" ,"values" :  [1], "tmin" : 0,"tmax" : .5, 'baseline' : 'hpf', "smooth":0.0,"clip":0},
]
nf.ttest(EpochsSetups,EOIlabel='Ramp1stword',split=[[0.,0.3],[0.150,0.450],[0.2,0.5]])
ov.SubjectOverview(EpochsSetups=EpochsSetups_overview,picks='EOI-Ramp1stword',key='trialtype')

# avg activation during  last word > first word
EpochsSetups =  [
    {"name" : "last-word","sample":"trialtype=='sent_main'" ,'key' : "position_end" ,"values" :  [-1], "tmin" : 0.,"tmax" : .5, 'baseline' : 'hpf', "smooth":0.0,"clip":0},
    {"name" : "secondt-word","sample":"trialtype=='sent_main'" ,'key' : "position" ,"values" :  [2], "tmin" : 0,"tmax" : .5, 'baseline' : 'hpf', "smooth":0.0,"clip":0},
]
nf.ttest(EpochsSetups,EOIlabel='Ramp2ndword',split=[[0.,0.3],[0.150,0.450],[0.2,0.5]])
ov.SubjectOverview(EpochsSetups=EpochsSetups_overview,picks='EOI-Ramp2ndword',key='trialtype')
'''





'''

EpochsSetups =  [
           {"name" : "word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.01,"clip":5,"query":"blocktype=='Localizer' and  position>2."},
           {"name" : "previous-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : -0.25,"tmax" : -.0, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.01,"clip":5,"query":"blocktype=='Localizer' and  position>2."},
]

nf.ttest(EpochsSetups,paired=True)



EpochsSetups =  [
           {"name" : "last3words" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .5, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.01,"clip":5,"query":"blocktype=='Localizer' and  position_end>-4."},
           {"name" : "first3words" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : 0.5, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.01,"clip":.3, "query":"blocktype=='Localizer' and  position<4."},
]

nf.ttest(EpochsSetups,split=[[0.,0.25],[0.25,0.5]])


EpochsSetups =  [
           {"name" : "last-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .3, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and  position_end==-1."},
 {"name" : "mid-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .3, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and position>1 and position_end<-1"},
]

nf.ttest(EpochsSetups,picks='EOI-TR')





EpochsSetups =  [
           {"name" : "last-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.3,"tmax" : .6, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and  position_end>-2"},
 {"name" : "mid-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.3,"tmax" : .6, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and position>1 and position_end<-1"},
]

nf.ttest(EpochsSetups,EOIlabel='Ramping')



EpochsSetups =  [
           {"name" : "sent" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .3, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and  position_end>-2"},
 {"name" : "list" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .3, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and position>1 and position_end<-1"},
]

nf.ttest(EpochsSetups,EOIlabel='SentvsList')
'''







# REGRESSIONS
'''
bands = [     
          {"name": "broad", "fmin" : None,  "fmax" : None, "method": None},    
          {"name": "gamma", "fmin" : 30, "fmax" : 70, "method": "hilbert" },  
          {"name": "hgamma", "fmin" : 70, "fmax" : 140, "method": "hilbert" },
        ] 

model = {
         'tag':'sent-b-010',
         'query' : 'trialtype == "sent_main" and position>1', 
         'predictors': cfg.features,
         'predictors_stepwise': cfg.features,
         'EpochsSetup': {'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .6 , "level" : "word" , "xticks": [0.5*n for n in range(3)],'baseline': cfg.baseline}
        }

#rg.regression_wrap(model)
#rg.reg_statistics(model,predictors='all',plot=False)
rg.reg_maps(model,predictors=['position','opennodes','nodesclosing'],time=False,plot=False,bands=bands)




model['predictors_stepwise'] = ['position','opennodes','nodesclosing']
rg.reg_single_channel_wrap(model,name='_syntactic')



model = {
         'tag':'sent',
         'query' : 'trialtype == "sent_main" and position>1', 
         'predictors': cfg.features,
         'predictors_stepwise': cfg.features,
         'EpochsSetup': {'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .6 , "level" : "word" , "xticks": [0.5*n for n in range(3)],'baseline':None}
        }

rg.regression_wrap(model)
rg.reg_statistics(model,predictors='all',plot=False)
rg.reg_maps(model,predictors='all',plot=False)

model['predictors_stepwise'] = ['position','opennodes','nodesclosing']
rg.reg_single_channel_wrap(model,name='_syntactic',bands=bands)
'''





'''
# ENHANCE METADATA
import utils.linguistic_features as lf
for sub in cfg.subs:
    print(sub)
    dp = cf.sub_params(sub)
    metadata = pd.read_csv(dp['metadata_file'], delimiter=',')
    metadata = lf.enhance_metadata(metadata)
    metadata.to_csv(dp['metadata_file'],index=False)  
'''








print("\ndone!\n")



















