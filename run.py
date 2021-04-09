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
#pr.rejection_wrap()
#pr.referencing_wrap()
#pr.raw2TFR_wrap()
#pr.clear_ROIs()
#pr.add_ROI_from_atlas(subs=cfg.subs[:11])

# OVERVIEW 

'''
ep_setups = [  
    {"name" : "sent-begining",'key' : 'position', "values" : [ 1 ], "tmin" : -1., "tmax" : 7.5, 'baseline' : None, "xticks": [n for n in range(8)],"smooth":0.02,"clip":0,"split":"sentlength","query":"blocktype=='Main'" }, 
    {"name" : "word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : -0.25,"tmax" : .75, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.02,"clip":0,"split":"position","query":"blocktype=='Main' and position>1"}
]

ov.subject_overview(ep_setups=ep_setups)

ep_setups = [  
    {"name" : "sent-begining",'key' : 'position', "values" : [ 1 ], "tmin" : -1., "tmax" : 7.5, 'baseline' : None, "xticks": [n for n in range(8)],"smooth":0.02,"clip":0}, 
    {"name" : "word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : -0.25,"tmax" : .75, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.02,"clip":0,"query":"position>1"}
]

ov.subject_overview(ep_setups=ep_setups,key='trialtipe')

#ov.electrode_positions()
#features=['w_length','n_vowels','n_consonants','uni_freq','bi_freq','func_word','opennodes','nodesclosing']
#ov.metadata_overview(features=features,query='eventtype=="wordonset"',key='trialtype',plot_subs=False)
'''

# INFERENCE


ep_setup =  {"name" : "word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .6, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"split":"position","query":"blocktype=='Localizer'"}

nf.PCT(ep_setup,EOIlabel='TaskResponsive')



ep_setups =  [
           {"name" : "last-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .3, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and is_last_word==1"},
 {"name" : "mid-word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .3, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.0,"clip":5,"query":"blocktype=='Localizer' and position>1 and position_end<-1"},
]

nf.ttest(ep_setups,picks='EOI-TaskResponsive')








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
         'ep_setup': {'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .6 , "level" : "word" , "xticks": [0.5*n for n in range(3)],'baseline': cfg.baseline}
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
         'ep_setup': {'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : 0.,"tmax" : .6 , "level" : "word" , "xticks": [0.5*n for n in range(3)],'baseline':None}
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



















