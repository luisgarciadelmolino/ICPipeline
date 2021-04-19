#############################################################################################################
#    
#   CONFIG FILE 
#                                                                                                           
#############################################################################################################


subs = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11" , "sub-12", "sub-13", "sub-14"]

#subs = [ "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09"]



# __PREPROCESSING PARAMETERS_________________________________________________________

source_srate = 2000.0           # sampling rate of source data in Hz
soa = 0.534                     # time between stimuli
srate = 500.0                   # sampling rate of preprocessed in Hz
landline_noise = 60             # land line freq for linenoise filtering (60 USA, 50 Eur,...)
hpf = 0.05          # high pass filter for raw data in HZ
lpf = None          # high pass filter for raw data in HZ

RefMethod = "laplacian"       # options are 'laplacian', 'bipolar', 'cmr' or 'None'

InputFormat = 'raw'

# __ROI PARAMETERS_________________________________________________________
ROITolerance = 1.5
replacements = { }

# __CLEANING PARAMETERS_______________________________________________________________

vlow, vhigh = -2, 3       
dvlow, dvhigh = -2, 3
slow, shigh = -2, .5

# smoothing, clipping and rescaling
smooth = 0.00     # std of gaussian convolution (in seconds), set to 0 for NO convolution
clip = 0        # number of IQR units above and below the median to clip. Set to 0 for NO clip
zscore = False

# __TFR PARAMETERS_____________________________________________________________________

bands = [     
          #{"name": "broad", "fmin" : None,  "fmax" : None, "method": None},   
          #{"name": "theta", "fmin" : 4,  "fmax" : 7 , "method": "filter" }, 
          #{"name": "alpha", "fmin" : 8,  "fmax" : 13, "method": "hilbert" },  
          #{"name": "beta",  "fmin" : 13, "fmax" : 30, "method": "hilbert" },    
          #{"name": "gamma-w", "fmin" : 30, "fmax" : 70, "method": "wavelet",'nf':5,'norm':'std'},  
          #{"name": "gamma-h", "fmin" : 30, "fmax" : 70, "method": "hilbert" },  
          #{"name": "hgamma-w", "fmin" : 70, "fmax" : 140, "method": "wavelet",'nf':5,'norm':'std'},
          {"name": "hgamma-h", "fmin" : 70, "fmax" : 140, "method": "hilbert" },
          #{"name": "bgamma-w", "fmin" : 30, "fmax" : 140, "method": "wavelet",'nf':5,'norm':'std'},
        ] 

# __EPOCHING PARAMETERS_______________________________________________________________
baseline = {'key' : 'position', "values" : [ 1 ], "tmin" : -.2, "tmax" : 0}
EpochsSetups =  [  
            {"name" : "sent-begining",'key' : 'position', "values" : [ 1 ], "tmin" : -1., "tmax" : 7.5, 'baseline' : None, "xticks": [n for n in range(8)],"smooth":0.,"clip":3,"split":"sentlength" }, 
# {"name" : "sent-end",'key' : 'position_end', "values" : [ -1 ], "tmin" : -7.5, "tmax" : 1, 'baseline' : high_pass_filter, "xticks": [-n for n in range(8)] }, 
           {"name" : "word" ,'key' : "eventtype" ,"values" :  ["wordonset"], "tmin" : -0.25,"tmax" : .75, 'baseline' : None, "xticks": [0.5*n for n in range(3)],"smooth":0.,"clip":3,"split":"position"}
          ]


# __STATISTICAL INFERENCE PARAMETERS_________________________________________________

n_permutations = 100       # for PCT
t_adj = 0.05     # time points withn 0.05s are connected
s_adj = 2


# __REGRESSION PARAMETERS________________________________________________________________

fdr = False
features=['w_length','uni_freq','bi_freq','func_word','opennodes','nodesclosing','position']
n_splits = 5

#tws = [[0.3,0.5]]
#tws = [[0.3,0.35],[0.35,0.4],[0.4,0.45],[0.45,0.5],[.5,.55],[0.55,0.6]]
tws = .1 # twsize in seconds
ntws = 8
tmin=0
tmax=0.6

# __OTHER PARAMETERS __

# significance threshold
alpha = 0.05

skip = True                       # do not recompute raw if it already exists
debug = 0                       # int, run for small number of channels, run for all if P['debug']==0 
debugMNE = False                # change MNE verbosity 


import datetime
out = '_'+RefMethod+datetime.datetime.now().strftime("_%Y.%m.%d")

DerivativesTag = '_'+RefMethod



