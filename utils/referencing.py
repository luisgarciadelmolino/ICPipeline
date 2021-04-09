import mne
import numpy as np
import os.path as op
import os
from scipy import signal
from copy import deepcopy


def bipolar_referencing(raw):
    ''' 
    Apply bipolar reference for SEEG, 
    ie referencing with the average of the neighboring electrode.

    Expects that the channels in raw.ch_names are ordered by probe, 
    and with ascending order for each probe, and the names consists of 
    the probe name followed by the position on the probe.
    eg: ['OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'OF6', 'OF7', 'OF8', 'OF9', 
    'OF10', 'OF11', 'OF12', 'OF13', 'OF14', 'H1', 'H2', 'H3', 'H4', 'H5', 
    'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', ...]
    
    WATCH-OUT: this will take the closest electrode on the probe, 
    meaning that if the neighboring electrode is missing for some 
    reason (eg: rejected before applying the reference) then the next
    electrode will be used for referencing. 
    '''


    reference_ch = [ch for ch in raw.copy().pick('eeg').info['ch_names']]
    anodes = reference_ch[0:-1]
    cathodes = reference_ch[1::]
    to_del = []
    for i, (a, c) in enumerate(zip(anodes,cathodes)):
         if [i for i in a if not i.isdigit()] != [i for i in c if not i.isdigit()]:
              to_del.append(i)
    for idx in to_del[::-1]:
         del anodes[idx]
         del cathodes[idx]
    # for a, c in zip(anodes, cathodes): print(a,c)

    raw = mne.set_bipolar_reference(raw.copy(), anodes, cathodes)

    return raw


def get_position(ch):
    # get the channel position on the probe by keeping digits only
    return ''.join(c for c in ch if c.isdigit())

def get_label(ch):
    # get the electrode shaft label by keeping letters only
    return ''.join(c for c in ch if not c.isdigit())


def laplacian_referencing(raw):
    ''' 
    Apply laplacian reference for SEEG, 
    ie referencing with the average of the 2 neighboring electrodes.

    Expects that the channels in raw.ch_names are ordered by probe, 
    and with ascending order for each probe, and the names consists of 
    the probe name followed by the position on the probe.
    eg: ['OF1', 'OF2', 'OF3', 'OF4', 'OF5', 'OF6', 'OF7', 'OF8', 'OF9', 
    'OF10', 'OF11', 'OF12', 'OF13', 'OF14', 'H1', 'H2', 'H3', 'H4', 'H5', 
    'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', ...]
    
    WATCH-OUT: this will take the closest electrodes on the probe, 
    meaning that if the neighboring electrode is missing for some 
    reason (eg: rejected before applying the reference) then the next
    electrode will be used for referencing. 
    '''



    ch_names = raw.ch_names
    ch_probes = [get_label(ch) for ch in ch_names]

    data = raw.get_data(picks=ch_names)

    # channels at the extremes of the probe are discarded, the remaining channels are stored here to be extracted later
    picks = []

    # create a dummy raw that contains the references
    reference_data = np.zeros_like(data)

    for i_ch, ch_name in enumerate(ch_names[:-1]):     
        reference_data[i_ch] = np.mean([data[i_ch-1], data[i_ch+1]], 0)
        if ch_probes[i_ch -1]==ch_probes[i_ch +1]: picks +=[ch_name]

    info = raw.info.copy()
    ref_raw = mne.io.RawArray(data-reference_data, info)
    
    return ref_raw.pick(picks), picks


def common_median_referencing(raw):
    ''' 
    Apply CMR reference for SEEG, 
    ie referencing with the median over all electrodes.
    '''

    ref = np.median(raw.get_data(), 0).reshape((1, -1))
    ref_info = mne.create_info(ch_names=['CMR'], sfreq=raw.info['sfreq'], ch_types='misc')
    ref_ch = mne.io.RawArray(ref, ref_info)
    raw = raw.add_channels([ref_ch])
    raw = raw.set_eeg_reference(ref_channels=['CMR'])
    raw = raw.drop_channels(['CMR'])

    return raw
