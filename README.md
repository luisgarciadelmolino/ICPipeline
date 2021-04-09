<a id='top'></a>
# IntraCranial pipeline description

1. Modules
1. Folder structure
1. IO

## Modules

1. **Preprocessing**: Make mne raw files, basic cleaning (notch filter, resaple, low/high pass filter), rejection, referencing
1. **Inference**:
1. **Regressions**:

Other
1. **Figures**:
1. **Overview**: make figures with evoked responses for all channels / subjects / bands.
1. **Common functions**:


#### Tip to navigate the code:
Most functions come in pairs of low level and high level functions (names follow the pattern `func()`,`func_wrap()`)

**Low level functions** take genereal mne objects as arguments (raw, epochs). They are meant to be easily implemented in any MNE based code.  
**High level functions** are basically wraps for low level functions, they perform four main operations: (i) Loop over subjects/bands/conditions... (ii) load data, (iii) call low level functions, (iv) save results and or modified data. Wraps need to "know" the folder structure (among other things) and therefore are functional as long as the basic structure of the pipeline is respected.


## Folder structure

BIDS naming and folder structure

- **Code**
    - `ICPipeline.ipynb`
    - `static_parameters.py`
    - `run.py`
    - utils
        - modules
- **Data** 
    - raw
        - sub-XX 
            - `sub-XX_metadata.csv`
            - `sub-XX_raw.fif`
    - derivatives
        - sub-XX
            - `sub-XX_raw.fif`
            - `sub-XX_band-XXXX_raw.fif`
- **Figures**
    - overview
        - `sub-XX_overview.pdf`
    
## IO

### In

Raw data (from `Data/raw/sub-XX`)

1. ieeg data in one single `raw.fif` file with mne raw objects with **montage included**, **line-noise removed** and **resampled** to desired sampling rate.
1. metadata in a `csv` file

If montage is not included coords are read as `nan` for compatibility with the rest of the pipeline.


### Out

Preprocessed data (at `Data/derivatives/sub-XX`)

1. `mne.raw` stored as `sub-XX_band-XXX_raw.fif`
