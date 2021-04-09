from ipdb import set_trace
import pandas as pd
import numpy as np
import nilearn
from nilearn import datasets, plotting
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
from nilearn import image
import nibabel as nib 
import pickle

strip_RL = True
strip_str = "_GD" if not strip_RL else ''


from utils.BNA2all import *
# BNA2subregion = {'A8m': ' Superior Frontal Gyrus  ', 'A8dl': ' Superior Frontal Gyrus  ', 'A9l': ' Superior Frontal Gyrus  ', 'A6dl': ' Superior Frontal Gyrus  ', 'A6m': ' Superior Frontal Gyrus  ', 'A9m': ' Superior Frontal Gyrus  ', 'A10m': ' Superior Frontal Gyrus  ', 'A9/46d': ' Middle Frontal Gyrus ', 'IFJ': ' Middle Frontal Gyrus ', 'A46': ' Middle Frontal Gyrus ', 'A9/46v': ' Middle Frontal Gyrus ', 'A8vl': ' Middle Frontal Gyrus ', 'A6vl': ' Middle Frontal Gyrus ', 'A10l': ' Middle Frontal Gyrus ', 'A44d': ' Inferior Frontal Gyrus', 'IFS': ' Inferior Frontal Gyrus', 'A45c': ' Inferior Frontal Gyrus', 'A45r': ' Inferior Frontal Gyrus', 'A44op': ' Inferior Frontal Gyrus', 'A44v': ' Inferior Frontal Gyrus', 'A14m': ' Orbital Gyrus', 'A12/47o': ' Orbital Gyrus', 'A11l': ' Orbital Gyrus', 'A11m': ' Orbital Gyrus', 'A13': ' Orbital Gyrus', 'A12/47l': ' Orbital Gyrus', 'A4hf': ' Precentral Gyrus', 'A6cdl': ' Precentral Gyrus', 'A4ul': ' Precentral Gyrus', 'A4t': ' Precentral Gyrus', 'A4tl': ' Precentral Gyrus', 'A6cvl': ' Precentral Gyrus', 'A1/2/3ll': ' Paracentral Lobule', 'A4ll': ' Paracentral Lobule', 'A38m': ' Superior Temporal Gyrus', 'A41/42': ' Superior Temporal Gyrus', 'TE1.0 and TE1.2': ' Superior Temporal Gyrus', 'A22c': ' Superior Temporal Gyrus', 'A38l': ' Superior Temporal Gyrus', 'A22r': ' Superior Temporal Gyrus', 'A21c': ' Middle Temporal Gyrus', 'A21r': ' Middle Temporal Gyrus', 'A37dl': ' Middle Temporal Gyrus', 'aSTS': ' Middle Temporal Gyrus', 'A20iv': ' Inferior Temporal Gyrus', 'A37elv': ' Inferior Temporal Gyrus', 'A20r': ' Inferior Temporal Gyrus', 'A20il': ' Inferior Temporal Gyrus', 'A37vl': ' Inferior Temporal Gyrus', 'A20cl': ' Inferior Temporal Gyrus', 'A20cv': ' Inferior Temporal Gyrus', 'A20rv': ' Fusiform Gyrus', 'A37mv': ' Fusiform Gyrus', 'A37lv': ' Fusiform Gyrus', 'A35/36r': ' Parahippocampal Gyrus', 'A35/36c': ' Parahippocampal Gyrus', 'TL': ' Parahippocampal Gyrus', 'A28/34': ' Parahippocampal Gyrus', 'TI': ' Parahippocampal Gyrus', 'TH': ' Parahippocampal Gyrus', 'rpSTS': ' posterior Superior Temporal Sulcus ', 'cpSTS': ' posterior Superior Temporal Sulcus ', 'A7r': ' Superior Parietal Lobule', 'A7c': ' Superior Parietal Lobule', 'A5l': ' Superior Parietal Lobule', 'A7pc': ' Superior Parietal Lobule', 'A7ip': ' Superior Parietal Lobule', 'A39c': ' Inferior Parietal Lobule', 'A39rd': ' Inferior Parietal Lobule', 'A40rd': ' Inferior Parietal Lobule', 'A40c': ' Inferior Parietal Lobule', 'A39rv': ' Inferior Parietal Lobule', 'A40rv': ' Inferior Parietal Lobule', 'A7m': ' Precuneus', 'A5m': ' Precuneus', 'dmPOS': ' Precuneus', 'A31': ' Precuneus', 'A1/2/3ulhf': ' Postcentral Gyrus', 'A1/2/3tonIa': ' Postcentral Gyrus', 'A2': ' Postcentral Gyrus', 'A1/2/3tru': ' Postcentral Gyrus', 'G': ' Insular Gyrus', 'vIa': ' Insular Gyrus', 'dIa': ' Insular Gyrus', 'vId/vIg': ' Insular Gyrus', 'dIg': ' Insular Gyrus', 'dId': ' Insular Gyrus', 'A23d': ' Cingulate Gyrus', 'A24rv': ' Cingulate Gyrus', 'A32p': ' Cingulate Gyrus', 'A23v': ' Cingulate Gyrus', 'A24cd': ' Cingulate Gyrus', 'A23c': ' Cingulate Gyrus', 'A32sg': ' Cingulate Gyrus', 'cLinG': ' MedioVentral Occipital Cortex', 'rCunG': ' MedioVentral Occipital Cortex', 'cCunG': ' MedioVentral Occipital Cortex', 'rLinG': ' MedioVentral Occipital Cortex', 'vmPOS': ' MedioVentral Occipital Cortex', 'mOccG': ' lateral Occipital Cortex', 'V5/MT+': ' lateral Occipital Cortex', 'OPC': ' lateral Occipital Cortex', 'iOccG': ' lateral Occipital Cortex', 'msOccG': ' lateral Occipital Cortex', 'lsOccG': ' lateral Occipital Cortex', 'mAmyg': ' Amygdala', 'lAmyg': ' Amygdala', 'rHipp': ' Hippocampus', 'cHipp': ' Hippocampus', 'vCa': ' Basal Ganglia', 'GP': ' Basal Ganglia', 'NAC': ' Basal Ganglia', 'vmPu': ' Basal Ganglia', 'dCa': ' Basal Ganglia', 'dlPu': ' Basal Ganglia', 'mPFtha': ' Thalamus', 'mPMtha': ' Thalamus', 'Stha': ' Thalamus', 'rTtha': ' Thalamus', 'PPtha': ' Thalamus', 'Otha': ' Thalamus', 'cTtha': ' Thalamus', 'lPFtha': ' Thalamus'}
# BNA2subregion["Unknown"] = "Unknown"
# BNA2subregion_abrev = {'A8m': 'SFG', 'A8dl': 'SFG', 'A9l': 'SFG', 'A6dl': 'SFG', 'A6m': 'SFG', 'A9m': 'SFG', 'A10m': 'SFG', 'A9/46d': 'MFG', 'IFJ': 'MFG', 'A46': 'MFG', 'A9/46v': 'MFG', 'A8vl': 'MFG', 'A6vl': 'MFG', 'A10l': 'MFG', 'A44d': 'IFG', 'IFS': 'IFG', 'A45c': 'IFG', 'A45r': 'IFG', 'A44op': 'IFG', 'A44v': 'IFG', 'A14m': 'OrG', 'A12/47o': 'OrG', 'A11l': 'OrG', 'A11m': 'OrG', 'A13': 'OrG', 'A12/47l': 'OrG', 'A4hf': 'PrG', 'A6cdl': 'PrG', 'A4ul': 'PrG', 'A4t': 'PrG', 'A4tl': 'PrG', 'A6cvl': 'PrG', 'A1/2/3ll': 'PCL', 'A4ll': 'PCL', 'A38m': 'STG', 'A41/42': 'STG', 'TE1.0 and TE1.2': 'STG', 'A22c': 'STG', 'A38l': 'STG', 'A22r': 'STG', 'A21c': 'MTG', 'A21r': 'MTG', 'A37dl': 'MTG', 'aSTS': 'MTG', 'A20iv': 'ITG', 'A37elv': 'ITG', 'A20r': 'ITG', 'A20il': 'ITG', 'A37vl': 'ITG', 'A20cl': 'ITG', 'A20cv': 'ITG', 'A20rv': 'FuG', 'A37mv': 'FuG', 'A37lv': 'FuG', 'A35/36r': 'PhG', 'A35/36c': 'PhG', 'TL': 'PhG', 'A28/34': 'PhG', 'TI': 'PhG', 'TH': 'PhG', 'rpSTS': 'pSTS', 'cpSTS': 'pSTS', 'A7r': 'SPL', 'A7c': 'SPL', 'A5l': 'SPL', 'A7pc': 'SPL', 'A7ip': 'SPL', 'A39c': 'IPL', 'A39rd': 'IPL', 'A40rd': 'IPL', 'A40c': 'IPL', 'A39rv': 'IPL', 'A40rv': 'IPL', 'A7m': 'Pcun', 'A5m': 'Pcun', 'dmPOS': 'Pcun', 'A31': 'Pcun', 'A1/2/3ulhf': 'PoG', 'A1/2/3tonIa': 'PoG', 'A2': 'PoG', 'A1/2/3tru': 'PoG', 'G': 'INS', 'vIa': 'INS', 'dIa': 'INS', 'vId/vIg': 'INS', 'dIg': 'INS', 'dId': 'INS', 'A23d': 'CG', 'A24rv': 'CG', 'A32p': 'CG', 'A23v': 'CG', 'A24cd': 'CG', 'A23c': 'CG', 'A32sg': 'CG', 'cLinG': 'MVOcC', 'rCunG': 'MVOcC', 'cCunG': 'MVOcC', 'rLinG': 'MVOcC', 'vmPOS': 'MVOcC', 'mOccG': 'LOcC', 'V5/MT+': 'LOcC', 'OPC': 'LOcC', 'iOccG': 'LOcC', 'msOccG': 'LOcC', 'lsOccG': 'LOcC', 'mAmyg': 'Amyg', 'lAmyg': 'Amyg', 'rHipp': 'Hipp', 'cHipp': 'Hipp', 'vCa': 'BG', 'GP': 'BG', 'NAC': 'BG', 'vmPu': 'BG', 'dCa': 'BG', 'dlPu': 'BG', 'mPFtha': 'Tha', 'mPMtha': 'Tha', 'Stha': 'Tha', 'rTtha': 'Tha', 'PPtha': 'Tha', 'Otha': 'Tha', 'cTtha': 'Tha', 'lPFtha': 'Tha'}
# BNA2subregion_abrev["Unknown"] = "Unknown"
# subregion_abrev2lobe = {'SFG': 'Frontal', 'MFG': 'Frontal', 'IFG': 'Frontal', 'OrG': 'Frontal', 'PrG': 'Frontal', 'PCL': 'Frontal', 'STG': 'Temporal', 'MTG': 'Temporal', 'ITG': 'Temporal', 'FuG': 'Temporal', 'PhG': 'Temporal', 'pSTS': 'Temporal', 'SPL': 'Parietal', 'IPL': 'Parietal', 'Pcun': 'Parietal', 'PoG': 'Parietal', 'INS': 'Insular', 'CG': 'Limbic', 'MVOcC': 'Occipital', 'LOcC': 'Occipital', 'Amyg': 'Subcortical', 'Hipp': 'Subcortical', 'BG': 'Subcortical', 'Tha': 'Subcortical'}
# subregion_abrev2lobe["Unknown"] = "Unknown"
# Harvard_Oxford_to_BNA = {'Angular Gyrus': "IPL", 'Background': "Unknown", 'Central Opercular Cortex': "IPL", 'Cingulate Gyrus, anterior division': "CG", \
# 'Cingulate Gyrus, posterior division': "CG", 'Frontal Operculum Cortex': "IFG", 'Frontal Orbital Cortex': "OrG", 'Frontal Pole': "OrG", \
# "Heschl's Gyrus (includes H1 and H2)": "STG", 'Inferior Frontal Gyrus, pars opercularis': "IFG", 'Inferior Frontal Gyrus, pars triangularis': "IFG", \
# 'Inferior Temporal Gyrus, temporooccipital part': "ITG", 'Insular Cortex': "INS", 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': "SFG", \
# 'Lateral Occipital Cortex, inferior division': "LOcC", 'Lateral Occipital Cortex, superior division': "LOcC", 'Lingual Gyrus': "MVOcC", \
# 'Middle Frontal Gyrus': "MFG", 'Middle Temporal Gyrus, anterior division': "MTG", 'Middle Temporal Gyrus, posterior division': "MTG",
# 'Occipital Fusiform Gyrus': "FuG", 'Paracingulate Gyrus': "CG", 'Parahippocampal Gyrus, anterior division': "PhG",
# 'Parietal Operculum Cortex': "STG", 'Planum Polare': "STG", 'Planum Temporale': "STG", 'Postcentral Gyrus': "PoG", 'Precentral Gyrus': "PrG", \
# 'Precuneous Cortex': "Pcun", 'Subcallosal Cortex': "Unknown", 'Superior Frontal Gyrus': "SFG", 'Superior Parietal Lobule': "SPL",
# 'Superior Temporal Gyrus, anterior division': "STG", 'Superior Temporal Gyrus, posterior division': "STG", \
# 'Supramarginal Gyrus, anterior division': "IPL", 'Supramarginal Gyrus, posterior division': "IPL",
# 'Temporal Occipital Fusiform Cortex': "FuG", 'Temporal Pole': "MTG"}

def load_atlas(name='cort-maxprob-thr25-2mm', symmetric_split=True):
	# atlas = image.load_img("/Users/tdesbordes/BN_Atlas_246_1mm.nii") 
	# atlas = nib.load("/Users/tdesbordes/BNA_FC_4D.nii") 
	atlas = datasets.fetch_atlas_harvard_oxford(name, symmetric_split=symmetric_split)
	labels = atlas['labels']
	maps = nilearn.image.load_img(atlas['maps'])
	return maps, labels

# atlas_detailed = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1)
# labels = [roi.decode("utf-8") for roi in atlas_detailed['labels']]
# atlas_maps = nilearn.image.load_img(atlas_detailed['maps'])

def load_BNA(patient, root_path='/neurospin/unicog/protocols/Marseille_Jab/Code/jab-intra', strip_RL=True): 
    ''' Get coordinate to electrode name dict in the braintome atlas
    '''
    path2braintome = root_path + f"/JABBER_SEEGContactsPerRoi/{patient}_SEEGcontactsInBrainnetome.csv"
    regions_df = pd.read_csv(path2braintome)
    if strip_RL: 
        # remove the trailing characters if they have the hemispheric side included. ONLY REMOVE THIS, NOT LETTERS IN THE MIDDLE OF THE REGION NAME. 
        regions_df.BN_Label = regions_df.BN_Label.apply(lambda x: x[0:-2] if x[-2::] in ["_R", "_L"] else x)
    regions_df.BN_Label = regions_df.BN_Label.str.replace('Not found', 'Unknown')
    regions_df.Contact = regions_df.Contact.str.replace("'", 'p') # apostrophe replace "p"in BNA file........................
    # regions_df.X.apply(lambda x: np.around(x, 6))
    coord2elec = {f"{x}_{y}_{z}": ch for x, y, z, ch in zip(regions_df.X, regions_df.Y, regions_df.Z, regions_df.Contact)}
    # replace some region names that are bad in the csv
    regions_df.BN_Label.str = regions_df.BN_Label.replace("TE1.0/TE1.2", "TE1.0 and TE1.2")
    elec2BNA = dict(zip(regions_df.Contact, regions_df.BN_Label.str)) # .str is important because else the modifications are not taken into account
    elec2BNA["Unknown"] = "Unknown"
    return coord2elec, elec2BNA
    

def voxel_masker(coords, labels, atlas_map, **kwargs_masker):
	"""Argument:
	 - coords: (x, y, z)
	 - labels: name of the atlas' regions
	 - atlas_map: Nifit image (3D)
	 Returns:
	 - masker for given coordinates
	 - roi
	 - voxel coordinates
	 """
	# if type(atlas_map)==str:
	# 	atlas_map = nib.load(atlas_map)
	affine = atlas_map.affine[:3, :3]
	translation = atlas_map.affine[:3, 3]
	data_coords = np.matmul(np.linalg.inv(affine), np.array(coords) - translation)
	a,b,c = np.apply_along_axis(lambda x: np.round(x, 0), 0, data_coords).astype(int)
	# a,b,c = np.apply_along_axis(lambda x: np.round(x, 0), 0, coords).astype(int) # no affine - do not do that
	index = atlas_map.get_data()[a, b, c]
	roi = labels[index]
	new_data = np.zeros(atlas_map.get_data().shape)
	new_data[a,b,c] = 1
	masker = NiftiMasker(nilearn.image.new_img_like(atlas_map, new_data))
	masker.set_params(**kwargs_masker)
	masker.fit()
	return masker, roi, [a, b, c]

def coordstr2np(coords):
	return np.array([float(coord) for coord in coords.split('_')])

# regions = ["SFG", "MFG", "IFG", "OrG", "STG", "MTG", "ITG", "FuG", "PhG", "pSTS", "SPL", "IPL", "Pcun", "PoG", "PrG", "PCL", "MVOcC", "LOcC", "Amyg", "Hipp", "BG", "Tha", "Unknown", "INS", "CG"]
# region_cmap = plt.cm.get_cmap('terrain', len(regions)+1) # +1 because the last color is white and we don't want it
# region_colors = {region: region_cmap(i) for i, region in enumerate(regions)}

rois = []
rois_bna = []
all_rois_inter_bna_prime = []
all_rois_inter_harvox_prime = []
all_colors = []
all_coords = []
all_ch_names = []
for sub in ['JA_P01', 'JA_P02', 'JA_P03', 'JA_P04', 'JA_P05', 'JA_P06', 'JA_P07', 'JA_P08', 'JA_P09', 'JA_P10', 'JA_P11']:
	coord2elec, elec2BNA = load_BNA(sub, root_path='/Users/tdesbordes/Documents/jab-intra', strip_RL=strip_RL)
	atlas_map, labels = load_atlas(symmetric_split=not strip_RL)
	for coord, el in coord2elec.items(): 
		masker, roi, transf_coord = voxel_masker(coordstr2np(coord), labels, atlas_map)
		rois.append(roi)
		area = elec2BNA[el]
		if not strip_RL: # remove the last characters because there is no left/right distinction in the areas dict
			if area[-2::] in ["_R", "_L"]: # BNA area
				G_or_D = area[-2::]
				area_stripped = area[0:-2]
			else: # backup with harvox area (happens with Unknown BNA region)
				if roi.split(' ')[0] in ["Left"]:
					G_or_D = "_L"
				elif roi.split(' ')[0] in ["Right"]:
					G_or_D = "_R"
				else:
					G_or_D = ''
				area_stripped = area
		else:
			G_or_D = ''
			area_stripped = area
		print(area, area_stripped)
		if "TE1.0/TE1.2" in area_stripped: area_stripped = "TE1.0 and TE1.2"
		roi_bna = BNA2subregion_abrev[area_stripped]
		rois_bna.append(f"{roi_bna}{G_or_D}")
		print(f"{el} - {coord} - {roi} -- {BNA2subregion[area_stripped]}")
		# nilearn.plotting.plot_glass_brain(masker.mask_img, display_mode='lzry')
		# nilearn.plotting.show()
		# all_colors.append(region_colors[BNA2subregion_abrev[elec2BNA[el]]])
		# all_coords.append(transf_coord) # do not store this, its always outside the brain
		all_coords.append(coordstr2np(coord))
		all_ch_names.append(f"{sub}_{el}")
		
		if not strip_RL:
			if roi != "Background": 
				roi_strip = ' '.join(roi.split(' ')[1::])
			else:
				roi_strip = roi
		else:
			roi_strip = roi
		roi_harvox = Harvard_Oxford_to_BNA[roi_strip]
		if roi_harvox != "Unknown": roi_harvox += G_or_D
		if roi_bna != "Unknown": roi_bna += G_or_D

		# store regions, giving primauty to Harvard Oxford atlas
		if roi != "Background":
			all_rois_inter_harvox_prime.append(roi_harvox)
		elif roi_bna != "Unknown":
			all_rois_inter_harvox_prime.append(roi_bna)
		else:
			all_rois_inter_harvox_prime.append("Unknown")

		# store regions, giving primauty to BNA atlas
		if roi_bna != "Unknown":
			all_rois_inter_bna_prime.append(roi_bna)
		elif roi != "Background":
			all_rois_inter_bna_prime.append(roi_harvox)
		else:
			all_rois_inter_bna_prime.append("Unknown")

		if "Unknown_R" in roi_harvox or"Unknown_R" in roi_bna:
			set_trace()


##################  PLOTS OF ELECS COLORED BY REGION, DEPENDING ON WHETHER WE TRUST MORE THE BNA OR HARVARD-OXFORD ATLASES.
regions = sorted(np.unique(all_rois_inter_bna_prime))
region_cmap = plt.cm.get_cmap('terrain', len(regions)+1) # +1 because the last color is white and we don't want it
region_colors = {region: region_cmap(i) for i, region in enumerate(regions)}
all_colors = []
for region in all_rois_inter_bna_prime:
	all_colors.append(region_colors[region])
sizes = np.ones(len(all_colors)) * 10
view = plotting.view_markers(all_coords, all_colors, marker_size=sizes)
view.save_as_html(f'BNA_Lobes_bna_prime{strip_str}.html')

regions = sorted(np.unique(all_rois_inter_bna_prime))
region_colors = {region: region_cmap(i) for i, region in enumerate(regions)}
all_colors = []
for region in all_rois_inter_harvox_prime:
	all_colors.append(region_colors[region])
sizes = np.ones(len(all_colors)) * 10
view = plotting.view_markers(all_coords, all_colors, marker_size=sizes)
view.save_as_html(f'BNA_Lobes_HarvOxf_prime{strip_str}.html')

################## COMPARE THE 2 LISTS OF REGIONS
print("Nb of elec with the same label in both lists: ", (np.array(all_rois_inter_bna_prime) == np.array(all_rois_inter_harvox_prime)).sum())
print("Nb of elec with the same label in both lists, not taking into account unknown: ", np.logical_and(np.array(all_rois_inter_bna_prime) == np.array(all_rois_inter_harvox_prime), (np.array(all_rois_inter_harvox_prime) != "Unknown")).sum())
print("Nb of unkown elec before: ", (np.array(rois_bna) == "Unknown").sum())
print("Nb of unkown elec : ", (np.array(all_rois_inter_bna_prime) == "Unknown").sum())

################## SAVE DICT OF ELECS2AREA 
bna_prime_dict = dict(zip(all_ch_names, all_rois_inter_bna_prime))
pickle.dump(bna_prime_dict, open(f"JABBER_SEEGContactsPerRoi/Regions_joint_BNA_prime{strip_str}.p", "wb"))

harvox_prime_dict = dict(zip(all_ch_names, all_rois_inter_harvox_prime))
pickle.dump(harvox_prime_dict, open(f"JABBER_SEEGContactsPerRoi/Regions_joint_HarvOx_prime{strip_str}.p", "wb"))

# also save electrodes coordinates for easier use
coords_dict = dict(zip(all_ch_names, all_coords))
pickle.dump(coords_dict, open(f"JABBER_SEEGContactsPerRoi/all_elecs_coords{strip_str}.p", "wb"))


rois = np.array(rois)
rois_bna = np.array(rois_bna)
print(f"did not found {(rois=='Background').sum()} out of the total {len(rois)}")
print(f"with BNA: did not found {(rois_bna=='Unknown').sum()} out of the total {len(rois_bna)}")
set_trace()

# np.intersect1d(np.where(rois=="Background")[0], np.where(rois_bna=="Unknown")[0])

# with affine, P_01: 
# did not found 142 out of the total 259
# with BNA: did not found 75 out of the total 259
# without affine:
# did not found 222 out of the total 259
# with BNA: did not found 75 out of the total 259

# with affine, P_03:
# did not found 107 out of the total 211
# with BNA: did not found 69 out of the total 211
# without: 
# did not found 179 out of the total 211
# with BNA: did not found 69 out of the total 211




# Nb of elec with the same label in both lists:  1690
# Nb of elec with the same label in both lists, not taking into account unknown:  1351
# Nb of unkown elec before:  339
# Nb of unkown elec :  339
# did not found 1176 out of the total 2243
# with BNA: did not found 339 out of the total 2243
# --Return--

