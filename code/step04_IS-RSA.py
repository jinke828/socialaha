"""
step04_IS-RSA-avg_ISC.py

Analysis: For each ROI, test whether inter-subject neural synchrony (ISC) during
movie-watching predicts between-subject impression similarity (IS-RSA).

Both use 10,000-iteration bootstrapping + FDR correction.
Outputs: .mat files and brain surface map figures.
"""

import numpy as np
from nilearn.image import load_img, new_img_like
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import scipy.io
import pickle
import os
import warnings
import seaborn as sns
import nibabel as nib
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_stat_map
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# =============================================================================
# Utility functions
# =============================================================================

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))


def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def nanspearmanr(tc1, tc2):
    nanid1 = np.where(np.isnan(tc1))
    nanid2 = np.where(np.isnan(tc2))
    nanid = np.union1d(nanid1, nanid2)
    tc1 = np.delete(tc1, nanid)
    tc2 = np.delete(tc2, nanid)
    rval, pval = spearmanr(tc1, tc2)
    return rval, pval


def nanpearsonr(tc1, tc2):
    nanid1 = np.where(np.isnan(tc1))
    nanid2 = np.where(np.isnan(tc2))
    nanid = np.union1d(nanid1, nanid2)
    tc1 = np.delete(tc1, nanid)
    tc2 = np.delete(tc2, nanid)
    rval, pval = pearsonr(tc1, tc2)
    return rval, pval


def onetail_p(real, null):
    return (np.sum(null >= real)) / (1 + len(null))


def twotail_p(real, null):
    return (1 + np.sum(abs(null) >= abs(real))) / (1 + len(null))


# =============================================================================
# Setup paths and parameters
# =============================================================================

os.chdir('/Users/jinke/Desktop/socialaha_github/')
directory = './socialaha-collab'

nroi_cor, nroi_sub = 100, 16
nroi = nroi_cor + nroi_sub
n_iterations = 10000

# =============================================================================
# Load event index table
# =============================================================================

event_idxs = pd.read_csv(directory + '/socialaha-fMRI/socialaha_groupscene.csv')


def rearrange(group_id, run_id, brain):
    current_indices = event_idxs[event_idxs['run'] == run_id]['g' + str(group_id) + '.char'].tolist()
    sorted_order = np.argsort(current_indices)
    rearranged_brain = brain[sorted_order]
    return rearranged_brain


def rearrange_new(group_id, run_id, brain):
    """Rearrange neural patterns by character, averaging where a character appears twice."""
    current_indices = event_idxs[event_idxs['run'] == run_id]['g' + str(group_id) + '.char'].tolist()
    sorted_order = np.argsort(current_indices)
    rearranged_brain = brain[sorted_order]

    if 5 in current_indices:
        rearranged_brain4 = []
        for i in range(4):
            if i in [1, 3]:
                this_run = np.nanmean(rearranged_brain[[i, 4], :], axis=0)
            else:
                this_run = rearranged_brain[i]
            rearranged_brain4.append(this_run)
    else:
        duplicates = [x for x in set(current_indices) if current_indices.count(x) > 1]
        if len(duplicates) == 0:
            return rearranged_brain
        else:
            rearranged_brain4 = []
            for i in range(1, 4 + 1):
                if duplicates[0] == i:
                    this_run = np.nanmean(rearranged_brain[[i, i - 1], :], axis=0)
                else:
                    this_run = rearranged_brain[i]
                rearranged_brain4.append(this_run)

    return np.array(rearranged_brain4)


def char_order(group_id, run_id):
    current_indices = event_idxs[event_idxs['run'] == run_id]['g' + str(group_id) + '.char'].tolist()
    current_indices.sort()
    return current_indices


# =============================================================================
# Build combined ROI atlas mask
# =============================================================================

def niftimask(nroi_cor, nroi_sub, directory):
    cortical = (directory + '/template/tpl-MNI152NLin2009cAsym/'
                'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
                + str(nroi_cor) + 'Parcels17Networks_dseg.nii.gz')
    if nroi_sub == 16:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S1_3T_2009cAsym.nii.gz'
    elif nroi_sub == 32:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S2_3T_2009cAsym.nii.gz'
    elif nroi_sub == 50:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S3_3T_2009cAsym.nii.gz'
    elif nroi_sub == 54:
        subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz'

    mask_cor = load_img(cortical).dataobj[:]
    mask_sub = load_img(subcortical).dataobj[:]

    for i1 in range(mask_sub.shape[0]):
        for i2 in range(mask_sub.shape[1]):
            for i3 in range(mask_sub.shape[2]):
                if mask_sub[i1, i2, i3] > 0:
                    mask_sub[i1, i2, i3] = mask_sub[i1, i2, i3] + nroi_cor

    overlap_id = np.where(np.multiply(mask_cor, mask_sub) > 0)
    mask = mask_cor + mask_sub
    mask[overlap_id[0], overlap_id[1], overlap_id[2]] = 0
    return mask


print("Building ROI mask...")
mask = niftimask(nroi_cor, nroi_sub, directory)

# =============================================================================
# Load data
# =============================================================================

print("Loading data...")
thoughts_similarity = np.load('./data/beh/similarity/impressions_byrun_bychar.npy',
                               allow_pickle=True).item()  # {(groupid, run): nchar × nsubjpairs}
neuralISC = np.load('./data/brain/similarity/neuralISC_byevent.npy',
                    allow_pickle=True).item()  # {(roi, groupid, run): array}

# =============================================================================
# Prepare similarity matrices
# =============================================================================

# brain_sim[roi][group]: shape (nscene*nchar, nsubjpairs) = (40, nsubjpairs)
print("Preparing brain similarity matrices...")
brain_sim = []
for roi in range(1, nroi + 1):
    this_roi = []
    for groupid in range(1, 3 + 1):
        this_group = []
        for run in range(10):
            this_brain = neuralISC[roi, groupid, run]
            if run == 6:  # 7th run: no character structure — replicate mean 4 times
                this_run = np.nanmean(this_brain, axis=0)
                this_run = np.vstack([this_run, this_run, this_run, this_run])
            else:
                this_run = rearrange_new(groupid, run + 1, this_brain)
            this_group.append(this_run)
        this_group = np.concatenate(this_group)
        this_roi.append(np.array(this_group))
    brain_sim.append(this_roi)

# thought_sim[group]: shape (40, nsubjpairs)
print("Preparing thought similarity matrices...")
thought_sim = []
for groupid in range(1, 3 + 1):
    this_group = []
    for run in range(10):
        this_group.append(thoughts_similarity[groupid, run + 1])
    this_group = np.concatenate(this_group)
    thought_sim.append(np.array(this_group))

# =============================================================================
# Bootstrap function
# =============================================================================

def bootstrapping(data):
    """One-tailed bootstrap test: p = P(mean <= 0)."""
    boot_means = []
    np.random.seed(42)
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.nanmean(sample))
    boot_means = np.array(boot_means)
    p_value = np.nanmean(boot_means <= 0)
    return p_value


# =============================================================================
# Surface plotting helper
# =============================================================================

ref_img_path = (directory + '/template/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018'
                '_desc-100Parcels17Networks_dseg.nii.gz')


def make_brain_map(mean_r_values, sig_rois_idx=None):
    """Map mean_r values onto volumetric brain image. If sig_rois_idx given, mask non-sig to NaN."""
    ref_img = load_img(ref_img_path)
    brain = np.zeros(mask.shape)
    for i in range(brain.shape[0]):
        for j in range(brain.shape[1]):
            for k in range(brain.shape[2]):
                idx = int(mask[i, j, k]) - 1
                if sig_rois_idx is not None:
                    if idx in sig_rois_idx:
                        brain[i, j, k] = mean_r_values[idx]
                    else:
                        brain[i, j, k] = np.nan
                else:
                    brain[i, j, k] = mean_r_values[idx]
    return new_img_like(ref_img, brain)


def plot_surface_maps(surface_left, surface_right, fsaverage, outpath,
                      cmap='cold_hot', vmin=-0.10, vmax=0.10):
    sns.set_context('paper')
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    plot_surf_stat_map(fsaverage.infl_left, surface_left, hemi='left',
                       view='lateral', bg_map=fsaverage.sulc_left,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=a)
    plot_surf_stat_map(fsaverage.infl_right, surface_right, hemi='right',
                       view='lateral', bg_map=fsaverage.sulc_right,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=b)
    plot_surf_stat_map(fsaverage.infl_left, surface_left, hemi='left',
                       view='medial', bg_map=fsaverage.sulc_left,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=c)
    plot_surf_stat_map(fsaverage.infl_right, surface_right, hemi='right',
                       view='medial', bg_map=fsaverage.sulc_right,
                       cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, axes=d)
    plt.savefig(outpath, dpi=1000)
    plt.tight_layout()
    plt.close()


# =============================================================================
# ANALYSIS 1: After-movie impressions
# Brain ISC (all 10 runs) vs. post-movie impression similarity (all 10 runs)
# =============================================================================

print("\n--- ANALYSIS 1: After-movie IS-RSA ---")

actual_r_after = []
for roi in range(nroi):
    this_roi = brain_sim[roi]
    rvals = []
    for groupid in range(3):
        for spair in range(thought_sim[groupid].shape[1]):
            tmp_r = nanspearmanr(
                conv_z2r(this_roi[groupid][:, spair]),
                thought_sim[groupid][:, spair]
            )[0]
            rvals.append(tmp_r)
    actual_r_after.append(np.array(rvals))

print("Bootstrapping (after)...")
pvals_after = []
for roi in range(nroi):
    pval = float(bootstrapping(actual_r_after[roi]))
    pvals_after.append(pval)

# FDR correction
alpha = 0.05
rejected_after, pvals_corr_after, _, _ = multipletests(pvals_after, alpha=alpha, method='fdr_bh')
pvals_corr_after = [round(i, 4) for i in pvals_corr_after]
sig_rois_after = np.where(np.array(pvals_corr_after) < alpha)
print('After-movie: significant ROIs (FDR-corrected p < 0.05):')
print(sig_rois_after[0])

# Mean r (Fisher z averaged, converted back)
mean_r_after = [conv_z2r(np.nanmean(conv_r2z(actual_r_after[roi]))) for roi in range(nroi)]

# Surface projection and plotting
print("Projecting after-movie results to surface...")
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

brain_map_after = make_brain_map(mean_r_after)
surf_left_after = surface.vol_to_surf(brain_map_after, fsaverage.pial_left)
surf_right_after = surface.vol_to_surf(brain_map_after, fsaverage.pial_right)
plot_surface_maps(surf_left_after, surf_right_after, fsaverage,
                  './results/figures/IS-RSA_byevent_bootstrapping_scenes_bycharacter.png')

brain_map_after_thr = make_brain_map(mean_r_after, sig_rois_idx=sig_rois_after[0])
surf_left_after_thr = surface.vol_to_surf(brain_map_after_thr, fsaverage.pial_left)
surf_right_after_thr = surface.vol_to_surf(brain_map_after_thr, fsaverage.pial_right)
plot_surface_maps(surf_left_after_thr, surf_right_after_thr, fsaverage,
                  './results/figures/IS-RSA_byevent_thresholded_bootstrapping_scenes_bycharacter.png')

# Save .mat
os.makedirs('./results/IS-RSA', exist_ok=True)
scipy.io.savemat('./results/IS-RSA/after_bootstrapping_bycharacter.mat',
                 {'rvals': mean_r_after, 'pvals': pvals_after,
                  'corrected_p': pvals_corr_after, 'sig_rois': sig_rois_after})
print("Saved: after_bootstrapping_bycharacter.mat")

# =============================================================================
# ANALYSIS 2: Before-movie impressions
# Brain ISC from runs 2-10 vs. impression similarity from runs 1-9
# (i.e., 1-run lag: impression from run N is the before-impression for run N+1 movie)
# brain[4:, :] skips run 1 (4 chars); thought[:-4, :] excludes run 10 (4 chars)
# =============================================================================

print("\n--- ANALYSIS 2: Before-movie IS-RSA ---")

actual_r_before = []
for roi in range(nroi):
    this_roi = brain_sim[roi]
    rvals = []
    for groupid in range(3):
        for spair in range(thought_sim[groupid].shape[1]):
            tmp_r = nanspearmanr(
                conv_z2r(this_roi[groupid][4:, spair]),
                thought_sim[groupid][:-4, spair]
            )[0]
            rvals.append(tmp_r)
    actual_r_before.append(np.array(rvals))

print("Bootstrapping (before)...")
pvals_before = []
for roi in range(nroi):
    pval = float(bootstrapping(actual_r_before[roi]))
    pvals_before.append(pval)

# FDR correction
rejected_before, pvals_corr_before, _, _ = multipletests(pvals_before, alpha=alpha, method='fdr_bh')
pvals_corr_before = [round(i, 4) for i in pvals_corr_before]
sig_rois_before = np.where(np.array(pvals_corr_before) < alpha)
print('Before-movie: significant ROIs (FDR-corrected p < 0.05):')
print(sig_rois_before[0])

# Mean r
mean_r_before = [conv_z2r(np.nanmean(conv_r2z(actual_r_before[roi]))) for roi in range(nroi)]

# Surface projection and plotting
print("Projecting before-movie results to surface...")
brain_map_before = make_brain_map(mean_r_before)
surf_left_before = surface.vol_to_surf(brain_map_before, fsaverage.pial_left)
surf_right_before = surface.vol_to_surf(brain_map_before, fsaverage.pial_right)
plot_surface_maps(surf_left_before, surf_right_before, fsaverage,
                  './results/figures/IS-RSA_byevent_before_bootstrapping_scenes_bycharacter.png')

brain_map_before_thr = make_brain_map(mean_r_before, sig_rois_idx=sig_rois_before[0])
surf_left_before_thr = surface.vol_to_surf(brain_map_before_thr, fsaverage.pial_left)
surf_right_before_thr = surface.vol_to_surf(brain_map_before_thr, fsaverage.pial_right)
plot_surface_maps(surf_left_before_thr, surf_right_before_thr, fsaverage,
                  './results/figures/IS-RSA_byevent_before_thresholded_bootstrapping_scenes_bycharacter.png')

# Save .mat
scipy.io.savemat('./results/IS-RSA/before_bootstrapping_bycharacter.mat',
                 {'rvals': mean_r_before, 'pvals': pvals_before,
                  'corrected_p': pvals_corr_before, 'sig_rois': sig_rois_before})
print("Saved: before_bootstrapping_bycharacter.mat")

# =============================================================================
# COMBINED PLOT: Overlap of before vs after significant ROIs
# =============================================================================

print("\n--- COMBINED OVERLAP PLOT ---")

# Categorize ROIs
sig_before_only, sig_after_only, overlap_rois = [], [], []
for roi in range(nroi):
    in_before = roi in sig_rois_before[0]
    in_after = roi in sig_rois_after[0]
    if in_before and in_after:
        overlap_rois.append(roi)
    elif in_before:
        sig_before_only.append(roi)
    elif in_after:
        sig_after_only.append(roi)

ref_img = load_img(ref_img_path)
brain = np.zeros(mask.shape)
for i in range(brain.shape[0]):
    for j in range(brain.shape[1]):
        for k in range(brain.shape[2]):
            idx = int(mask[i, j, k]) - 1
            if idx in sig_before_only:
                brain[i, j, k] = 2   # green
            elif idx in sig_after_only:
                brain[i, j, k] = 0   # purple
            elif idx in overlap_rois:
                brain[i, j, k] = 1   # yellow
            else:
                brain[i, j, k] = np.nan
brain_map_overlap = new_img_like(ref_img, brain)

surf_left_ov = surface.vol_to_surf(brain_map_overlap, fsaverage.pial_left)
surf_right_ov = surface.vol_to_surf(brain_map_overlap, fsaverage.pial_right)

# Custom 3-color colormap: 0=purple, 1=yellow, 2=green
colors = [
    (128 / 255, 0, 128 / 255),   # purple  (sig after only)
    (1, 1, 0),                    # yellow  (overlap)
    (0, 128 / 255, 0)             # green   (sig before only)
]
cmap_overlap = ListedColormap(colors)

sns.set_context('paper')
fig, ((a, b), (c, d)) = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(10, 10))
plot_surf_stat_map(fsaverage.infl_left, surf_left_ov, hemi='left',
                   view='lateral', bg_map=fsaverage.sulc_left,
                   cmap=cmap_overlap, colorbar=False, axes=a, vmin=0, vmax=2)
plot_surf_stat_map(fsaverage.infl_right, surf_right_ov, hemi='right',
                   view='lateral', bg_map=fsaverage.sulc_right,
                   cmap=cmap_overlap, colorbar=False, axes=b, vmin=0, vmax=2)
plot_surf_stat_map(fsaverage.infl_left, surf_left_ov, hemi='left',
                   view='medial', bg_map=fsaverage.sulc_left,
                   cmap=cmap_overlap, colorbar=False, axes=c, vmin=0, vmax=2)
plot_surf_stat_map(fsaverage.infl_right, surf_right_ov, hemi='right',
                   view='medial', bg_map=fsaverage.sulc_right,
                   cmap=cmap_overlap, colorbar=False, axes=d, vmin=0, vmax=2)
plt.tight_layout()
plt.savefig('./results/figures/IS-RSA_byevent_bootstrapping_overlap-new_color_bycharacter.png', dpi=1000)
plt.close()
print("Saved: IS-RSA_byevent_bootstrapping_overlap-new_color_bycharacter.png")

print("\nDone.")
