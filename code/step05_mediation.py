"""
step05_mediation_fixed.py

Plots the fixed mediation results (step05b/c *_fixed outputs) with multiple-comparison
correction restricted to the ROIs that were FDR-significant in the fixed step04 IS-RSA
(after-movie and/or before-movie analysis), rather than all 116 ROIs.

ROI set (0-indexed) = union of step04 FDR-significant ROIs:
  after:  [2, 9, 13, 60, 64, 98, 99]
  before: [50, 64, 88, 99]
  union:  [2, 9, 13, 50, 60, 64, 88, 98, 99]  (9 ROIs)
The set is read from the step04 _fixed .mat files, not hard-coded.

Benjamini-Hochberg FDR (q < 0.05) is applied to the step05b ACME p-values (1,000 bootstrap
sims) of these 9 ROIs only.

Inputs:
  results/IS-RSA/after_bootstrapping_bycharacter_fixed.mat   (step04_IS-RSA_fixed.py)
  results/IS-RSA/before_bootstrapping_bycharacter_fixed.mat  (step04_IS-RSA_fixed.py)
  results/IS-RSA/mediation_results_all_rois_fixed.csv        (step05b_mediation_fixed.R)
  results/IS-RSA/mediation_results_roi98_fixed.csv           (step05c_mediation_rSTS_fixed.R)
  results/IS-RSA/ACME_null_roi98_fixed.csv                   (step05c_mediation_rSTS_fixed.R)

Outputs:
  results/IS-RSA/step05_mediation_step04rois_fixed.csv       - ACME, p, FDR q for the 9 ROIs
  results/figures/step05_mediation_ACME_raw_fixed.png        - ACME, all 116 ROIs
  results/figures/step05_mediation_ACME_step04rois_thresholded_fixed.png
                                                             - ACME, FDR-sig ROIs within the 9-ROI set
  results/figures/step05_mediation_ACME_step04rois_uncorrected_fixed.png
                                                             - ACME, step04 ROIs with uncorrected p < 0.05
  results/figures/step05_mediation_rSTS_ACME_distribution_fixed.png
                                                             - rSTS (ROI 98) bootstrap ACME distribution
"""

import os
import warnings

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets, surface
from nilearn.image import load_img, new_img_like
from nilearn.plotting import plot_surf_stat_map
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

BASE_DIR = '/Users/jinke/Desktop/socialaha_github'
os.chdir(BASE_DIR)
directory = './socialaha-collab'
ISRSA_DIR = './results/IS-RSA'
FIG_DIR = './results/figures'

nroi_cor, nroi_sub = 100, 16
nroi = nroi_cor + nroi_sub
alpha = 0.05
RSTS_ROI = 98          # 0-indexed (brain_sim_98)
VLIM = 0.006           # colour range, same as mediation_ACME_fixed.png


# =============================================================================
# Build combined ROI atlas mask (same as step04_IS-RSA_fixed.py)
# =============================================================================

def niftimask(nroi_cor, nroi_sub, directory):
    cortical = (directory + '/template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
                + str(nroi_cor) + 'Parcels17Networks_dseg.nii.gz')
    subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S1_3T_2009cAsym.nii.gz'

    mask_cor = load_img(cortical).dataobj[:]
    mask_sub = np.array(load_img(subcortical).dataobj[:])
    mask_sub[mask_sub > 0] += nroi_cor

    overlap = np.multiply(mask_cor, mask_sub) > 0
    mask = mask_cor + mask_sub
    mask[overlap] = 0
    return mask


ref_img_path = (directory + '/template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018'
                '_desc-100Parcels17Networks_dseg.nii.gz')


def make_brain_map(values, sig_rois_idx=None):
    """Map per-ROI values onto the volume. If sig_rois_idx given, set all other ROIs to NaN.

    Background voxels (mask == 0) get values[-1], matching the original notebook/step04 behaviour.
    """
    values = np.asarray(values, float)
    if sig_rois_idx is not None:
        keep = np.full(nroi, np.nan)
        keep[list(sig_rois_idx)] = values[list(sig_rois_idx)]
        values = keep
    idx = mask.astype(int) - 1
    return new_img_like(load_img(ref_img_path), values[idx])


def plot_surface_maps(brain_map, outpath, cmap='cold_hot', vmin=-VLIM, vmax=VLIM):
    surface_left = surface.vol_to_surf(brain_map, fsaverage.pial_left)
    surface_right = surface.vol_to_surf(brain_map, fsaverage.pial_right)

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
    plt.close()
    print(f'Saved: {outpath}')


print('Building ROI mask...')
mask = niftimask(nroi_cor, nroi_sub, directory)
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')


# =============================================================================
# 1. ROI set: union of step04 (fixed) FDR-significant ROIs
# =============================================================================

def step04_sig_rois(name):
    mat = scipy.io.loadmat(os.path.join(ISRSA_DIR, f'{name}_bootstrapping_bycharacter_fixed.mat'))
    return set(np.where(np.asarray(mat["corrected_p"]).ravel() < alpha)[0].tolist())


sig_after = step04_sig_rois('after')
sig_before = step04_sig_rois('before')
roi_set = np.array(sorted(sig_after | sig_before))
print(f'step04 after  FDR-sig ROIs (0-idx): {sorted(sig_after)}')
print(f'step04 before FDR-sig ROIs (0-idx): {sorted(sig_before)}')
print(f'Union ({len(roi_set)} ROIs):          {roi_set.tolist()}')


# =============================================================================
# 2. FDR correction within the step04 ROI set
# =============================================================================

results = pd.read_csv(os.path.join(ISRSA_DIR, 'mediation_results_all_rois_fixed.csv'))
acme = results['ACME'].to_numpy()
pvals = results['ACME_p'].to_numpy()

rejected, qvals, _, _ = multipletests(pvals[roi_set], alpha=alpha, method='fdr_bh')
sig_rois = roi_set[rejected]

summary = pd.DataFrame({
    'roi': roi_set,
    'in_step04_after': [r in sig_after for r in roi_set],
    'in_step04_before': [r in sig_before for r in roi_set],
    'ACME': acme[roi_set],
    'ACME_ci_low': results['ACME_ci_low'].to_numpy()[roi_set],
    'ACME_ci_high': results['ACME_ci_high'].to_numpy()[roi_set],
    'ACME_p': pvals[roi_set],
    'ACME_q_fdr_step04rois': qvals,
    'sig': rejected,
})
print(f'\nMediation ACME, FDR-corrected within {len(roi_set)} step04 ROIs:')
print(summary.to_string(index=False))
print(f'Significant ROIs (0-idx): {sig_rois.tolist()}')

out_csv = os.path.join(ISRSA_DIR, 'step05_mediation_step04rois_fixed.csv')
summary.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')


# =============================================================================
# 3. Brain maps: raw ACME (all 116 ROIs) and thresholded (FDR within step04 ROIs)
# =============================================================================

plot_surface_maps(make_brain_map(acme),
                  os.path.join(FIG_DIR, 'step05_mediation_ACME_raw_fixed.png'))
plot_surface_maps(make_brain_map(acme, sig_rois),
                  os.path.join(FIG_DIR, 'step05_mediation_ACME_step04rois_thresholded_fixed.png'))

# uncorrected: step04 ROIs with ACME p < 0.05 before FDR
sig_rois_uncorr = roi_set[pvals[roi_set] < alpha]
print(f'Uncorrected p < {alpha} within step04 ROIs (0-idx): {sig_rois_uncorr.tolist()}')
plot_surface_maps(make_brain_map(acme, sig_rois_uncorr),
                  os.path.join(FIG_DIR, 'step05_mediation_ACME_step04rois_uncorrected_fixed.png'))


# =============================================================================
# 4. rSTS (ROI 98): bootstrap distribution of the ACME (step05c, 10,000 sims)
# =============================================================================

# Note: step05c saves med_out$d0.sims as 'ACME_null', but these are bootstrap draws of the
# ACME (its sampling distribution), not a null distribution. p = how often draws cross 0.
boot = pd.read_csv(os.path.join(ISRSA_DIR, 'ACME_null_roi98_fixed.csv'))['ACME_null'].to_numpy()
rsts = pd.read_csv(os.path.join(ISRSA_DIR, 'mediation_results_roi98_fixed.csv')).iloc[0]

sns.set_context('paper')
plt.figure(figsize=(4, 3), dpi=600)
plt.hist(boot, bins=30, density=True, alpha=0.7)
plt.xlabel('ACME (bootstrap)')
plt.ylabel('Density')
plt.title(f"rSTS (ROI {RSTS_ROI}), p = {rsts['ACME_p']:.4f}", fontsize=8)
plt.tight_layout()
out = os.path.join(FIG_DIR, 'step05_mediation_rSTS_ACME_distribution_fixed.png')
plt.savefig(out, dpi=1000)
plt.close()
print(f'Saved: {out}')
