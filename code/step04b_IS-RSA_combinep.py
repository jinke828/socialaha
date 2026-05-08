"""
step04b_IS-RSA_fisher_combined.py

Fisher's combined test across the two IS-RSA analyses from step04:
  - Analysis A ("after"):  brain ISC (all 10 runs) ~ post-movie impression similarity
  - Analysis B ("before"): brain ISC (runs 2–10)   ~ pre-movie impression similarity

For each of the 116 ROIs, combine the two bootstrap p-values using Fisher's method:
    X² = -2 * (ln p_after + ln p_before)
    X² ~ χ²(4)  under H0 (2 independent tests × 2 df each)

Direction filter: Fisher's test is only applied to ROIs where rvals_after and
rvals_before share the same sign (both positive or both negative). Opposite-direction
ROIs are excluded (fisher_pvals set to 1). FDR correction (BH, q < 0.05) is applied
across all 116 ROIs.

Inputs:
  results/IS-RSA/after_bootstrapping_bycharacter.mat   — fields: pvals, rvals
  results/IS-RSA/before_bootstrapping_bycharacter.mat  — fields: pvals, rvals

Outputs:
  results/IS-RSA/step19_fisher_combined.mat            — combined stats for all 116 ROIs
  results/figures/step19_fisher_combined_raw.png       — brain map: combined -log10(p)
  results/figures/step19_fisher_combined_fdr.png       — brain map: FDR-significant ROIs
  results/figures/step19_fisher_rvals_mean.png         — brain map: mean r (after + before)
  results/figures/step19_pvalue_comparison.png         — scatter: individual vs combined p
"""

import numpy as np
import scipy.io
import scipy.stats
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from nilearn.image import load_img, new_img_like
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_stat_map

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
os.chdir('/Users/jinke/Desktop/socialaha_github/')
directory   = './socialaha-collab'
RESULTS_DIR = './results'
ISRSA_DIR   = os.path.join(RESULTS_DIR, 'IS-RSA')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')

os.makedirs(ISRSA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

nroi_cor, nroi_sub = 100, 16
nroi = nroi_cor + nroi_sub   # 116

# Bootstrap sample size used in step04 (needed for p-value floor)
N_BOOT = 10000

# ---------------------------------------------------------------------------
# Atlas mask (same build logic as step04)
# ---------------------------------------------------------------------------
def niftimask(nroi_cor, nroi_sub, directory):
    cortical = (directory + '/template/tpl-MNI152NLin2009cAsym/'
                'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
                + str(nroi_cor) + 'Parcels17Networks_dseg.nii.gz')
    subcortical = (directory +
                   '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/'
                   'Tian_Subcortex_S1_3T_2009cAsym.nii.gz')
    mask_cor = load_img(cortical).dataobj[:]
    mask_sub = load_img(subcortical).dataobj[:].copy()
    mask_sub[mask_sub > 0] += nroi_cor
    overlap_id = np.where(np.multiply(mask_cor, mask_sub) > 0)
    mask = mask_cor + mask_sub
    mask[overlap_id[0], overlap_id[1], overlap_id[2]] = 0
    return mask


print('Building ROI mask...')
mask = niftimask(nroi_cor, nroi_sub, directory)

ref_img_path = (directory + '/template/tpl-MNI152NLin2009cAsym/'
                'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
                '100Parcels17Networks_dseg.nii.gz')
ref_img = load_img(ref_img_path)

# ---------------------------------------------------------------------------
# Surface plotting helpers (reused from step04)
# ---------------------------------------------------------------------------
def make_brain_map(values, sig_rois_idx=None):
    brain = np.full(mask.shape, np.nan)
    for i in range(brain.shape[0]):
        for j in range(brain.shape[1]):
            for k in range(brain.shape[2]):
                idx = int(mask[i, j, k]) - 1
                if idx < 0:
                    continue
                if sig_rois_idx is not None:
                    if idx in sig_rois_idx:
                        brain[i, j, k] = values[idx]
                else:
                    brain[i, j, k] = values[idx]
    return new_img_like(ref_img, brain)


def plot_surface_maps(surf_l, surf_r, fsaverage, outpath,
                      cmap='cold_hot', vmin=-0.10, vmax=0.10, title=''):
    sns.set_context('paper')
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, subplot_kw={'projection': '3d'},
                                          figsize=(10, 8))
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False)
    plot_surf_stat_map(fsaverage.infl_left,  surf_l, hemi='left',
                       view='lateral', bg_map=fsaverage.sulc_left,  axes=a, **kw)
    plot_surf_stat_map(fsaverage.infl_right, surf_r, hemi='right',
                       view='lateral', bg_map=fsaverage.sulc_right, axes=b, **kw)
    plot_surf_stat_map(fsaverage.infl_left,  surf_l, hemi='left',
                       view='medial',  bg_map=fsaverage.sulc_left,  axes=c, **kw)
    plot_surf_stat_map(fsaverage.infl_right, surf_r, hemi='right',
                       view='medial',  bg_map=fsaverage.sulc_right, axes=d, **kw)
    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {outpath}')


# ---------------------------------------------------------------------------
# 1. Load step04 outputs
# ---------------------------------------------------------------------------
print('\nLoading step04 .mat files...')
after_mat  = scipy.io.loadmat(os.path.join(ISRSA_DIR, 'after_bootstrapping_bycharacter.mat'))
before_mat = scipy.io.loadmat(os.path.join(ISRSA_DIR, 'before_bootstrapping_bycharacter.mat'))

pvals_after  = np.array(after_mat['pvals']).flatten().astype(float)   # (116,)
pvals_before = np.array(before_mat['pvals']).flatten().astype(float)  # (116,)
rvals_after  = np.array(after_mat['rvals']).flatten().astype(float)   # (116,)
rvals_before = np.array(before_mat['rvals']).flatten().astype(float)  # (116,)

print(f'  After  p-values: min={pvals_after.min():.4f}  max={pvals_after.max():.4f}')
print(f'  Before p-values: min={pvals_before.min():.4f}  max={pvals_before.max():.4f}')

# ---------------------------------------------------------------------------
# 2. Floor p-values at 1/N_BOOT to avoid log(0)
#    Bootstrap p=0 means the statistic was larger than all N_BOOT resamples;
#    the most conservative interpretable value is 1/N_BOOT.
# ---------------------------------------------------------------------------
p_floor = 1.0 / N_BOOT
pvals_after_safe  = np.maximum(pvals_after,  p_floor)
pvals_before_safe = np.maximum(pvals_before, p_floor)

n_floored_after  = np.sum(pvals_after  < p_floor)
n_floored_before = np.sum(pvals_before < p_floor)
if n_floored_after or n_floored_before:
    print(f'  Note: floored {n_floored_after} after-p and {n_floored_before} '
          f'before-p values at {p_floor} (1/N_BOOT)')

# ---------------------------------------------------------------------------
# 3. Direction filter: only combine p-values for ROIs where r-values agree in sign
# ---------------------------------------------------------------------------
same_direction = np.sign(rvals_after) == np.sign(rvals_before)   # (116,) bool
n_same = same_direction.sum()
n_opposite = (~same_direction).sum()
print(f'\n  Direction filter: {n_same} ROIs same direction, {n_opposite} ROIs opposite direction')
print(f'  ROIs excluded (opposite direction): {list(np.where(~same_direction)[0] + 1)}  (1-indexed)')

# ---------------------------------------------------------------------------
# 4. Fisher's combined test: χ²(4) = -2 * (ln p_after + ln p_before)
#    Only applied to same-direction ROIs; opposite-direction ROIs get p=1.
# ---------------------------------------------------------------------------
print('\nApplying Fisher\'s combined test (same-direction ROIs only)...')
fisher_stat = np.zeros(nroi)
fisher_pvals = np.ones(nroi)

fisher_stat[same_direction] = -2.0 * (
    np.log(pvals_after_safe[same_direction]) + np.log(pvals_before_safe[same_direction])
)
fisher_pvals[same_direction] = scipy.stats.chi2.sf(fisher_stat[same_direction], df=4)

# ---------------------------------------------------------------------------
# 5. FDR correction (Benjamini–Hochberg) across same-direction ROIs only
# ---------------------------------------------------------------------------
rejected_all, pvals_fdr, _, _ = multipletests(
    fisher_pvals, alpha=0.05, method='fdr_bh'
)

sig_rois = np.where(rejected_all)[0]
print(f'\n  Fisher combined p (same-direction ROIs): '
      f'min={fisher_pvals[same_direction].min():.2e}  '
      f'max={fisher_pvals[same_direction].max():.4f}')
print(f'  Significant ROIs after FDR correction (n={len(sig_rois)}):')
print(f'  {list(sig_rois + 1)}  (1-indexed)')

# ---------------------------------------------------------------------------
# 6. Mean r-value across the two analyses (Fisher z averaged, then back to r)
# ---------------------------------------------------------------------------
def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

mean_r = conv_z2r((conv_r2z(rvals_after) + conv_r2z(rvals_before)) / 2.0)

# ---------------------------------------------------------------------------
# 7. Save combined results
# ---------------------------------------------------------------------------
scipy.io.savemat(
    os.path.join(ISRSA_DIR, 'step19_fisher_combined.mat'),
    {
        'fisher_stat':      fisher_stat,
        'fisher_pvals':     fisher_pvals,
        'pvals_fdr':        pvals_fdr,
        'sig_rois':         sig_rois,           # 0-indexed
        'sig_rois_1idx':    sig_rois + 1,       # 1-indexed (ROI label)
        'same_direction':   same_direction,     # bool mask
        'pvals_after':      pvals_after,
        'pvals_before':     pvals_before,
        'rvals_after':      rvals_after,
        'rvals_before':     rvals_before,
        'mean_r':           mean_r,
    }
)
print(f'\nSaved: {os.path.join(ISRSA_DIR, "step19_fisher_combined.mat")}')

# ---------------------------------------------------------------------------
# 8. Print ROI-level summary
# ---------------------------------------------------------------------------
print('\n=== ROI-level summary (significant ROIs) ===')
print(f'  {"ROI":>5}  {"p_after":>10}  {"p_before":>10}  {"fisher_χ²":>12}  '
      f'{"combined_p":>12}  {"FDR_p":>10}  {"mean_r":>8}')
for roi in sig_rois:
    print(f'  {roi+1:>5}  {pvals_after[roi]:>10.4f}  {pvals_before[roi]:>10.4f}  '
          f'{fisher_stat[roi]:>12.3f}  {fisher_pvals[roi]:>12.2e}  '
          f'{pvals_fdr[roi]:>10.4f}  {mean_r[roi]:>8.4f}')

# ---------------------------------------------------------------------------
# 9. Brain surface maps
# ---------------------------------------------------------------------------
print('\nGenerating brain surface maps...')
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

# 8a. -log10(Fisher combined p) — raw (all 116 ROIs)
neg_log_p = -np.log10(fisher_pvals)
neg_log_p_map = make_brain_map(neg_log_p)
surf_l = surface.vol_to_surf(neg_log_p_map, fsaverage.pial_left)
surf_r = surface.vol_to_surf(neg_log_p_map, fsaverage.pial_right)
thresh = -np.log10(0.05)
plot_surface_maps(
    surf_l, surf_r, fsaverage,
    os.path.join(FIG_DIR, 'step19_fisher_combined_raw.png'),
    cmap='YlOrRd', vmin=0, vmax=max(neg_log_p.max(), thresh + 0.5),
    title='IS-RSA Fisher combined: −log₁₀(p) [raw]')

# 8b. -log10(Fisher combined p) — only FDR-significant ROIs
neg_log_p_sig_map = make_brain_map(neg_log_p, sig_rois_idx=sig_rois)
surf_l_sig = surface.vol_to_surf(neg_log_p_sig_map, fsaverage.pial_left)
surf_r_sig = surface.vol_to_surf(neg_log_p_sig_map, fsaverage.pial_right)
plot_surface_maps(
    surf_l_sig, surf_r_sig, fsaverage,
    os.path.join(FIG_DIR, 'step19_fisher_combined_fdr.png'),
    cmap='YlOrRd', vmin=0, vmax=max(neg_log_p.max(), thresh + 0.5),
    title='IS-RSA Fisher combined: −log₁₀(p) [FDR q < 0.05, all 116 ROIs]')

# 8c. Mean r-value at FDR-significant ROIs — fixed scale matching step04 (vmin/vmax = ±0.10)
mean_r_map = make_brain_map(mean_r, sig_rois_idx=sig_rois)
surf_l_r = surface.vol_to_surf(mean_r_map, fsaverage.pial_left)
surf_r_r = surface.vol_to_surf(mean_r_map, fsaverage.pial_right)
plot_surface_maps(
    surf_l_r, surf_r_r, fsaverage,
    os.path.join(FIG_DIR, 'step19_fisher_rvals_mean.png'),
    cmap='cold_hot', vmin=-0.10, vmax=0.10,
    title='IS-RSA Fisher combined: mean r (after + before) [FDR sig. ROIs]')

# ---------------------------------------------------------------------------
# 10. Diagnostic scatter: individual p-values vs Fisher combined p
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (x_p, x_label) in zip(axes, [
    (pvals_after,  'p (after)'),
    (pvals_before, 'p (before)'),
]):
    colors = ['#e31a1c' if roi in sig_rois else '#a6cee3' for roi in range(nroi)]
    ax.scatter(-np.log10(np.maximum(x_p, p_floor)),
               -np.log10(fisher_pvals),
               c=colors, s=25, alpha=0.8, edgecolors='none')
    ax.axhline(-np.log10(0.05), color='grey', linewidth=0.8, linestyle='--', label='p=0.05')
    ax.axvline(-np.log10(0.05), color='grey', linewidth=0.8, linestyle=':')
    # Annotate a few top ROIs
    top_rois = np.argsort(fisher_pvals)[:5]
    for roi in top_rois:
        ax.annotate(f'ROI {roi+1}',
                    xy=(-np.log10(max(x_p[roi], p_floor)), -np.log10(fisher_pvals[roi])),
                    xytext=(3, 3), textcoords='offset points', fontsize=6.5)
    ax.set_xlabel(f'−log₁₀({x_label})', fontsize=11)
    ax.set_ylabel('−log₁₀(Fisher combined p)', fontsize=11)
    ax.set_title(f'{x_label}  vs  Fisher combined p', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

# Add legend annotation
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e31a1c',
           markersize=8, label='FDR significant'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#a6cee3',
           markersize=8, label='not significant'),
]
axes[0].legend(handles=legend_elements, fontsize=9)

fig.suptitle(
    'IS-RSA: Fisher\'s combined test (after + before)\n'
    'Red = FDR q < 0.05 (all 116 ROIs)',
    fontsize=12, fontweight='bold')
fig.tight_layout()
out = os.path.join(FIG_DIR, 'step19_pvalue_comparison.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved: {out}')

# ---------------------------------------------------------------------------
# 11. Bar chart: individual vs combined -log10(p) for sig ROIs
# ---------------------------------------------------------------------------
if len(sig_rois) > 0:
    fig, ax = plt.subplots(figsize=(max(6, len(sig_rois) * 0.7), 5))
    x = np.arange(len(sig_rois))
    w = 0.25
    ax.bar(x - w,   -np.log10(np.maximum(pvals_after[sig_rois],  p_floor)),
           w, label='After',    color='#4393c3', alpha=0.85)
    ax.bar(x,       -np.log10(np.maximum(pvals_before[sig_rois], p_floor)),
           w, label='Before',   color='#92c5de', alpha=0.85)
    ax.bar(x + w,   -np.log10(fisher_pvals[sig_rois]),
           w, label='Combined', color='#d6604d', alpha=0.85)
    ax.axhline(-np.log10(0.05), color='black', linewidth=0.8, linestyle='--',
               label='p = 0.05')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ROI {r+1}' for r in sig_rois], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('−log₁₀(p)', fontsize=11)
    ax.set_title('IS-RSA Fisher combined: FDR-significant ROIs\n'
                 'Blue = after / Light blue = before / Red = Fisher combined',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'step19_sig_rois_barchart.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')

print('\nDone.')
