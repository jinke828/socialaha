"""
step07_neural-pattern-shift--character_only.py

Analysis: Test whether neural pattern shifts (moment-to-moment dissimilarity,
1 − Pearson r between consecutive TRs) are elevated around character-related
aha moments relative to a permutation null of non-aha timepoints.

Three conditions:
  1. Character aha:     annotations with character_all >= 2 (multi-rater consensus)
  2. Non-character aha: annotations with character_all == 0
  3. All aha:           all annotations

Steps:
  1. For each condition, compute per-subject per-ROI pattern shifts averaged
     over a [-5, +3] TR window centered on aha moments → shape (116, 33, 9)
  2. Generate null distribution: for each subject/run/ROI draw 10,000 random
     non-aha timepoints and record their [-5, +3] windows
  3. Statistical testing: two-tailed p-value vs. null at each ROI × time point;
     FDR-correct within time point across ROIs; then FDR-correct across 9 TPs
  4. Visualize: time course of significant ROI counts, brain surface maps

Inputs:
  ./data/brain/pattern_shift/1TR_nearbytp.npy
  ./data/beh/annotations/ahaannot_all.xlsx

Outputs:
  ./results/neural_updates/null_nonearaha_1TR_nb_character.npy  (if recomputed)
  ./results/figures/pattern-shift_<tp>_raw-map_character.png
  ./results/figures/pattern-shift_<tp>_thresholded-map_character.png
  ./results/figures/sig_roi_timecourse.png
  ./results/figures/roi98_pattern_shift.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no popup windows
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import seaborn as sns
from nilearn.image import load_img, new_img_like
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_stat_map
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = '/Users/jinke/Desktop/socialaha_github'
COLLAB_DIR = os.path.join(BASE_DIR, 'socialaha-collab')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(os.path.join(RESULTS_DIR, 'neural_updates'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)

# ---------------------------------------------------------------------------
# Participant setup
# ---------------------------------------------------------------------------
flist = {
    1: ['sub-1001','sub-1005','sub-1008','sub-1011','sub-1014','sub-1017',
        'sub-1020','sub-1023','sub-1026','sub-1029','sub-1033','sub-1039'],
    2: ['sub-2006','sub-2009','sub-2012','sub-2015','sub-2018','sub-2021',
        'sub-2024','sub-2027','sub-2034','sub-2038','sub-2040'],
    3: ['sub-3004','sub-3007','sub-3013','sub-3016','sub-3019','sub-3022',
        'sub-3025','sub-3031','sub-3037','sub-3041'],
}
nroi_cor, nroi_sub = 100, 16
nroi  = nroi_cor + nroi_sub   # 116
nsubj = sum(len(v) for v in flist.values())  # 33


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def fdr_correct(pvals, alpha=0.05):
    """FDR-BH correction; returns corrected p-values as list of floats."""
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return [float(round(p, 4)) for p in pvals_corrected]


def twotail_p(real, null):
    """Two-tailed permutation p-value: P(|null| >= |real|)."""
    return np.sum(np.abs(null) >= np.abs(real)) / (1 + len(null))


def build_nifti_mask(nroi_cor, nroi_sub, directory):
    """Vectorized: combine cortical (Schaefer) and subcortical (Tian) atlas masks."""
    cortical_path = os.path.join(
        directory, 'template', 'tpl-MNI152NLin2009cAsym',
        f'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
        f'{nroi_cor}Parcels17Networks_dseg.nii.gz')
    if nroi_sub == 16:
        subcortical_path = os.path.join(
            directory, 'template', 'Tian2020MSA_v1.1_3T_Subcortex-Only',
            'Tian_Subcortex_S1_3T_2009cAsym.nii.gz')
    mask_cor = load_img(cortical_path).get_fdata()
    mask_sub = load_img(subcortical_path).get_fdata().copy()

    # Offset subcortical labels so they don't overlap with cortical
    mask_sub[mask_sub > 0] += nroi_cor

    # Zero out any voxels that appear in both masks
    overlap = (mask_cor > 0) & (mask_sub > 0)
    mask = mask_cor + mask_sub
    mask[overlap] = 0
    return mask


def _build_brain_volume(mask, values, sig_rois=None):
    """
    Vectorized: map ROI values onto brain volume using atlas mask.

    mask     : 3-D int array, values 0 (background) to nroi (1-indexed labels)
    values   : 1-D array of length nroi (0-indexed)
    sig_rois : if supplied (0-indexed ROI indices), only those ROIs receive
               values; all others → NaN.  If None, all ROIs are mapped and
               background voxels → 0.
    """
    max_label = int(mask.max())
    if sig_rois is not None:
        lookup = np.full(max_label + 1, np.nan)
        for idx in sig_rois:
            if 1 <= idx + 1 <= max_label:
                lookup[idx + 1] = values[idx]
    else:
        lookup = np.zeros(max_label + 1)
        for idx in range(min(len(values), max_label)):
            lookup[idx + 1] = values[idx]
    return lookup[mask.astype(int)]


# ---------------------------------------------------------------------------
# Pre-build annotation lookups (avoids repeated pandas filtering)
# ---------------------------------------------------------------------------
def build_df_lookup(df, flist):
    """
    Pre-compute aha TR lists indexed by (groupid, sub_idx, run, scene).
    Called once per condition; eliminates O(116×33×8) pandas queries in the
    inner loops.
    """
    lookup = {}
    for groupid in range(1, 4):
        for sub_idx, sub_str in enumerate(flist[groupid]):
            sub_num = int(sub_str[4:])
            df_sub = df[df['subject'] == sub_num]
            for run in range(8):
                run_idx = run + 2 if run < 5 else run + 3
                df_run = df_sub[df_sub['run'] == run_idx]
                for scene in range(1, 6):
                    trs = df_run[df_run['scene'] == scene]['TR (run)'].tolist()
                    lookup[(groupid, sub_idx, run, scene)] = trs
    return lookup


def build_null_lookup(df, flist):
    """
    Pre-compute all aha TRs per (groupid, sub_idx, run) for null exclusion.
    """
    lookup = {}
    for groupid in range(1, 4):
        for sub_idx, sub_str in enumerate(flist[groupid]):
            sub_num = int(sub_str[4:])
            df_sub = df[df['subject'] == sub_num]
            for run in range(8):
                run_idx = run + 2 if run < 5 else run + 3
                trs = df_sub[df_sub['run'] == run_idx]['TR (run)'].tolist()
                lookup[(groupid, sub_idx, run)] = trs
    return lookup


# ---------------------------------------------------------------------------
# Core analysis: pattern shifts around aha events
# ---------------------------------------------------------------------------
def _shift_windows_for_run(this_run, scene_trs_list):
    """
    Compute pattern-shift windows for one run.

    scene_trs_list : list of 5 TR lists, one per scene.
    Returns (5, 9) array; NaN where a scene had no aha events.
    """
    dissims = []
    for scene_trs in scene_trs_list:
        if len(scene_trs) == 0:
            dissims.append(np.full(9, np.nan))
        else:
            windows = []
            for tp in scene_trs:
                if tp - 5 < 0:
                    pad = 5 - tp
                    arr = np.concatenate([np.full(pad, np.nan), this_run[0:tp + 4]])
                elif tp + 4 > len(this_run):
                    pad = 4 + tp - len(this_run)
                    arr = np.concatenate([this_run[tp - 5:], np.full(pad, np.nan)])
                else:
                    arr = this_run[tp - 5: tp + 4]
                windows.append(arr)
            dissims.append(np.nanmean(np.array(windows), axis=0))
    return np.array(dissims)  # (5, 9)


def compute_avg_subs(df_lookup, pattern_shift, flist, nroi):
    """
    For all ROIs and subjects, compute mean pattern shift in the aha window.
    ROIs are processed in parallel using threads.
    Returns (nroi, nsubj, 9).
    """
    def _process_roi(roi):
        this_roi = []
        for groupid in range(1, 4):
            for sub_idx in range(len(flist[groupid])):
                sub_runs = []
                for run in range(8):
                    this_run = pattern_shift[groupid, roi, sub_idx][run]
                    scene_trs = [df_lookup[(groupid, sub_idx, run, s)]
                                 for s in range(1, 6)]
                    dissims = _shift_windows_for_run(this_run, scene_trs)
                    sub_runs.append(np.nanmean(dissims, axis=0))
                this_roi.append(np.nanmean(np.array(sub_runs), axis=0))
        return this_roi  # (nsubj, 9)

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_process_roi)(roi) for roi in range(1, nroi + 1)
    )
    return np.array(results)  # (nroi, nsubj, 9)


# ---------------------------------------------------------------------------
# Null distribution: non-aha timepoints
# ---------------------------------------------------------------------------
def _generate_null_for_run(this_run, aha_trs, iterations=10000):
    """
    Vectorized: draw `iterations` random non-aha timepoints and return
    their [-5, +3] windows.  Returns (iterations, 9).
    """
    excluded = set()
    for t in aha_trs:
        excluded.update(range(max(0, t - 5), min(len(this_run), t + 4)))
    # Only keep timepoints with fully valid windows
    available = np.array([t for t in range(5, len(this_run) - 3)
                          if t not in excluded])
    if len(available) == 0:
        return np.full((iterations, 9), np.nan)

    chosen  = np.random.choice(available, size=iterations, replace=True)
    offsets = np.arange(-5, 4)                          # (9,)
    indices = chosen[:, None] + offsets[None, :]        # (iterations, 9)
    return this_run[indices]


def compute_null_distribution(null_lookup, pattern_shift, flist, nroi,
                               iterations=10000):
    """
    Generate null distribution for all ROIs / subjects / runs.
    ROIs are processed in parallel.
    Returns (nroi, nsubj, 8, iterations, 9).
    """
    def _process_roi(roi):
        subs = []
        for groupid in range(1, 4):
            for sub_idx in range(len(flist[groupid])):
                runs = []
                for run in range(8):
                    this_run = pattern_shift[groupid, roi, sub_idx][run]
                    aha_trs  = null_lookup[(groupid, sub_idx, run)]
                    runs.append(_generate_null_for_run(this_run, aha_trs, iterations))
                subs.append(np.array(runs))          # (8, iterations, 9)
        return np.array(subs)                        # (nsubj, 8, iterations, 9)

    print('Generating null distribution...')
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_process_roi)(roi) for roi in range(1, nroi + 1)
    )
    return np.array(results)  # (nroi, nsubj, 8, iterations, 9)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------
def test_rois(avg_subs, null_nonearaha):
    """
    For each ROI × time point, compute one-tailed permutation p-value and
    z-score vs. null; apply within-TP FDR correction.

    null_nonearaha : (116, 33, iterations, 9)  [averaged over runs already]

    Returns:
      all_rois_p : list of 116 lists of 9 FDR-corrected p-values
      dissim_roi : (116, 9) z-score array
    """
    all_rois_p = []
    dissim_roi = []
    for roi in range(nroi):
        pvals = []
        zvals = []
        actual   = np.nanmean(avg_subs[roi], axis=0)     # (9,)
        null_avg = np.nanmean(null_nonearaha[roi], axis=0)  # (iterations, 9)
        for tp in range(9):
            null_tp = null_avg[:, tp]
            pvals.append(float(twotail_p(actual[tp], null_tp)))
            zvals.append((actual[tp] - np.nanmean(null_tp)) / np.nanstd(null_tp))
        all_rois_p.append(fdr_correct(np.array(pvals)))
        dissim_roi.append(np.array(zvals))
    return all_rois_p, np.array(dissim_roi)


# ---------------------------------------------------------------------------
# Brain surface plotting
# ---------------------------------------------------------------------------
def _surface_arrays(brain, ref_img, fsaverage):
    """Project a volume to fsaverage left/right cortical surfaces."""
    brain_map = new_img_like(ref_img, brain)
    surf_l = surface.vol_to_surf(brain_map, fsaverage.pial_left)
    surf_r = surface.vol_to_surf(brain_map, fsaverage.pial_right)
    return surf_l, surf_r


def _draw_surface_views(axes, surf_l, surf_r, fsaverage, vmin=-8, vmax=8):
    """Draw lateral and medial views on a 2x2 axes block."""
    (a, b), (c, d) = axes
    kw = dict(cmap='cold_hot', vmin=vmin, vmax=vmax, colorbar=False)
    plot_surf_stat_map(fsaverage.infl_left,  surf_l, hemi='left',
                       view='lateral', bg_map=fsaverage.sulc_left,  axes=a, **kw)
    plot_surf_stat_map(fsaverage.infl_right, surf_r, hemi='right',
                       view='lateral', bg_map=fsaverage.sulc_right, axes=b, **kw)
    plot_surf_stat_map(fsaverage.infl_left,  surf_l, hemi='left',
                       view='medial',  bg_map=fsaverage.sulc_left,  axes=c, **kw)
    plot_surf_stat_map(fsaverage.infl_right, surf_r, hemi='right',
                       view='medial',  bg_map=fsaverage.sulc_right, axes=d, **kw)


def _plot_surface(brain, ref_img, fsaverage, tp_label, title, save_path):
    """Shared surface-plotting routine for raw and thresholded maps."""
    surf_l, surf_r = _surface_arrays(brain, ref_img, fsaverage)

    sns.set_context('paper')
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    _draw_surface_views(axes, surf_l, surf_r, fsaverage)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_brain_grid(brain_list, ref_img, fsaverage, tp_labels, title, save_path):
    """Save a single 4x9 composite figure spanning TR -5 to +3."""
    sns.set_context('paper')
    fig, axes = plt.subplots(
        4, len(tp_labels),
        figsize=(2.0 * len(tp_labels), 7.6),
        subplot_kw={'projection': '3d'})

    row_labels = ['L lateral', 'R lateral', 'L medial', 'R medial']
    for col, (brain, tp_label) in enumerate(zip(brain_list, tp_labels)):
        surf_l, surf_r = _surface_arrays(brain, ref_img, fsaverage)
        _draw_surface_views(axes[:, col].reshape(2, 2), surf_l, surf_r, fsaverage)
        axes[0, col].set_title(f'TR {tp_label:+d}', fontsize=10, pad=10)

    for row, label in enumerate(row_labels):
        axes[row, 0].text2D(-0.12, 0.5, label, transform=axes[row, 0].transAxes,
                            rotation=90, va='center', ha='right', fontsize=9)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.02,
                        wspace=0.02, hspace=0.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_brain_raw(values, mask, ref_img, fsaverage, tp_label, save_path=None):
    """Plot z-scores for all ROIs on fsaverage5 surface (unthresholded)."""
    brain = _build_brain_volume(mask, values)
    _plot_surface(brain, ref_img, fsaverage, tp_label,
                  f'Neural pattern shift — unthresholded, TR {tp_label}', save_path)


def plot_brain_thresholded(values, sig_rois, mask, ref_img, fsaverage,
                            tp_label, save_path=None):
    """Plot z-scores only for significant ROIs on fsaverage5 surface.
    Saves an empty brain map when no ROIs survive correction."""
    brain = _build_brain_volume(mask, values, sig_rois=sig_rois)
    _plot_surface(brain, ref_img, fsaverage, tp_label,
                  f'Neural pattern shift — thresholded, TR {tp_label}', save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)

    # ---- Load pattern shift data ----
    pattern_shift = np.load(
        os.path.join(BASE_DIR, 'data/brain/pattern_shift/1TR_nearbytp.npy'),
        allow_pickle=True).item()

    # ---- Load & annotate aha events ----
    annot_path = os.path.join(BASE_DIR, 'data/beh/annotations/ahaannot_all.xlsx')
    df_full = pd.read_excel(annot_path)
    categories = ['character', 'relationship', 'retrieval', 'current',
                  'inference', 'temporal', 'oops', 'causal']
    for cat in categories:
        df_full[cat + '_all'] = df_full[[cat + '_rater1', cat + '_rater2',
                                          cat + '_rater3']].sum(axis=1)

    df_char    = df_full[df_full['character_all'] >= 2].copy()
    df_nonchar = df_full[df_full['character_all'] == 0].copy()
    df_all     = df_full.copy()

    # ---- Build brain mask (vectorized) ----
    mask = build_nifti_mask(nroi_cor, nroi_sub, COLLAB_DIR)
    ref_img = load_img(os.path.join(
        COLLAB_DIR, 'template', 'tpl-MNI152NLin2009cAsym',
        'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
        '100Parcels17Networks_dseg.nii.gz'))
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    # ---- Pre-build annotation lookups (eliminates repeated pandas filtering) ----
    print('Building annotation lookups...')
    lookup_char    = build_df_lookup(df_char,    flist)
    lookup_nonchar = build_df_lookup(df_nonchar, flist)
    lookup_all     = build_df_lookup(df_all,     flist)

    # ---- Compute per-subject pattern shifts for each condition ----
    print('Computing pattern shifts: character aha...')
    avg_subs = compute_avg_subs(lookup_char, pattern_shift, flist, nroi)

    print('Computing pattern shifts: non-character aha...')
    avg_subs_nonchar = compute_avg_subs(lookup_nonchar, pattern_shift, flist, nroi)

    print('Computing pattern shifts: all aha...')
    avg_subs_all = compute_avg_subs(lookup_all, pattern_shift, flist, nroi)
    # shapes: (116, 33, 9)

    # ---- Load or compute null distribution ----
    null_path = os.path.join(RESULTS_DIR, 'neural_updates',
                              'null_nonearaha_1TR_nb_character.npy')
    if os.path.exists(null_path):
        print(f'Loading pre-computed null from {null_path}')
        null_nonearaha_raw = np.load(null_path, allow_pickle=True)
        # Average over runs (axis 2) → (116, 33, 10000, 9)
        if null_nonearaha_raw.ndim == 5:
            null_nonearaha = np.nanmean(null_nonearaha_raw, axis=2)
        else:
            null_nonearaha = null_nonearaha_raw
    else:
        print('Pre-computed null not found; generating from scratch...')
        null_lookup = build_null_lookup(df_char, flist)
        null_nonearaha_raw = compute_null_distribution(
            null_lookup, pattern_shift, flist, nroi, iterations=10000)
        np.save(null_path, null_nonearaha_raw, allow_pickle=True)
        null_nonearaha = np.nanmean(null_nonearaha_raw, axis=2)
    # null_nonearaha shape: (116, 33, 10000, 9)

    # ---- Statistical testing: per-ROI × per-TP ----
    print('Running statistical tests...')
    all_rois_p, dissim_roi = test_rois(avg_subs, null_nonearaha)

    x = list(range(-5, 4))
    sig_counts = []
    for tp in range(9):
        p_vals_tp = np.array(all_rois_p)[:, tp]
        corrected = fdr_correct(p_vals_tp)
        sig       = np.where(np.array(corrected) < 0.05)[0]
        sig_counts.append(len(sig))
        print(f'TR {tp - 5:+d}: {len(sig)} significant ROIs  {list(sig)}')

    # ---- Plot: time course of significant ROI counts ----
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(x, sig_counts, marker='o', linewidth=2.5, color='#1A4E8A')
    plt.xlabel('Time (TRs relative to aha)')
    plt.ylabel('Number of significant ROIs')
    plt.title('Significant ROI count over time')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'sig_roi_timecourse.png'), dpi=300)
    plt.close()

    # ---- Brain surface maps: thresholded and unthresholded at each TP ----
    raw_brains = []
    thresholded_brains = []
    for tp in range(9):
        tp_label    = tp - 5
        p_vals_tp   = np.array(all_rois_p)[:, tp]
        corrected   = fdr_correct(p_vals_tp)
        sig_rois_tp = np.where(np.array(corrected) < 0.05)[0]
        z_vals      = dissim_roi[:, tp]
        raw_brains.append(_build_brain_volume(mask, z_vals))
        thresholded_brains.append(_build_brain_volume(mask, z_vals, sig_rois=sig_rois_tp))

        plot_brain_raw(
            z_vals, mask, ref_img, fsaverage, tp_label,
            save_path=os.path.join(RESULTS_DIR, 'figures',
                                   f'pattern-shift_{tp_label}_raw-map_character.png'))

        # Always save thresholded map (empty brain when no ROIs survive)
        plot_brain_thresholded(
            z_vals, sig_rois_tp, mask, ref_img, fsaverage, tp_label,
            save_path=os.path.join(RESULTS_DIR, 'figures',
                                   f'pattern-shift_{tp_label}_thresholded-map_character.png'))

    plot_brain_grid(
        raw_brains, ref_img, fsaverage, x,
        title='Neural pattern shift — unthresholded',
        save_path=os.path.join(RESULTS_DIR, 'figures',
                               'pattern-shift_allTRs_raw-map_character.png'))
    plot_brain_grid(
        thresholded_brains, ref_img, fsaverage, x,
        title='Neural pattern shift — thresholded',
        save_path=os.path.join(RESULTS_DIR, 'figures',
                               'pattern-shift_allTRs_thresholded-map_character.png'))

    # ---- ROI 98 detail plot (STS) ----
    roi_idx = 98  # 0-indexed
    null_roi98   = null_nonearaha[roi_idx]                  # (33, 10000, 9)
    null_dist_98 = np.nanmean(null_roi98, axis=0).T         # (9, 10000)

    avg_actual  = np.nanmean(avg_subs[roi_idx],         axis=0)
    avg_nonchar = np.nanmean(avg_subs_nonchar[roi_idx], axis=0)
    avg_allaha  = np.nanmean(avg_subs_all[roi_idx],     axis=0)

    lower98 = np.percentile(null_dist_98, 2.5,  axis=1)
    upper98 = np.percentile(null_dist_98, 97.5, axis=1)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.plot(x, avg_actual,  color='red',   linewidth=2.5, label='character aha')
    ax.plot(x, avg_nonchar, color='green', linewidth=2.5, label='non-character aha')
    ax.plot(x, avg_allaha,  color='black', linewidth=2.5, label='all aha')
    ax.fill_between(x, lower98, upper98, color='lightgray', alpha=0.3,
                    label='Null 95% CI')
    ax.set_xlabel('Time from aha button press (TR)')
    ax.set_ylabel('Neural pattern shift')
    fig.savefig(os.path.join(RESULTS_DIR, 'figures', 'roi98_pattern_shift.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print('\nDone. Figures saved to', os.path.join(RESULTS_DIR, 'figures'))


if __name__ == '__main__':
    main()
