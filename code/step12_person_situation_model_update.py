"""
step13_person_situation_model_update.py

Tests a neural double dissociation in impression updating:
  - Neural pattern shifts at CHARACTER aha moments predict PERSON-MODEL
    impression updates (how much dispositional/ToM/uncertainty/individual-focus
    language changes between runs)
  - Neural pattern shifts at NON-CHARACTER aha moments predict SITUATION-MODEL
    impression updates (how much social-relationship/group-focus/certainty/
    event-action language changes between runs)

Impression update operationalization (extends step08 framework with step12 categories):
  - person_update:    |Δ person_score| between consecutive runs
                      person_score = mean(dispositional, ToM, uncertainty, individual focus)
  - situation_update: |Δ situation_score| between consecutive runs
                      situation_score = mean(social-relationship, group focus, certainty, event/action)

Both updates are mapped to the same (33, 40) scene structure as step08.

Neural shifts (loaded from step08 caches; not recomputed here):
  - Character aha:     results/neural_updates/step08_shifts_perscene.npy
  - Non-character aha: results/neural_updates/step08_nonchar_shifts_perscene.npy

Pipeline:
  1. Load cached character/non-character pattern shift arrays (116 × 33 × 40 × 9)
  2. Compute person/situation impression updates from impression transcription CSV
  3. For all 4 combinations, compute scene-wise Spearman r (Fisher z-transformed shifts)
  4. Permutation null (n=1000): shuffle subject order within each permutation,
     averaged over 40 scenes. Parallel over ROIs.
  5. FDR correction (single-stage across cortical ROIs per TR)
  6. Double-threshold: intersect with step07 character / nonchar significant ROIs
  7. Dissociation contrasts:
       char_dissoc    = r(char_shifts, person)    - r(char_shifts, situation)
       nonchar_dissoc = r(nonchar_shifts, situation) - r(nonchar_shifts, person)
  8. Brain surface maps (raw, thresholded, double-thresh) for all 4 conditions
  9. Dissociation contrast maps and summary figures

Inputs:
  results/neural_updates/step08_shifts_perscene.npy
  results/neural_updates/step08_nonchar_shifts_perscene.npy
  socialaha-collab/socialaha-fMRI/socialaha_transcribe_rhea_char_updated_clean_jin.csv
  socialaha-collab/socialaha-fMRI/socialaha_groupscene.csv
  results/neural_updates/step07_nonchar_sig_rois.npy  (for double-threshold)

Outputs:
  results/neural_updates/step13_rvals.npy              (4, 116, 9)
  results/neural_updates/step13_dissociation_rvals.npy (2, 116, 9)
  results/neural_updates/step13_null_{key}.npy         (4 cached null arrays)
  results/neural_updates/step13_sig_rois_{key}_step13.npy
  results/neural_updates/step13_sig_rois_{key}_double.npy
  results/figures/neural_update_person_situation/{key}/raw_TR{t}.png
  results/figures/neural_update_person_situation/{key}/thresholded_TR{t}.png
  results/figures/neural_update_person_situation/{key}/doublethresh_TR{t}.png
  results/figures/neural_update_person_situation/dissociation/char_dissoc_TR{t}.png
  results/figures/neural_update_person_situation/dissociation/nonchar_dissoc_TR{t}.png
  results/figures/step13_roi98_dissociation.png
  results/figures/step13_dissociation_barchart.png
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import warnings
import seaborn as sns
from scipy.stats import spearmanr, ttest_rel, ttest_ind
from statsmodels.stats.multitest import multipletests
from nilearn.image import load_img, new_img_like
from nilearn import datasets, surface
from nilearn.plotting import plot_surf_stat_map
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = '/Users/jinke/Desktop/socialaha_github'
COLLAB_DIR  = os.path.join(BASE_DIR, 'socialaha-collab')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures', 'neural_update_person_situation')
NU_DIR      = os.path.join(RESULTS_DIR, 'neural_updates')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(FIG_DIR, 'dissociation'), exist_ok=True)
os.makedirs(NU_DIR, exist_ok=True)

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

VALID_SUBS = set(
    [1001,1005,1008,1011,1014,1017,1020,1023,1026,1029,1033,1039] +
    [2006,2009,2012,2015,2018,2021,2024,2027,2034,2038,2040] +
    [3004,3007,3013,3016,3019,3022,3025,3031,3037,3041]
)

# ---------------------------------------------------------------------------
# Semantic dictionaries
# ---------------------------------------------------------------------------
CATS = {
    'Mental state\n(ToM)': {
        # 'think', 'thought', 'feel', 'felt', 'believe', 'believed', 'want', 'wanted', 'know', 'knew', 
        # 'wonder', 'realize', 'realized', 'understand', 'understood', 'expect', 'hope', 'remember', 
        # 'remembered', 'forget', 'forgot', 'trust', 'doubt', 'wish', 'imagine', 'notice', 'noticed', 
        # 'decide', 'decided', 'love', 'hate', 'enjoy', 'prefer', 'fear', 'intend', 'assume', 'supposed', 
        # 'suspect', 'admiration', 
        'affection', 'agitation', 'alarm', 'alertness', 'amazement', 'ambivalence', 
        'amusement', 'anger', 'annoyance', 'anticipation', 'anxiety', 'apathy', 'appreciation', 'apprehension', 
        'attention', 'awareness', 'awe', 'belief', 'bewilderment', 'bias', 'bitterness', 'boredom', 'calmness', 
        'certainty', 'cheerfulness', 'cognition', 'concern', 'confusion', 'consciousness', 'contemplation', 'contempt', 
        'contentment', 'craziness', 'curiosity', 'decision', 'delight', 'depression', 'derangement', 'desire', 'despair', 
        'disappointment', 'disarray', 'disbelief', 'disgust', 'distress', 'distrust', 'dominance', 'dread', 'dreaminess', 
        'drowsiness', 'drunkenness', 'earnestness', 'ecstasy', 'elation', 'embarrassment', 'emotion', 'empathy', 'enjoyment', 
        'enthusiasm', 'envy', 'exaltation', 'exasperation', 'excitement', 'exhaustion', 'expectation', 'fascination', 
        'fatigue', 'feeling', 'frenzy', 'friendliness', 'frustration', 'fury', 'gloominess', 'guilt', 'hallucination', 
        'happiness', 'horror', 'humiliation', 'humor', 'hunger', 'hypnosis', 'hysteria', 'imagination', 'impatience', 
        'indecisiveness', 'indifference', 'insanity', 'inspiration', 'intention', 'interconnectedness', 'interest', 
        'intrigue', 'irritation', 'jealousy', 'judgment', 'laziness', 'lethargy', 'loneliness', 'lust', 'melancholy', 
        'memory', 'misery', 'mortification', 'nervousness', 'objectivity', 'opinion', 'optimism', 'outrage', 'pain', 
        'panic', 'patience', 'peacefulness', 'pensiveness', 'pity', 'planning', 'playfulness', 'pleasure', 'prejudice', 
        'preoccupation', 'pride', 'rage', 'reason', 'regret', 'relaxation', 'relief', 'remorse', 'resentment', 'sadness', 
        'satisfaction', 'self-consciousness', 'self-control', 'self-pity', 'serenity', 'seriousness', 'shame', 'shock', 
        'skepticism', 'sleepiness', 'sorrow', 'stress', 'stupor', 'subordination', 'surprise', 'suspicion', 'sympathy', 
        'terror', 'thirst', 'tiredness', 'torpor', 'trance', 'transcendence', 'uncertainty', 'uneasiness', 'unhappiness', 
        'vengeance', 'wakefulness', 'warmth', 'weariness', 'woe', 'worry'},
        
    'Dispositional\ntrait': {
        'warmth', 'competence', 'agency', 'experience', 'trustworthiness', 'dominance', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'attractiveness', 'intelligence', 'kind', 'generous', 'selfish', 'ambitious', 'insecure', 'confident', 'shy', 'honest', 'dishonest', 'caring', 'cold', 'warm', 'jealous', 'loyal', 'responsible', 'determined', 'stubborn', 'sensitive', 'emotional', 'successful', 'talented', 'smart', 'proud', 'humble', 'bitter', 'guilty', 'ashamed', 'strong', 'weak', 'depressed', 'anxious', 'upset', 'angry', 'frustrated', 'passionate', 'devoted', 'dedicated', 'driven', 'identity', 'personality', 'character', 'nature'
    },
    'Social\nrelationship': {
        'father', 'dad', 'mother', 'mom', 'son', 'daughter', 'brother',
        'sister', 'sibling', 'siblings', 'wife', 'husband', 'partner',
        'friend', 'family', 'parent', 'parents', 'child', 'children',
        'baby', 'babies', 'couple', 'triplet', 'triplets', 'adopted',
        'biological', 'stepfather', 'stepmother', 'marriage', 'relationship',
        'together', 'married',
    },
    'Causal\nlanguage': {
        'because', 'therefore', 'thus', 'hence', 'explains', 'explanation',
        'reason', 'caused', 'resulted', 'consequently', 'since', 'due',
        'therefore', 'why', 'leads', 'led', 'means', 'implies', 'suggest',
        'suggests', 'indicating', 'indicates',
    },
    'Individual focus\n(he/she)': {
        'he', 'him', 'his', 'she', 'her', "he's", "she's",
    },
    'Group focus\n(they/their)': {
        'they', 'them', 'their', "they're", 'themselves',
    },
    'Event/action\n(situation)': {
        'happened','occurs','occurred','found','discover','discovered',
        'revealed','reveal','learned','learn','met','meet','told','tell',
        'showed','show','died','dead','born','born','left','arrived','arrive',
        'went','came','came','saw','seen','heard','did','done','got','gotten',
        'took','taken','gave','given','brought','sent','lost','won','fell',
        'broke','ran','turned','moved','started','ended','began','finished',
        'decided','chose','chosen','tried','failed','succeeded',
    },
    'Certainty': {
        'definitely','clearly','obviously','certainly','sure','must',
        'absolutely','undoubtedly','confirmed','certain',
    },
    'Uncertainty': {
        'maybe','perhaps','probably','possibly','might','could',
        'seems','seem','seemed','guess','suppose','presumably',
        'apparently','likely','unlikely',
    },
}

# Composite model definitions (from person/situation model framework)
#   Person model    = who someone IS   → dispositional + ToM + uncertainty + individual pronouns
#   Situation model = what happened / who's related → relational + group pronouns + certainty + event/action
# PERSON_CATS    = ['Dispositional\ntrait', 'Mental state\n(ToM)',
#                   'Uncertainty', 'Individual focus\n(he/she)']
# SITUATION_CATS = ['Social\nrelationship', 'Group focus\n(they/their)','Causal\nlanguage',
#                   'Certainty', 'Event/action\n(situation)']

PERSON_CATS    = ['Dispositional\ntrait', 'Mental state\n(ToM)',
                  'Individual focus\n(he/she)']
SITUATION_CATS = ['Social\nrelationship', 'Group focus\n(they/their)','Causal\nlanguage',
                  'Event/action\n(situation)']

# Step07 character aha significant ROIs (hardcoded, identical to step08)
SIG_ROIS_STEP07_CHAR = [
    np.array([], dtype=int),                                                                # TR -5
    np.array([74, 81, 88]),                                                                 # TR -4
    np.array([29, 30, 35, 44, 74, 83, 84, 87]),                                            # TR -3
    np.array([6, 15, 20, 25, 27, 29, 30, 31, 32, 33, 36, 38, 40, 41, 42, 43, 44, 46,
              48, 49, 66, 74, 75, 76, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 92, 93,
              94, 98]),                                                                     # TR -2
    np.array([2, 6, 13, 15, 20, 25, 27, 29, 30, 31, 33, 35, 36, 38, 39, 42, 43, 44, 46,
              49, 65, 66, 74, 75, 79, 80, 81, 82, 83, 84, 85, 87, 88, 91, 92, 98, 99]),   # TR -1
    np.array([2, 6, 13, 15, 19, 20, 23, 25, 28, 29, 30, 31, 32, 33, 35, 36, 38, 40,
              42, 43, 44, 49, 51, 65, 66, 74, 75, 79, 80, 81, 82, 83, 84, 85, 87,
              88, 94, 95, 97, 98, 99]),                                                    # TR  0
    np.array([2, 6, 13, 14, 15, 25, 30, 32, 35, 44, 49, 65, 66, 74, 79, 80, 83, 84,
              85, 87]),                                                                     # TR +1
    np.array([], dtype=int),                                                                # TR +2
    np.array([], dtype=int),                                                                # TR +3
]

# ---------------------------------------------------------------------------
# Helper functions (identical interface to step08)
# ---------------------------------------------------------------------------
def fdr_correct(pvals, alpha=0.05):
    _, corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return corrected.astype(float)


def onetail_p(real, null):
    return np.sum(null >= real) / (1 + len(null))


def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))


def nanspearmanr(tc1, tc2):
    nanid = np.union1d(np.where(np.isnan(tc1)), np.where(np.isnan(tc2)))
    tc1 = np.delete(tc1, nanid)
    tc2 = np.delete(tc2, nanid)
    if len(tc1) < 3:
        return np.nan, np.nan
    rval, pval = spearmanr(tc1, tc2)
    return rval, pval


# ---------------------------------------------------------------------------
# Atlas helpers (identical to step08)
# ---------------------------------------------------------------------------
def build_nifti_mask(nroi_cor, nroi_sub, directory):
    cortical_path = os.path.join(
        directory, 'template', 'tpl-MNI152NLin2009cAsym',
        f'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
        f'{nroi_cor}Parcels17Networks_dseg.nii.gz')
    subcortical_path = os.path.join(
        directory, 'template', 'Tian2020MSA_v1.1_3T_Subcortex-Only',
        'Tian_Subcortex_S1_3T_2009cAsym.nii.gz')
    mask_cor = load_img(cortical_path).get_fdata()
    mask_sub = load_img(subcortical_path).get_fdata().copy()
    mask_sub[mask_sub > 0] += nroi_cor
    overlap = (mask_cor > 0) & (mask_sub > 0)
    mask = mask_cor + mask_sub
    mask[overlap] = 0
    return mask


def build_brain_volume(mask, values, sig_rois=None):
    max_label = int(mask.max())
    if sig_rois is not None:
        lookup = np.full(max_label + 1, np.nan)
        for idx in sig_rois:
            if 0 <= idx < max_label:
                lookup[idx + 1] = values[idx]
    else:
        lookup = np.zeros(max_label + 1)
        for idx in range(min(len(values), max_label)):
            lookup[idx + 1] = values[idx]
    return lookup[mask.astype(int)]


# ---------------------------------------------------------------------------
# Character-order helpers (identical to step08)
# ---------------------------------------------------------------------------
def char_order(group_id, run_id, event_idxs):
    return event_idxs[event_idxs['run'] == run_id][f'g{group_id}.char'].tolist()


# ---------------------------------------------------------------------------
# Text scoring helpers
# ---------------------------------------------------------------------------
def tokenise(text):
    return re.findall(r"[a-z']+", str(text).lower())


def cat_rate(text, word_set):
    tokens = tokenise(text)
    if not tokens:
        return np.nan
    return sum(1 for t in tokens if t in word_set) / len(tokens)


# ---------------------------------------------------------------------------
# Impression update: |Δ semantic score| between consecutive runs
# ---------------------------------------------------------------------------
def compute_semantic_updates(transcribe_path, flist, event_idxs, score_type='person'):
    """
    Returns (nsubj=33, 40) array of |Δ score| between consecutive runs.
    40 = 8 valid runs (2–10, skip 7) × 5 character slots per run, ordered by
    character assignment (same scene structure as step08).

    score_type: 'person' or 'situation'
    """
    df_i = pd.read_csv(transcribe_path)
    df_i = df_i[df_i['subject'].isin(VALID_SUBS)].dropna(subset=['transcribe']).copy()

    for cat, words in CATS.items():
        df_i[cat] = df_i['transcribe'].apply(lambda t: cat_rate(t, words))

    df_i['person_score']    = df_i[PERSON_CATS].mean(axis=1)
    df_i['situation_score'] = df_i[SITUATION_CATS].mean(axis=1)

    score_col = 'person_score' if score_type == 'person' else 'situation_score'

    # Build lookup: (subject_int, run, character_int) → score
    score_lookup = {}
    for _, row in df_i.iterrows():
        key = (int(row['subject']), int(row['run']), int(row['character']))
        score_lookup[key] = row[score_col]

    all_subs = []
    for groupid in range(1, 4):
        for sub_str in flist[groupid]:
            sub_num = int(sub_str[4:])
            this_sub = []
            for run in range(2, 11):
                if run == 7:
                    continue
                # Raw per-character updates (1-indexed characters: 1=Jack, 2=Kate, 3=Randall, 4=Kevin)
                updates = []
                for char in range(1, 5):
                    s_cur  = score_lookup.get((sub_num, run,     char), np.nan)
                    s_prev = score_lookup.get((sub_num, run - 1, char), np.nan)
                    if np.isnan(s_cur) or np.isnan(s_prev):
                        updates.append(np.nan)
                    else:
                        updates.append(abs(s_cur - s_prev))

                # Reorder by character slot assignment (matches step08 scene ordering)
                order = char_order(groupid, run, event_idxs)
                reordered = []
                for idx in order:
                    if idx < 5:
                        reordered.append(updates[int(idx) - 1])
                    else:   # char id 5: kate+kevin combined scene
                        v2, v4 = updates[1], updates[3]
                        combined = (v2 + v4) / 2 if not (np.isnan(v2) or np.isnan(v4)) else np.nan
                        reordered.append(combined)
                this_sub.extend(reordered)
            all_subs.append(np.array(this_sub))

    return np.array(all_subs)   # (33, 40)


# ---------------------------------------------------------------------------
# Brain–behavior correlation
# ---------------------------------------------------------------------------
def compute_rvals(shifts_perscene, imp_updates):
    """
    For each ROI × TR: scene-averaged Spearman r.
    Neural shifts are Fisher z-transformed before correlation.
    Returns (nroi, 9).
    """
    nroi_n, nsubj_n, nscene, ntr = shifts_perscene.shape
    rvals = np.full((nroi_n, ntr), np.nan)
    for roi in range(nroi_n):
        for tr in range(ntr):
            z_data = conv_r2z(shifts_perscene[roi, :, :, tr])   # (33, 40)
            scene_rs = [
                nanspearmanr(z_data[:, scene], imp_updates[:, scene])[0]
                for scene in range(nscene)
            ]
            rvals[roi, tr] = np.nanmean(scene_rs)
    return rvals   # (nroi, 9)


# ---------------------------------------------------------------------------
# Permutation null: subject-order shuffle, parallel over ROIs
# ---------------------------------------------------------------------------
def compute_permutation_null(shifts_perscene, imp_updates, n_perms=1000, seed=0):
    """
    Shuffles subject order for each permutation (maintains scene structure).
    Returns (n_perms, nroi, 9).
    Parallelised over ROIs.
    """
    rng = np.random.RandomState(seed)
    nroi_n, nsubj_n, nscene, ntr = shifts_perscene.shape

    # Pre-generate permutation indices so all ROIs use the same shuffles
    perm_indices = [rng.permutation(nsubj_n) for _ in range(n_perms)]

    def _roi_null(roi):
        null_roi = np.full((n_perms, ntr), np.nan)
        for perm, idx in enumerate(perm_indices):
            shuffled = imp_updates[idx, :]          # (33, 40) — subjects shuffled
            for tr in range(ntr):
                z_data = conv_r2z(shifts_perscene[roi, :, :, tr])
                scene_rs = [
                    nanspearmanr(z_data[:, scene], shuffled[:, scene])[0]
                    for scene in range(nscene)
                ]
                null_roi[perm, tr] = np.nanmean(scene_rs)
        return null_roi   # (n_perms, 9)

    print(f'    Parallel null over {nroi_n} ROIs × {n_perms} permutations...')
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_roi_null)(roi) for roi in range(nroi_n)
    )
    return np.stack(results, axis=1)   # (n_perms, nroi, 9)


# ---------------------------------------------------------------------------
# Statistical testing (same logic as step08)
# ---------------------------------------------------------------------------
def test_significance(rvals, null_rvals, nroi_cor=100):
    """Single-stage FDR across cortical ROIs per TR."""
    ntr = rvals.shape[1]
    sig_rois_per_tp = []
    for tr in range(ntr):
        pvals = [
            onetail_p(rvals[roi, tr], null_rvals[:, roi, tr])
            for roi in range(nroi_cor)
        ]
        corrected = fdr_correct(pvals)
        sig_rois_per_tp.append(np.where(np.array(corrected) < 0.05)[0])
    return sig_rois_per_tp


def double_threshold(rvals, null_rvals, sig_rois_step07, nroi_cor=100):
    """FDR on step07-subset only."""
    ntr = rvals.shape[1]
    sig_rois_double = []
    for tr in range(ntr):
        step07 = sig_rois_step07[tr]
        if len(step07) == 0:
            sig_rois_double.append(np.array([], dtype=int))
            continue
        pvals = np.array([
            onetail_p(rvals[roi, tr], null_rvals[:, roi, tr])
            for roi in range(nroi_cor)
        ])
        corrected = fdr_correct(pvals[step07])
        keep = np.where(np.array(corrected) < 0.05)[0]
        sig_rois_double.append(np.array(step07)[keep])
    return sig_rois_double


# ---------------------------------------------------------------------------
# Brain surface plotting (identical to step08)
# ---------------------------------------------------------------------------
def surface_arrays_from_values(values, mask, ref_img, fsaverage, sig_rois=None):
    """Project ROI values onto fsaverage left/right cortical surfaces."""
    brain = build_brain_volume(mask, values, sig_rois=sig_rois)
    brain_map = new_img_like(ref_img, brain)
    surf_l = surface.vol_to_surf(brain_map, fsaverage.pial_left)
    surf_r = surface.vol_to_surf(brain_map, fsaverage.pial_right)
    return surf_l, surf_r


def draw_surface_views(axes, surf_l, surf_r, fsaverage, vmin=-0.20, vmax=0.20):
    """Draw lateral and medial surface views on a 2x2 axes block."""
    sns.set_context('paper')
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


def plot_surface_map(values, mask, ref_img, fsaverage,
                     sig_rois=None, title='', save_path=None,
                     vmin=-0.20, vmax=0.20):
    surf_l, surf_r = surface_arrays_from_values(
        values, mask, ref_img, fsaverage, sig_rois=sig_rois)
    sns.set_context('paper')
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    draw_surface_views(axes, surf_l, surf_r, fsaverage, vmin=vmin, vmax=vmax)
    plt.suptitle(title, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.close()


def plot_surface_grid(surface_pairs, fsaverage, tp_labels, title, save_path,
                      vmin=-0.20, vmax=0.20):
    """Save a single 4x9 composite figure spanning TR -5 to +3."""
    sns.set_context('paper')
    fig, axes = plt.subplots(
        4, len(tp_labels),
        figsize=(2.0 * len(tp_labels), 7.6),
        subplot_kw={'projection': '3d'})

    row_labels = ['L lateral', 'R lateral', 'L medial', 'R medial']
    for col, ((surf_l, surf_r), tp_label) in enumerate(zip(surface_pairs, tp_labels)):
        draw_surface_views(
            axes[:, col].reshape(2, 2), surf_l, surf_r, fsaverage,
            vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'TR {tp_label:+d}', fontsize=10, pad=10)

    for row, label in enumerate(row_labels):
        axes[row, 0].text2D(-0.12, 0.5, label, transform=axes[row, 0].transAxes,
                            rotation=90, va='center', ha='right', fontsize=9)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.02,
                        wspace=0.02, hspace=0.02)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)
    n_perms = 1000

    # ------------------------------------------------------------------
    # 1. Load cached pattern shift arrays
    # ------------------------------------------------------------------
    print('Loading cached pattern shift arrays...')
    char_shifts = np.load(
        os.path.join(NU_DIR, 'step08_shifts_perscene.npy'),
        allow_pickle=True)
    nonchar_shifts = np.load(
        os.path.join(NU_DIR, 'step08_nonchar_shifts_perscene.npy'),
        allow_pickle=True)
    print(f'  Character shifts:     {char_shifts.shape}')    # (116, 33, 40, 9)
    print(f'  Non-character shifts: {nonchar_shifts.shape}') # (116, 33, 40, 9)

    # ------------------------------------------------------------------
    # 2. Load event index table (needed for scene-order alignment)
    # ------------------------------------------------------------------
    event_idxs = pd.read_csv(
        os.path.join(COLLAB_DIR, 'socialaha-fMRI', 'socialaha_groupscene.csv'))

    # ------------------------------------------------------------------
    # 3. Compute person / situation impression updates
    # ------------------------------------------------------------------
    transcribe_path = os.path.join(
        COLLAB_DIR, 'socialaha-fMRI',
        'socialaha_transcribe_rhea_char_updated_clean_jin.csv')

    print('Computing person-model impression updates...')
    person_updates = compute_semantic_updates(
        transcribe_path, flist, event_idxs, score_type='person')
    print(f'  Shape: {person_updates.shape}   '
          f'NaN rate: {np.isnan(person_updates).mean():.2%}')

    print('Computing situation-model impression updates...')
    situation_updates = compute_semantic_updates(
        transcribe_path, flist, event_idxs, score_type='situation')
    print(f'  Shape: {situation_updates.shape}   '
          f'NaN rate: {np.isnan(situation_updates).mean():.2%}')

    # ------------------------------------------------------------------
    # 4. Load atlas + surface template
    # ------------------------------------------------------------------
    mask    = build_nifti_mask(nroi_cor, nroi_sub, COLLAB_DIR)
    ref_img = load_img(os.path.join(
        COLLAB_DIR, 'template', 'tpl-MNI152NLin2009cAsym',
        'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'
        '100Parcels17Networks_dseg.nii.gz'))
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')

    # ------------------------------------------------------------------
    # 5. Step07 significant ROIs for double-thresholding
    # ------------------------------------------------------------------
    nonchar_sig_path = os.path.join(NU_DIR, 'step07_nonchar_sig_rois.npy')
    if os.path.exists(nonchar_sig_path):
        sig_rois_step07_nonchar = list(
            np.load(nonchar_sig_path, allow_pickle=True))
    else:
        print(f'WARNING: {nonchar_sig_path} not found — '
              'double-threshold for nonchar conditions will be empty.')
        sig_rois_step07_nonchar = [np.array([], dtype=int)] * 9

    step07_rois = {
        'char':    SIG_ROIS_STEP07_CHAR,
        'nonchar': sig_rois_step07_nonchar,
    }

    # ------------------------------------------------------------------
    # 6. Compute r-values and permutation nulls for all 4 conditions
    # ------------------------------------------------------------------
    conditions = [
        ('char',    'person',    char_shifts,    person_updates),
        ('char',    'situation', char_shifts,    situation_updates),
        ('nonchar', 'person',    nonchar_shifts, person_updates),
        ('nonchar', 'situation', nonchar_shifts, situation_updates),
    ]

    rvals_all    = {}
    null_rvals_all = {}
    sig_step13   = {}
    sig_double   = {}

    for aha_type, imp_type, shifts, imp_updates in conditions:
        key = f'{aha_type}_{imp_type}'
        print(f'\n--- Condition: {key} ---')

        # r-values
        print('  Computing r-values...')
        rvals = compute_rvals(shifts, imp_updates)
        rvals_all[key] = rvals
        np.save(os.path.join(NU_DIR, f'step13_rvals_{key}.npy'), rvals)

        # # Permutation null (cached)  — commented out for raw-map preview
        # null_cache = os.path.join(NU_DIR, f'step13_null_{key}.npy')
        # if os.path.exists(null_cache):
        #     print(f'  Loading cached null from {null_cache}')
        #     null_rv = np.load(null_cache)
        # else:
        #     print(f'  Computing permutation null (n={n_perms})...')
        #     null_rv = compute_permutation_null(
        #         shifts, imp_updates, n_perms=n_perms, seed=42)
        #     np.save(null_cache, null_rv)
        # null_rvals_all[key] = null_rv
        # print(f'  Null shape: {null_rv.shape}')   # (1000, 116, 9)

        # # Significance — commented out for raw-map preview
        # sig_s13 = test_significance(rvals, null_rv, nroi_cor=nroi_cor)
        # sig_db  = double_threshold(
        #     rvals, null_rv, step07_rois[aha_type], nroi_cor=nroi_cor)
        # sig_step13[key] = sig_s13
        # sig_double[key] = sig_db

        # np.save(os.path.join(NU_DIR, f'step13_sig_rois_{key}_step13.npy'),
        #         np.array(sig_s13, dtype=object))
        # np.save(os.path.join(NU_DIR, f'step13_sig_rois_{key}_double.npy'),
        #         np.array(sig_db, dtype=object))

        # print(f'\n  {"TR":>5}  {"step13":>8}  {"double":>8}')
        # for tr in range(9):
        #     print(f'  {tr-5:>+5d}  {len(sig_s13[tr]):>8d}  {len(sig_db[tr]):>8d}'
        #           f'   {list(sig_db[tr])}')

    # Stack all r-values (4 × 116 × 9)
    rvals_stack = np.array([rvals_all[f'{a}_{i}'] for a, i, _, _ in conditions])
    np.save(os.path.join(NU_DIR, 'step13_rvals.npy'), rvals_stack)

    # ------------------------------------------------------------------
    # 7. Dissociation contrasts
    # ------------------------------------------------------------------
    # Positive = hypothesised direction
    dissoc_char    = rvals_all['char_person']       - rvals_all['char_situation']
    dissoc_nonchar = rvals_all['nonchar_situation'] - rvals_all['nonchar_person']
    np.save(os.path.join(NU_DIR, 'step13_dissociation_rvals.npy'),
            np.array([dissoc_char, dissoc_nonchar]))

    # ------------------------------------------------------------------
    # 8. Brain surface maps for all 4 conditions  [COMMENTED OUT]
    # ------------------------------------------------------------------
    # labels = {
    #     'char_person':       ('Char aha | Person-model Δ',    'blue'),
    #     'char_situation':    ('Char aha | Situation-model Δ', 'blue'),
    #     'nonchar_person':    ('Nonchar aha | Person-model Δ', 'red'),
    #     'nonchar_situation': ('Nonchar aha | Situation-model Δ', 'red'),
    # }

    # for aha_type, imp_type, _, _ in conditions:
    #     key = f'{aha_type}_{imp_type}'
    #     short_label = labels[key][0]
    #     rvals = rvals_all[key]
    #     print(f'\nPlotting brain maps: {key}...')

    #     sub_dir = os.path.join(FIG_DIR, key)
    #     os.makedirs(sub_dir, exist_ok=True)
    #     raw_surface_pairs = []

    #     for tr in range(9):
    #         tp_label = tr - 5
    #         r_tr = rvals[:, tr].copy()
    #         r_tr[nroi_cor:] = np.nan   # cortical surface only
    #         raw_surface_pairs.append(
    #             surface_arrays_from_values(r_tr, mask, ref_img, fsaverage, sig_rois=None))

    #         plot_surface_map(
    #             r_tr, mask, ref_img, fsaverage,
    #             sig_rois=None,
    #             title=f'{short_label} | Raw | TR {tp_label:+d}',
    #             save_path=os.path.join(sub_dir, f'raw_TR{tp_label:+d}.png'))

    #         # plot_surface_map(  # thresholded — needs null
    #         #     r_tr, mask, ref_img, fsaverage,
    #         #     sig_rois=sig_step13[key][tr],
    #         #     title=f'{short_label} | Thresholded | TR {tp_label:+d}',
    #         #     save_path=os.path.join(sub_dir, f'thresholded_TR{tp_label:+d}.png'))

    #         # plot_surface_map(  # double-thresh — needs null
    #         #     r_tr, mask, ref_img, fsaverage,
    #         #     sig_rois=sig_double[key][tr],
    #         #     title=f'{short_label} | Double-thresh (step07∩step13) | TR {tp_label:+d}',
    #         #     save_path=os.path.join(sub_dir, f'doublethresh_TR{tp_label:+d}.png'))

    #     plot_surface_grid(
    #         raw_surface_pairs, fsaverage, list(range(-5, 4)),
    #         title=f'{short_label} | Raw',
    #         save_path=os.path.join(sub_dir, 'raw_allTRs.png'))
    #     print(f'  Saved 9 raw maps to {sub_dir}/')

    # ------------------------------------------------------------------
    # 9. Dissociation contrast brain maps  [COMMENTED OUT]
    # ------------------------------------------------------------------
    # print('\nPlotting dissociation contrast maps...')
    # dissoc_dir = os.path.join(FIG_DIR, 'dissociation')
    # os.makedirs(dissoc_dir, exist_ok=True)
    # char_dissoc_surface_pairs = []
    # nonchar_dissoc_surface_pairs = []

    # for tr in range(9):
    #     tp_label = tr - 5

    #     d_char = dissoc_char[:, tr].copy()
    #     d_char[nroi_cor:] = np.nan
    #     char_dissoc_surface_pairs.append(
    #         surface_arrays_from_values(d_char, mask, ref_img, fsaverage, sig_rois=None))
    #     plot_surface_map(
    #         d_char, mask, ref_img, fsaverage,
    #         sig_rois=None,
    #         title=f'Char aha: Δr(person − situation) | TR {tp_label:+d}',
    #         save_path=os.path.join(dissoc_dir, f'char_dissoc_TR{tp_label:+d}.png'),
    #         vmin=-0.15, vmax=0.15)

    #     d_nonchar = dissoc_nonchar[:, tr].copy()
    #     d_nonchar[nroi_cor:] = np.nan
    #     nonchar_dissoc_surface_pairs.append(
    #         surface_arrays_from_values(d_nonchar, mask, ref_img, fsaverage, sig_rois=None))
    #     plot_surface_map(
    #         d_nonchar, mask, ref_img, fsaverage,
    #         sig_rois=None,
    #         title=f'Non-char aha: Δr(situation − person) | TR {tp_label:+d}',
    #         save_path=os.path.join(dissoc_dir, f'nonchar_dissoc_TR{tp_label:+d}.png'),
    #         vmin=-0.15, vmax=0.15)

    # plot_surface_grid(
    #     char_dissoc_surface_pairs, fsaverage, list(range(-5, 4)),
    #     title='Char aha: delta r (person - situation)',
    #     save_path=os.path.join(dissoc_dir, 'char_dissoc_allTRs.png'),
    #     vmin=-0.15, vmax=0.15)
    # plot_surface_grid(
    #     nonchar_dissoc_surface_pairs, fsaverage, list(range(-5, 4)),
    #     title='Non-char aha: delta r (situation - person)',
    #     save_path=os.path.join(dissoc_dir, 'nonchar_dissoc_allTRs.png'),
    #     vmin=-0.15, vmax=0.15)

    # print(f'  Saved dissociation maps to {dissoc_dir}/')

    # ------------------------------------------------------------------
    # 10. ROI 98 timecourse: all 4 conditions
    # ------------------------------------------------------------------
    roi98 = 97   # 0-indexed (ROI 98 = right STS)
    tps   = np.arange(-5, 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    line_styles  = {'person': '-',  'situation': '--'}
    line_colors  = {'char': '#2166ac', 'nonchar': '#d6604d'}
    imp_labels   = {'person': 'Person model', 'situation': 'Situation model'}

    for ax, (aha_type, aha_label) in zip(
            axes, [('char', 'Character aha'), ('nonchar', 'Non-character aha')]):
        for imp_type in ['person', 'situation']:
            key = f'{aha_type}_{imp_type}'
            ax.plot(tps, rvals_all[key][roi98],
                    label=imp_labels[imp_type],
                    color=line_colors[aha_type],
                    linestyle=line_styles[imp_type],
                    linewidth=2)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.axvline(0, color='grey',  linewidth=0.8, linestyle=':')
        ax.set_xlabel('TR relative to aha moment', fontsize=11)
        ax.set_ylabel('Spearman r (scene-averaged)', fontsize=11)
        ax.set_title(f'ROI 98 (rSTS) — {aha_label}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xticks(tps)

    fig.suptitle(
        'Neural shift ~ Impression model update  |  ROI 98 timecourse\n'
        'Solid = person model  |  Dashed = situation model',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'figures', 'step13_roi98_dissociation.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out}')

    # ------------------------------------------------------------------
    # 11. Summary bar chart: r-values at TR 0 across double-thresh ROIs
    #     — commented out for raw-map preview (needs null)
    # ------------------------------------------------------------------
    # tr0_idx = 5   # index of TR 0 in the 9-element window
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # bar_colors = {'person': '#2166ac', 'situation': '#d6604d'}
    # for ax, (aha_type, primary_imp, title_str) in zip(axes, [
    #     ('char',    'person',    'Character aha\n(double-thresh ROIs from char×person)'),
    #     ('nonchar', 'situation', 'Non-char aha\n(double-thresh ROIs from nonchar×situation)'),
    # ]):
    #     primary_key = f'{aha_type}_{primary_imp}'
    #     primary_rois = sig_double[primary_key][tr0_idx]
    #     if len(primary_rois) == 0:
    #         ax.text(0.5, 0.5, 'No significant ROIs\nat TR 0',
    #                 ha='center', va='center', transform=ax.transAxes, fontsize=12)
    #         ax.set_title(title_str, fontsize=11, fontweight='bold')
    #         continue
    #     r_person    = rvals_all[f'{aha_type}_person'][primary_rois, tr0_idx]
    #     r_situation = rvals_all[f'{aha_type}_situation'][primary_rois, tr0_idx]
    #     x = np.arange(len(primary_rois))
    #     width = 0.35
    #     ax.bar(x - width / 2, r_person,    width, label='Person model',
    #            color=bar_colors['person'],    alpha=0.85)
    #     ax.bar(x + width / 2, r_situation, width, label='Situation model',
    #            color=bar_colors['situation'], alpha=0.85)
    #     ax.axhline(0, color='black', linewidth=0.8)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels([f'ROI {r + 1}' for r in primary_rois],
    #                         rotation=45, ha='right', fontsize=7)
    #     ax.set_ylabel('Spearman r', fontsize=10)
    #     ax.set_title(title_str, fontsize=11, fontweight='bold')
    #     ax.legend(fontsize=9)
    # fig.suptitle(
    #     'Double dissociation at TR 0: person vs situation model update\n'
    #     'ROIs shown = double-threshold significant in primary condition',
    #     fontsize=12, fontweight='bold')
    # fig.tight_layout()
    # out = os.path.join(RESULTS_DIR, 'figures', 'step13_dissociation_barchart.png')
    # fig.savefig(out, dpi=150, bbox_inches='tight')
    # plt.close(fig)
    # print(f'Saved: {out}')

    # ------------------------------------------------------------------
    # 12. Dissociation timecourse summary across all cortical ROIs
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (dissoc, aha_type, dissoc_label) in zip(axes, [
        (dissoc_char,    'char',    'Char aha: Δr(person − situation)'),
        (dissoc_nonchar, 'nonchar', 'Non-char aha: Δr(situation − person)'),
    ]):
        # Mean ± SE over cortical ROIs
        cortical_dissoc = dissoc[:nroi_cor, :]   # (100, 9)
        mean_d = np.nanmean(cortical_dissoc, axis=0)
        se_d   = np.nanstd(cortical_dissoc, axis=0) / np.sqrt(nroi_cor)
        color  = '#2166ac' if aha_type == 'char' else '#d6604d'

        ax.fill_between(tps, mean_d - se_d, mean_d + se_d, alpha=0.25, color=color)
        ax.plot(tps, mean_d, color=color, linewidth=2)
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.axvline(0, color='grey',  linewidth=0.8, linestyle=':')
        ax.set_xlabel('TR relative to aha moment', fontsize=11)
        ax.set_ylabel('Δr (mean ± SE over cortical ROIs)', fontsize=11)
        ax.set_title(dissoc_label, fontsize=12, fontweight='bold')
        ax.set_xticks(tps)

    fig.suptitle(
        'Dissociation contrast timecourse (person vs situation model)\n'
        'Positive = hypothesised direction',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'figures', 'step13_dissociation_timecourse.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # ------------------------------------------------------------------
    # 13. Bar plots with t-tests: mean r in step07 sig ROIs
    #     Two versions: all 9 TRs and TRs -2 to +1
    # ------------------------------------------------------------------
    def sig_str(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

    def add_bracket(ax, x0, x1, y, label, dy=0.004):
        ax.plot([x0, x0, x1, x1], [y, y + dy, y + dy, y], 'k-', linewidth=1)
        ax.text((x0 + x1) / 2, y + dy + 0.001, label,
                ha='center', va='bottom', fontsize=8)

    tr_windows = {
        'all_9TRs': (list(range(9)), 'All 9 TRs (−5 to +3)'),
    }

    for window_key, (tr_idx, window_label) in tr_windows.items():

        # Step07 sig ROIs: union across this TR window
        char_sig_rois = np.unique(np.concatenate(
            [SIG_ROIS_STEP07_CHAR[t] for t in tr_idx]
        )).astype(int)
        nonchar_sig_rois = np.unique(np.concatenate(
            [sig_rois_step07_nonchar[t] for t in tr_idx]
        )).astype(int)

        # Mean r per ROI, averaged over TRs in window
        def roi_means(key, rois):
            if len(rois) == 0:
                return np.array([np.nan])
            return rvals_all[key][np.ix_(rois, tr_idx)].mean(axis=1)

        r_cp = roi_means('char_person',       char_sig_rois)     # char aha × person
        r_cs = roi_means('char_situation',    char_sig_rois)     # char aha × situation
        r_np = roi_means('nonchar_person',    nonchar_sig_rois)  # nonchar aha × person
        r_ns = roi_means('nonchar_situation', nonchar_sig_rois)  # nonchar aha × situation

        # --- Statistical tests ---
        # Paired t: within char aha ROIs, person vs situation
        t_c, p_c = ttest_rel(r_cp, r_cs)
        # Paired t: within nonchar aha ROIs, situation vs person (hypothesized direction)
        t_nc, p_nc = ttest_rel(r_ns, r_np)
        # Independent t: char×person (char ROIs) vs nonchar×person (nonchar ROIs)
        t_p, p_p = ttest_ind(r_cp, r_np)
        # Independent t: nonchar×situation (nonchar ROIs) vs char×situation (char ROIs)
        t_s, p_s = ttest_ind(r_ns, r_cs)

        print(f'\n=== {window_label} ===')
        print(f'  Char sig ROIs    (n={len(char_sig_rois)}): {list(char_sig_rois)}')
        print(f'  Nonchar sig ROIs (n={len(nonchar_sig_rois)}): {list(nonchar_sig_rois)}')
        print(f'  Paired t (char ROIs: person vs situation):      '
              f't={t_c:.3f}  p={p_c:.4f}  {sig_str(p_c)}')
        print(f'  Paired t (nonchar ROIs: situation vs person):   '
              f't={t_nc:.3f}  p={p_nc:.4f}  {sig_str(p_nc)}')
        print(f'  Indep t  (person model: char vs nonchar ROIs):  '
              f't={t_p:.3f}  p={p_p:.4f}  {sig_str(p_p)}')
        print(f'  Indep t  (situation model: nonchar vs char ROIs): '
              f't={t_s:.3f}  p={p_s:.4f}  {sig_str(p_s)}')

        # --- Bar plot ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        bar_styles = [
            dict(facecolor='white',   edgecolor='black', hatch='',    linewidth=1.5),  # person
            dict(facecolor='#555555', edgecolor='black', hatch='///', linewidth=1.5),  # situation
        ]
        rng = np.random.RandomState(42)

        bar_width = 0.3
        xs = [0, 0.38]

        for ax, r_person, r_sit, panel_title in [
            (axes[0], r_cp, r_cs, f'Character aha ROIs (n={len(char_sig_rois)})'),
            (axes[1], r_np, r_ns, f'Non-character aha ROIs (n={len(nonchar_sig_rois)})'),
        ]:
            vals   = [r_person, r_sit]
            means  = [np.nanmean(v) for v in vals]
            sems   = [np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))) for v in vals]

            for xi, mean, sem, style in zip(xs, means, sems, bar_styles):
                ax.bar(xi, mean, yerr=sem, capsize=6, width=bar_width,
                       facecolor=style['facecolor'], edgecolor=style['edgecolor'],
                       hatch=style['hatch'], linewidth=style['linewidth'],
                       error_kw=dict(linewidth=1.5, capthick=1.5))
            ax.axhline(0, color='black', linewidth=0.8)

            # Individual ROI dots (jittered)
            for xi, v in zip(xs, vals):
                jitter = rng.uniform(-0.06, 0.06, len(v))
                ax.scatter(xi + jitter, v, color='black', alpha=0.45, s=20, zorder=5)

            # # Paired t-test bracket
            # ymax = max(m + s for m, s in zip(means, sems))
            # bracket_y = ymax + abs(ymax) * 0.12 + 0.005
            # label = f'paired t={t_paired:.2f}, {sig_str(p_paired)} (p={p_paired:.3f})'
            # add_bracket(ax, 0, 1, bracket_y, label, dy=0.004)

            ax.set_xticks(xs)
            ax.set_xticklabels(['Person model', 'Situation model'], fontsize=11)
            ax.set_ylabel('Mean Spearman r (ROI-averaged)', fontsize=10)
            ax.set_title(panel_title, fontsize=11, fontweight='bold')
            ax.set_xlim(-0.35, 0.73)

        # Independent t-tests annotated in figure text below title
        fig.suptitle(
            f'Neural shift × Impression model update in step07 sig ROIs\n'
            f'{window_label}\n'
            f'Indep t (person: char vs nonchar ROIs): t={t_p:.2f}, {sig_str(p_p)} (p={p_p:.3f})   |   '
            f'Indep t (situation: nonchar vs char ROIs): t={t_s:.2f}, {sig_str(p_s)} (p={p_s:.3f})',
            fontsize=10, fontweight='bold')
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, 'figures',
                           f'step13_dissociation_barplot_{window_key}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

    print(f'\nDone. All outputs in {NU_DIR}/ and {FIG_DIR}/')


if __name__ == '__main__':
    main()
