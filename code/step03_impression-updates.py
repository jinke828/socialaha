"""
step03_impression-updates.py

Replicates the analysis in step03_impression-updates.ipynb:
  1. Load Google Universal Sentence Encoder (USE) and compute 512-dim embeddings
     for each participant's impression speech, both per-run (all characters combined)
     and per-run per-character.
  2. Test whether within-subject impression similarity exceeds between-subject
     similarity (Wilcoxon rank-sum test).
  3. Test whether impression similarity declines as a function of temporal distance
     between runs (Spearman correlation per character).
  4. Build and save the long-form dataframe (df.csv) used by step04_plot_impression_updates.R.
  5. Compute between-subject impression similarity matrices per run and per
     run-by-character, and save them.

All paths are relative to the directory containing this script.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums, spearmanr
import tensorflow as tf
import tensorflow_hub as hub
from absl import logging

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # one level up from code/

TRANSCRIBE_CSV = os.path.join(ROOT_DIR, 'socialaha-collab', 'socialaha-fMRI',
                              'socialaha_transcribe_rhea_char_updated_clean_jin.csv')
EMBED_DIR      = os.path.join(ROOT_DIR, 'data', 'beh', 'embeddings')
SIM_DIR        = os.path.join(ROOT_DIR, 'data', 'beh', 'similarity')
RESULTS_DIR    = os.path.join(ROOT_DIR, 'results', 'impression_updates')
FIG_DIR        = os.path.join(ROOT_DIR, 'results', 'figures')

for d in [EMBED_DIR, SIM_DIR, RESULTS_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Participant lists
# ---------------------------------------------------------------------------
flist = {
    1: ['sub-1001','sub-1005','sub-1008','sub-1011','sub-1014','sub-1017',
        'sub-1020','sub-1023','sub-1026','sub-1029','sub-1033','sub-1039'],
    2: ['sub-2006','sub-2009','sub-2012','sub-2015','sub-2018','sub-2021',
        'sub-2024','sub-2027','sub-2034','sub-2038','sub-2040'],  # sub-2030 excluded
    3: ['sub-3004','sub-3007','sub-3013','sub-3016','sub-3019','sub-3022',
        'sub-3025','sub-3031','sub-3037','sub-3041'],             # sub-3010, sub-3028 excluded
}
CHAR_NAMES = ['Jack', 'Kate', 'Randall', 'Kevin']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors (handles NaN vectors → NaN)."""
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return np.nan
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def nan_spearmanr(tc1, tc2):
    """Spearman r after dropping positions where either array is NaN."""
    tc1, tc2 = np.asarray(tc1, dtype=float), np.asarray(tc2, dtype=float)
    nan_idx = np.union1d(np.where(np.isnan(tc1))[0], np.where(np.isnan(tc2))[0])
    tc1 = np.delete(tc1, nan_idx)
    tc2 = np.delete(tc2, nan_idx)
    return spearmanr(tc1, tc2)


# ---------------------------------------------------------------------------
# 1. Load Google Universal Sentence Encoder
# ---------------------------------------------------------------------------
print("Loading Google Universal Sentence Encoder …")
logging.set_verbosity(logging.ERROR)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(module_url)
print(f"  module {module_url} loaded")

def embed(texts):
    return use_model(texts).numpy()


# ---------------------------------------------------------------------------
# 2. Load transcription CSV
# ---------------------------------------------------------------------------
print("\nLoading transcription CSV …")
tst = pd.read_csv(TRANSCRIBE_CSV)
print(f"  {len(tst)} rows loaded")

NAN_VEC = np.full(512, np.nan)

# ---------------------------------------------------------------------------
# 3. Compute embeddings
# ---------------------------------------------------------------------------
print("\nComputing embeddings …")
embeddings_byrun        = {}   # [groupid, sub_idx, run] → 512-dim array
embeddings_byrun_bychar = {}   # [groupid, sub_idx, run] → (4, 512) array

for groupid in range(1, 4):
    for sub_idx, sub_str in enumerate(flist[groupid]):
        subid = int(sub_str.split('-')[1])
        this_sub = tst[tst['subject'] == subid]
        for run in range(1, 11):
            this_run = this_sub[this_sub['run'] == run]
            full_txt  = ''
            char_vecs = []
            for char in range(1, 5):
                this_char  = this_run[this_run['character'] == char]
                if len(this_char) == 0:
                    char_vecs.append(NAN_VEC.copy())
                    continue
                speech = this_char['transcribe'].values[0]
                if pd.isna(speech):
                    char_vecs.append(NAN_VEC.copy())
                else:
                    full_txt += speech
                    char_vecs.append(embed([speech])[0])
            embeddings_byrun_bychar[groupid, sub_idx, run] = np.array(char_vecs)  # (4, 512)
            embeddings_byrun[groupid, sub_idx, run] = (
                embed([full_txt])[0] if len(full_txt) > 0 else NAN_VEC.copy()
            )
        print(f"  group {groupid}  {sub_str}  done")

# Save
np.save(os.path.join(EMBED_DIR, 'speech-embeddings_byrun.npy'),
        embeddings_byrun, allow_pickle=True)
np.save(os.path.join(EMBED_DIR, 'speech-embeddings_byrun_bychar.npy'),
        embeddings_byrun_bychar, allow_pickle=True)
print("  embeddings saved")

# ---------------------------------------------------------------------------
# 4. Within vs. Between subject similarity  (run-level, all 4 chars combined)
# ---------------------------------------------------------------------------
print("\n--- Within vs. Between subject impression similarity (run-level) ---")

# 4a. All pairwise run combinations (flat)
within_flat, between_flat = [], []
for groupid in range(1, 4):
    n = len(flist[groupid])
    for sub in range(n):
        w = [cosine_similarity(embeddings_byrun[groupid, sub, r1],
                               embeddings_byrun[groupid, sub, r2])
             for r1 in range(1, 10) for r2 in range(r1+1, 11)]
        b_all = []
        for sub2 in range(n):
            b_sub = [cosine_similarity(embeddings_byrun[groupid, sub, r1],
                                       embeddings_byrun[groupid, sub2, r2])
                     for r1 in range(1, 10) for r2 in range(r1+1, 11)]
            b_all.append(b_sub)
        b_all = np.array(b_all, dtype=float)
        b_all = np.delete(b_all, sub, axis=0)
        b_mean = np.nanmean(b_all, axis=0)
        within_flat.append(np.array(w))
        between_flat.append(b_mean)

res_flat = ranksums(np.array(within_flat).reshape(-1),
                    np.array(between_flat).reshape(-1), nan_policy='omit')
print(f"  [flat]  statistic={res_flat.statistic:.4f}  p={res_flat.pvalue:.2e}")

# 4b. Per-subject averaged (used for the figure)
within_subj  = np.array([np.nanmean(w) for w in within_flat])
between_subj = np.array([np.nanmean(b) for b in between_flat])
res_subj = ranksums(within_subj, between_subj, nan_policy='omit')
print(f"  [subj]  statistic={res_subj.statistic:.4f}  p={res_subj.pvalue:.2e}")

# Figure: boxplot + paired lines
df_long = pd.DataFrame({'within': within_subj, 'between': between_subj}) \
            .melt(var_name='Condition', value_name='Impression similarity')

plt.figure(figsize=(3, 4), dpi=400)
ax = sns.boxplot(data=df_long, x='Condition', y='Impression similarity',
                 hue='Condition', palette=["#8CD790", "#FDD692"],
                 width=0.4, fliersize=0, zorder=1, dodge=False)
for i in range(len(within_subj)):
    plt.plot([0, 1], [within_subj[i], between_subj[i]],
             color='lightgray', linewidth=0.5, zorder=0)
sns.stripplot(data=df_long, x='Condition', y='Impression similarity',
              color='black', size=3, jitter=False, zorder=2)
ax.set_ylim(0.45, 0.75)
ax.set_ylabel("Impression similarity", fontsize=14)
ax.set_xlabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'within-between.png')
plt.savefig(fig_path, dpi=1000)
plt.close()
print(f"  figure saved → {fig_path}")

# ---------------------------------------------------------------------------
# 5. Within vs. Between  (character-level, average across 4 chars)
# ---------------------------------------------------------------------------
print("\n--- Within vs. Between subject similarity (character-level) ---")

within_char_flat, between_char_flat = [], []
for groupid in range(1, 4):
    n = len(flist[groupid])
    for sub in range(n):
        w = []
        for r1 in range(1, 10):
            for r2 in range(r1+1, 11):
                sim_chars = [cosine_similarity(embeddings_byrun_bychar[groupid, sub, r1][c],
                                               embeddings_byrun_bychar[groupid, sub, r2][c])
                             for c in range(4)]
                w.append(np.nanmean(sim_chars))
        b_all = []
        for sub2 in range(n):
            b_sub = []
            for r1 in range(1, 10):
                for r2 in range(r1+1, 11):
                    sim_chars = [cosine_similarity(embeddings_byrun_bychar[groupid, sub, r1][c],
                                                   embeddings_byrun_bychar[groupid, sub2, r2][c])
                                 for c in range(4)]
                    b_sub.append(np.nanmean(sim_chars))
            b_all.append(b_sub)
        b_all = np.array(b_all, dtype=float)
        b_all = np.delete(b_all, sub, axis=0)
        b_mean = np.nanmean(b_all, axis=0)
        within_char_flat.append(np.array(w))
        between_char_flat.append(b_mean)

res_char = ranksums(np.array(within_char_flat).reshape(-1),
                    np.array(between_char_flat).reshape(-1), nan_policy='omit')
print(f"  statistic={res_char.statistic:.4f}  p={res_char.pvalue:.2e}")

# ---------------------------------------------------------------------------
# 6. Impression evolution: similarity as a function of run distance
# ---------------------------------------------------------------------------
print("\n--- Spearman r: impression similarity ~ run distance (per character) ---")

similarity_by_char = [[] for _ in range(4)]
distance_by_char   = [[] for _ in range(4)]

for chars in range(4):
    for groupid in range(1, 4):
        for sub in range(len(flist[groupid])):
            for r1 in range(1, 10):
                for r2 in range(r1+1, 11):
                    sim = cosine_similarity(embeddings_byrun_bychar[groupid, sub, r1][chars],
                                            embeddings_byrun_bychar[groupid, sub, r2][chars])
                    similarity_by_char[chars].append(float(sim))
                    distance_by_char[chars].append(r2 - r1)

for c in range(4):
    r, p = nan_spearmanr(similarity_by_char[c], distance_by_char[c])
    print(f"  {CHAR_NAMES[c]:8s}  r={r:.5f}  p={p:.5f}")

# ---------------------------------------------------------------------------
# 7. Build and save long-form dataframe (used by step04_plot_impression_updates.R)
# ---------------------------------------------------------------------------
print("\nBuilding df.csv …")
rows = {'similarity': [], 'distance': [], 'group': [], 'character': [], 'sub': []}

for chars in range(4):
    for groupid in range(1, 4):
        for sub in range(len(flist[groupid])):
            for r1 in range(1, 10):
                for r2 in range(r1+1, 11):
                    sim = cosine_similarity(embeddings_byrun_bychar[groupid, sub, r1][chars],
                                            embeddings_byrun_bychar[groupid, sub, r2][chars])
                    rows['similarity'].append(float(sim))
                    rows['distance'].append(r2 - r1)
                    rows['group'].append(groupid)
                    rows['character'].append(chars)
                    rows['sub'].append(sub)

df = pd.DataFrame(rows)
df_path = os.path.join(RESULTS_DIR, 'df.csv')
df.to_csv(df_path, index=False)
print(f"  saved → {df_path}  ({len(df)} rows)")

# ---------------------------------------------------------------------------
# 8. Between-subject impression similarity matrices  (per run and per run×char)
# ---------------------------------------------------------------------------
print("\nComputing between-subject similarity matrices …")

sim_byrun = {}
for groupid in range(1, 4):
    n = len(flist[groupid])
    for run in range(1, 11):
        sims = []
        for s1 in range(n - 1):
            for s2 in range(s1+1, n):
                sims.append(float(cosine_similarity(
                    embeddings_byrun[groupid, s1, run],
                    embeddings_byrun[groupid, s2, run]
                )))
        sim_byrun[groupid, run] = sims

sim_byrun_bychar = {}
for groupid in range(1, 4):
    n = len(flist[groupid])
    for run in range(1, 11):
        char_sims = []
        for char in range(1, 5):
            sims = []
            for s1 in range(n - 1):
                for s2 in range(s1+1, n):
                    sims.append(float(cosine_similarity(
                        embeddings_byrun_bychar[groupid, s1, run][char - 1],
                        embeddings_byrun_bychar[groupid, s2, run][char - 1]
                    )))
            char_sims.append(np.array(sims))
        sim_byrun_bychar[groupid, run] = np.array(char_sims)  # (4, n_pairs)

np.save(os.path.join(SIM_DIR, 'impressions_byrun.npy'),        sim_byrun,        allow_pickle=True)
np.save(os.path.join(SIM_DIR, 'impressions_byrun_bychar.npy'), sim_byrun_bychar, allow_pickle=True)
print(f"  similarity matrices saved → {SIM_DIR}")

print("\nDone.")
