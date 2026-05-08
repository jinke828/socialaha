"""
step09_aha_text_analysis.py

Outputs:
  ./results/figures/text_semantic_categories.png
  ./results/figures/text_person_situation_proportions.png
  ./results/figures/roc_curve.png
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
BASE_DIR        = '/Users/jinke/Desktop/socialaha_github'
CACHE_DIR       = os.path.join(BASE_DIR, 'results', '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', CACHE_DIR)
os.environ.setdefault('NUMBA_CACHE_DIR', CACHE_DIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ranksums, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANNOT_PATH      = os.path.join(BASE_DIR, 'data', 'beh', 'annotations', 'ahaannot_all.xlsx')
TRANSCRIBE_PATH = os.path.join(BASE_DIR, 'data', 'beh', 'annotations',
                               'socialaha_transcribe_rhea_aha.csv')
FIG_DIR         = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Semantic dictionaries
# ---------------------------------------------------------------------------
CATEGORIES = {
    'Mental state\n(ToM)': {
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
        'warmth', 'competence', 'agency', 'experience', 'trustworthiness', 'dominance', 'openness', 'conscientiousness',
        'extraversion', 'agreeableness', 'neuroticism', 'attractiveness', 'intelligence', 'kind', 'generous', 'selfish',
        'ambitious', 'insecure', 'confident', 'shy', 'honest', 'dishonest', 'caring', 'cold', 'warm', 'jealous', 'loyal',
        'responsible', 'determined', 'stubborn', 'sensitive', 'emotional', 'successful', 'talented', 'smart', 'proud', 'humble',
        'bitter', 'guilty', 'ashamed', 'strong', 'weak', 'depressed', 'anxious', 'upset', 'angry', 'frustrated', 'passionate',
        'devoted', 'dedicated', 'driven', 'identity', 'personality', 'character', 'nature'
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

PERSON_CATS = [
    'Dispositional\ntrait',
    'Mental state\n(ToM)',
    'Individual focus\n(he/she)',
]
SITUATION_CATS = [
    'Social\nrelationship',
    'Group focus\n(they/their)',
    'Event/action\n(situation)',
    'Causal\nlanguage',
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_annot = pd.read_excel(ANNOT_PATH)
for cat in ['character', 'relationship', 'retrieval', 'current',
            'inference', 'temporal', 'oops', 'causal']:
    df_annot[cat + '_all'] = df_annot[
        [cat + '_rater1', cat + '_rater2', cat + '_rater3']
    ].sum(axis=1)

df_t = (pd.read_csv(TRANSCRIBE_PATH)
          .drop_duplicates(subset=['subject', 'run', 'scene']))

df = df_annot.merge(
    df_t[['subject', 'run', 'scene', 'transcribe']],
    on=['subject', 'run', 'scene'], how='left',
)

df_char    = df[df['character_all'] >= 2].dropna(subset=['transcribe']).copy()
df_nonchar = df[df['character_all'] == 0].dropna(subset=['transcribe']).copy()

print(f"Character aha: n = {len(df_char)}")
print(f"Non-character aha: n = {len(df_nonchar)}")

def tokenise(text: str):
    return re.findall(r"[a-z']+", text.lower())

def category_rate(text: str, word_set: set) -> float:
    tokens = tokenise(text)
    if len(tokens) == 0:
        return np.nan
    return sum(1 for t in tokens if t in word_set) / len(tokens)

# ---------------------------------------------------------------------------
# 1. Semantic category scores
# ---------------------------------------------------------------------------
print("\n--- Semantic category scores ---")

results = []
for cat_name, word_set in CATEGORIES.items():
    char_scores    = df_char['transcribe'].apply(lambda t: category_rate(t, word_set)).dropna()
    nonchar_scores = df_nonchar['transcribe'].apply(lambda t: category_rate(t, word_set)).dropna()

    stat, pval = ranksums(char_scores, nonchar_scores)
    pooled_std = np.sqrt((char_scores.std()**2 + nonchar_scores.std()**2) / 2)
    cohens_d = (char_scores.mean() - nonchar_scores.mean()) / pooled_std if pooled_std > 0 else np.nan
    n1, n2 = len(char_scores), len(nonchar_scores)
    rank_biserial_r = stat / np.sqrt(n1 + n2)

    results.append({
        'category':        cat_name,
        'char_mean':       char_scores.mean(),
        'nonchar_mean':    nonchar_scores.mean(),
        'char_std':        char_scores.std(),
        'nonchar_std':     nonchar_scores.std(),
        'char_median':     char_scores.median(),
        'nonchar_median':  nonchar_scores.median(),
        'z_stat':          stat,
        'cohens_d':        cohens_d,
        'rank_biserial_r': rank_biserial_r,
        'n_char':          n1,
        'n_nonchar':       n2,
        'p_raw':           pval,
    })

res_df = pd.DataFrame(results)
_, res_df['p_fdr'], _, _ = multipletests(res_df['p_raw'], method='fdr_bh')
res_df['sig'] = res_df['p_fdr'] < 0.05

print(res_df[['category', 'char_mean', 'nonchar_mean', 'z_stat', 'cohens_d',
              'rank_biserial_r', 'p_raw', 'p_fdr', 'sig']].to_string(index=False))

# Compute per-transcription category rates and composites
for cat_name, word_set in CATEGORIES.items():
    df_char[cat_name]    = df_char['transcribe'].apply(lambda t: category_rate(t, word_set))
    df_nonchar[cat_name] = df_nonchar['transcribe'].apply(lambda t: category_rate(t, word_set))

for frame in (df_char, df_nonchar):
    frame['person_score']    = frame[PERSON_CATS].mean(axis=1)
    frame['situation_score'] = frame[SITUATION_CATS].mean(axis=1)
    total = frame['person_score'] + frame['situation_score']
    frame['person_prop']    = frame['person_score'] / total.replace(0, np.nan)
    frame['situation_prop'] = frame['situation_score'] / total.replace(0, np.nan)

comp_stats = []
for model_label, col_name in [
    ('Person model proportion', 'person_prop'),
    ('Situation model proportion', 'situation_prop'),
]:
    char_vals    = df_char[col_name].dropna()
    nonchar_vals = df_nonchar[col_name].dropna()
    stat, pval   = ranksums(char_vals, nonchar_vals)
    pooled_std   = np.sqrt((char_vals.std()**2 + nonchar_vals.std()**2) / 2)
    cohens_d     = (char_vals.mean() - nonchar_vals.mean()) / pooled_std if pooled_std > 0 else np.nan
    comp_stats.append({
        'category':        model_label,
        'char_mean':       char_vals.mean(),
        'nonchar_mean':    nonchar_vals.mean(),
        'char_std':        char_vals.std(),
        'nonchar_std':     nonchar_vals.std(),
        'char_sem':        char_vals.std() / np.sqrt(len(char_vals)),
        'nonchar_sem':     nonchar_vals.std() / np.sqrt(len(nonchar_vals)),
        'z_stat':          stat,
        'cohens_d':        cohens_d,
        'rank_biserial_r': stat / np.sqrt(len(char_vals) + len(nonchar_vals)),
        'p_raw':           pval,
    })

comp_stats_df = pd.DataFrame(comp_stats)
_, comp_stats_df['p_fdr'], _, _ = multipletests(comp_stats_df['p_raw'], method='fdr_bh')
comp_stats_df['sig'] = comp_stats_df['p_fdr'] < 0.05

paired_stats = {}
for group_label, frame in [('Character aha', df_char), ('Non-character aha', df_nonchar)]:
    paired = frame[['person_prop', 'situation_prop']].dropna()
    res = wilcoxon(paired['person_prop'], paired['situation_prop'], method='approx')
    n = len(paired)
    paired_stats[group_label] = {
        'stat': res.statistic, 'z': res.zstatistic,
        'r': res.zstatistic / np.sqrt(n), 'n': n, 'p_raw': res.pvalue,
    }

# --- text_semantic_categories.png ---
PERSON_ORDER    = ['Dispositional\ntrait', 'Individual focus\n(he/she)', 'Mental state\n(ToM)']
SITUATION_ORDER = ['Social\nrelationship', 'Group focus\n(they/their)',
                   'Event/action\n(situation)', 'Causal\nlanguage']
PLOT_ORDER  = PERSON_ORDER + SITUATION_ORDER
n_person    = len(PERSON_ORDER)
n_situation = len(SITUATION_ORDER)

res_plot = res_df.set_index('category').loc[PLOT_ORDER].reset_index()
scale, height = 100, 0.32
y = np.arange(len(res_plot))

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.axhspan(-0.5, n_person - 0.5, color='#EEF4FB', zorder=0)
ax.axhspan(n_person - 0.5, n_person + n_situation - 0.5 + 0.5, color='#FDF0ED', zorder=0)
ax.barh(y + height / 2, res_plot['char_mean'] * scale,
        height, color='#2166ac', alpha=0.9, label='Character aha', zorder=2)
ax.barh(y - height / 2, res_plot['nonchar_mean'] * scale,
        height, color='#d6604d', alpha=0.9, label='Non-character aha', zorder=2)

x_max = res_plot[['char_mean', 'nonchar_mean']].max().max() * scale
for i, row in res_plot.iterrows():
    x_tip  = max(row['char_mean'], row['nonchar_mean']) * scale + x_max * 0.03
    marker = ('***' if row['p_fdr'] < 0.001 else '**' if row['p_fdr'] < 0.01
              else '*' if row['p_fdr'] < 0.05 else 'n.s.')
    ax.text(x_tip, i, marker, ha='left', va='center', fontsize=10,
            color='#222222' if row['sig'] else '#AAAAAA',
            fontweight='bold' if row['sig'] else 'normal')

ax.axhline(n_person - 0.5, color='#AAAAAA', linewidth=0.8, linestyle='--', zorder=1)
ax.text(1.02, (n_person - 1) / 2, 'Person\nModel',
        transform=ax.get_yaxis_transform(),
        ha='left', va='center', fontsize=10, fontweight='bold', color='#154360')
ax.text(1.02, n_person + (n_situation - 1) / 2, 'Situation\nModel',
        transform=ax.get_yaxis_transform(),
        ha='left', va='center', fontsize=10, fontweight='bold', color='#641E16')

clean_labels = [c.replace('\n', ' ') for c in res_plot['category']]
ax.set_yticks(y)
ax.set_yticklabels(clean_labels, fontsize=11)
for i, lbl in enumerate(ax.get_yticklabels()):
    lbl.set_color('#154360' if i < n_person else '#641E16')

ax.invert_yaxis()
ax.set_xlabel('% of tokens in category', fontsize=12)
ax.set_title('Semantic Category Rates\nCharacter vs. Non-character Aha',
             fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.set_xlim(0, x_max * 1.45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
out = os.path.join(FIG_DIR, 'text_semantic_categories.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")

# --- text_person_situation_proportions.png ---
def p_to_marker(pval):
    if pval < 0.001: return '***'
    if pval < 0.01:  return '**'
    if pval < 0.05:  return '*'
    return 'n.s.'

def add_sig_bracket(ax, x1, x2, y, h, text, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, linewidth=1.2)
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=11, color=color)

x     = np.arange(2)
width = 0.34
char_means    = comp_stats_df['char_mean'].to_numpy() * 100
nonchar_means = comp_stats_df['nonchar_mean'].to_numpy() * 100
char_sems     = comp_stats_df['char_sem'].to_numpy() * 100
nonchar_sems  = comp_stats_df['nonchar_sem'].to_numpy() * 100

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.bar(x - width / 2, char_means,    width, yerr=char_sems,    capsize=5,
       color='#2166ac', alpha=0.9, label='Character aha')
ax.bar(x + width / 2, nonchar_means, width, yerr=nonchar_sems, capsize=5,
       color='#d6604d', alpha=0.9, label='Non-character aha')

for i, row in comp_stats_df.reset_index(drop=True).iterrows():
    y_pos = max(char_means[i] + char_sems[i], nonchar_means[i] + nonchar_sems[i]) + 1.5
    add_sig_bracket(ax, x[i] - width / 2, x[i] + width / 2, y_pos, 1.2,
                    p_to_marker(row['p_fdr']),
                    color='black' if row['sig'] else 'gray')

ax.set_ylabel('Composite proportion (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['Person model', 'Situation model'], fontsize=11)
ax.set_title('Person vs Situation Model Proportions in Aha Transcriptions\n'
             '(* p<.05, ** p<.01, *** p<.001, FDR-corrected)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, max(np.r_[char_means + char_sems, nonchar_means + nonchar_sems]) * 1.28)
ax.text(
    0.02, 0.98,
    f"Within character: person vs situation {p_to_marker(paired_stats['Character aha']['p_raw'])}\n"
    f"Within non-character: person vs situation {p_to_marker(paired_stats['Non-character aha']['p_raw'])}",
    transform=ax.transAxes, ha='left', va='top', fontsize=10,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.8', alpha=0.9),
)
fig.tight_layout()
out = os.path.join(FIG_DIR, 'text_person_situation_proportions.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# 2. USE embedding classifier — ROC curve
# ---------------------------------------------------------------------------
print("\n--- USE embeddings ---")

import tensorflow_hub as hub

print("  Loading USE model …")
embed_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

all_texts  = list(df_char['transcribe'].values) + list(df_nonchar['transcribe'].values)
labels     = np.array([1] * len(df_char) + [0] * len(df_nonchar))

print(f"  Encoding {len(all_texts)} transcriptions …")
embeddings = embed_model(all_texts).numpy()

pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, C=1.0, random_state=42)
)
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = cross_val_score(pipe, embeddings, labels, cv=cv, scoring='roc_auc')
print(f"  Logistic regression AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")

y_score      = cross_val_predict(pipe, embeddings, labels, cv=cv, method='predict_proba')[:, 1]
fpr, tpr, _  = roc_curve(labels, y_score)
roc_auc_val  = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_val:.3f})",
        linewidth=2, color='black')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('5-Fold Cross-Validated ROC Curve')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, 'roc_curve.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out}")

print("\nDone.")
