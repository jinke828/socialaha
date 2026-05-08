"""
step10_impression_text_analysis.py

Analyses
--------
1. USE embedding UMAP — impressions coloured by character.
2. Person model vs Situation model — stacked bar + line plot over runs.

Input
-----
  socialaha-collab/socialaha-fMRI/socialaha_transcribe_rhea_char_updated_clean_jin.csv

Outputs
-------
  results/figures/imp_umap_character.png
  results/figures/imp_person_vs_situation.png
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import umap

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = '/Users/jinke/Desktop/socialaha_github'
TRANSCRIBE   = os.path.join(BASE_DIR, 'socialaha-collab', 'socialaha-fMRI',
                            'socialaha_transcribe_rhea_char_updated_clean_jin.csv')
FIG_DIR      = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CHAR_NAMES   = {1: 'Jack', 2: 'Kate', 3: 'Randall', 4: 'Kevin'}
CHAR_COLORS  = {1: '#4393c3', 2: '#d6604d', 3: '#74c476', 4: '#fd8d3c'}
RUNS         = list(range(1, 11))

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
        'bitter', 'guilty', 'ashamed', 'strong', 'weak', 'depressed', 'anxious', 'upset', 'angry', 'frustrated', 'passionate', 'devoted',
        'dedicated', 'driven', 'identity', 'personality', 'character', 'nature'
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

PERSON_CATS    = ['Dispositional\ntrait', 'Mental state\n(ToM)',
                  'Individual focus\n(he/she)']
SITUATION_CATS = ['Social\nrelationship', 'Group focus\n(they/their)', 'Causal\nlanguage',
                  'Event/action\n(situation)']

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
VALID_SUBS = set()
for grp in [[1001,1005,1008,1011,1014,1017,1020,1023,1026,1029,1033,1039],
            [2006,2009,2012,2015,2018,2021,2024,2027,2034,2038,2040],
            [3004,3007,3013,3016,3019,3022,3025,3031,3037,3041]]:
    VALID_SUBS.update(grp)

df = pd.read_csv(TRANSCRIBE)
df = df[df['subject'].isin(VALID_SUBS)].dropna(subset=['transcribe']).copy()
df['char_name'] = df['character'].map(CHAR_NAMES)
print(f"Loaded {len(df)} impressions | {df['subject'].nunique()} subjects | "
      f"{df['run'].nunique()} runs | {df['character'].nunique()} characters")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tokenise(text):
    return re.findall(r"[a-z']+", str(text).lower())

def category_rate(text, word_set):
    tokens = tokenise(text)
    return sum(1 for t in tokens if t in word_set) / len(tokens) if tokens else np.nan

# ---------------------------------------------------------------------------
# Compute per-impression category rates
# ---------------------------------------------------------------------------
for cat, words in CATEGORIES.items():
    df[cat] = df['transcribe'].apply(lambda t: category_rate(t, words))

# ---------------------------------------------------------------------------
# 1. USE embeddings + UMAP — coloured by character
# ---------------------------------------------------------------------------
print("Loading USE model …")
embed_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

print(f"  Encoding {len(df)} impressions …")
embeddings = embed_model(df['transcribe'].tolist()).numpy()   # (N, 512)

print("  Running UMAP …")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.15)
emb2d = reducer.fit_transform(embeddings)

df['umap1'] = emb2d[:, 0]
df['umap2'] = emb2d[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
for char_id, char_name in CHAR_NAMES.items():
    mask = df['character'] == char_id
    ax.scatter(df.loc[mask, 'umap1'], df.loc[mask, 'umap2'],
               c=CHAR_COLORS[char_id], alpha=0.45, s=18, label=char_name, linewidths=0)
ax.legend(fontsize=11)
ax.set_title('UMAP of impression embeddings — by character', fontsize=13, fontweight='bold')
ax.set_xlabel('UMAP 1', fontsize=11)
ax.set_ylabel('UMAP 2', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'imp_umap_character.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: imp_umap_character.png")

# ---------------------------------------------------------------------------
# 2. Person model vs Situation model
# ---------------------------------------------------------------------------
print("Plotting person vs situation model …")

df['person_score']    = df[[c for c in PERSON_CATS    if c in df.columns]].mean(axis=1)
df['situation_score'] = df[[c for c in SITUATION_CATS if c in df.columns]].mean(axis=1)

total = df['person_score'] + df['situation_score']
df['person_prop']    = df['person_score']    / total.replace(0, np.nan)
df['situation_prop'] = df['situation_score'] / total.replace(0, np.nan)

char_ids    = sorted(CHAR_NAMES)
char_labels = [CHAR_NAMES[c] for c in char_ids]

person_means    = [df[df['character'] == c]['person_prop'].mean()    for c in char_ids]
situation_means = [df[df['character'] == c]['situation_prop'].mean() for c in char_ids]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(len(char_ids))
axes[0].bar(x, person_means,    label='Person model',    color='#2166ac', alpha=0.85)
axes[0].bar(x, situation_means, label='Situation model', color='#d6604d', alpha=0.85,
            bottom=person_means)
axes[0].set_xticks(x)
axes[0].set_xticklabels(char_labels, fontsize=12)
axes[0].set_ylabel('Proportion of model-related language', fontsize=11)
axes[0].set_title('Person vs Situation model\nby character (all runs)', fontsize=12, fontweight='bold')
axes[0].legend().set_visible(False)
axes[0].set_ylim(0, 1)

for xi, (pm, sm) in enumerate(zip(person_means, situation_means)):
    axes[0].text(xi, pm / 2,      f'{pm:.2f}', ha='center', va='center',
                 color='white', fontsize=11, fontweight='bold')
    axes[0].text(xi, pm + sm / 2, f'{sm:.2f}', ha='center', va='center',
                 color='white', fontsize=11, fontweight='bold')

for char_id, char_name in CHAR_NAMES.items():
    sub = df[df['character'] == char_id].groupby('run')[['person_prop', 'situation_prop']].mean()
    axes[1].plot(sub.index, sub['person_prop'],    color=CHAR_COLORS[char_id],
                 linestyle='-',  marker='o', linewidth=2, label=f'{char_name} (person)')
    axes[1].plot(sub.index, sub['situation_prop'], color=CHAR_COLORS[char_id],
                 linestyle='--', marker='s', linewidth=1.5, alpha=0.6,
                 label=f'{char_name} (situation)')

axes[1].set_xlabel('Run', fontsize=11)
axes[1].set_ylabel('Mean proportion', fontsize=11)
axes[1].set_title('Person (solid) vs Situation (dashed)\nmodel language over runs', fontsize=12, fontweight='bold')
axes[1].set_xticks(RUNS)
axes[1].legend().set_visible(False)

fig.suptitle('Person model vs Situation model in character impressions',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'imp_person_vs_situation.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: imp_person_vs_situation.png")

print("\nDone.")
