"""
step11_aha_impression_relation.py

Tests whether character vs non-character aha counts relate differentially to
person-model vs situation-model aspects of reported impressions.

Outputs
-------
  results/figures/aha_imp_combined.png
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, zscore
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR      = '/Users/jinke/Desktop/socialaha_github'
ANNOT_PATH    = os.path.join(BASE_DIR, 'data', 'beh', 'annotations', 'ahaannot_all.xlsx')
TRANSCRIBE    = os.path.join(BASE_DIR, 'socialaha-collab', 'socialaha-fMRI',
                             'socialaha_transcribe_rhea_char_updated_clean_jin.csv')
FIG_DIR       = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

CHAR_NAMES  = {1: 'Jack', 2: 'Kate', 3: 'Randall', 4: 'Kevin'}
VALID_SUBS  = set(
    [1001,1005,1008,1011,1014,1017,1020,1023,1026,1029,1033,1039] +
    [2006,2009,2012,2015,2018,2021,2024,2027,2034,2038,2040] +
    [3004,3007,3013,3016,3019,3022,3025,3031,3037,3041]
)

# ---------------------------------------------------------------------------
# Semantic dictionaries
# ---------------------------------------------------------------------------
CATS = {
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
}

PERSON_CATS    = ['Dispositional\ntrait', 'Mental state\n(ToM)',
                  'Individual focus\n(he/she)']
SITUATION_CATS = ['Social\nrelationship', 'Group focus\n(they/their)', 'Causal\nlanguage',
                  'Event/action\n(situation)']

def tokenise(text):
    return re.findall(r"[a-z']+", str(text).lower())

def cat_rate(text, word_set):
    tokens = tokenise(text)
    if not tokens:
        return np.nan
    return sum(1 for t in tokens if t in word_set) / len(tokens)

# ---------------------------------------------------------------------------
# Load aha counts per subject × run
# ---------------------------------------------------------------------------
print("Loading aha annotations …")
df_a = pd.read_excel(ANNOT_PATH)
df_a['character_all'] = df_a[['character_rater1','character_rater2',
                               'character_rater3']].sum(axis=1)
aha = (df_a.groupby(['subject', 'run'])
           .apply(lambda g: pd.Series({
               'n_char_aha':    int((g['character_all'] >= 2).sum()),
               'n_nonchar_aha': int((g['character_all'] == 0).sum()),
               'n_total_aha':   len(g),
           }))
           .reset_index())
aha = aha[aha['subject'].isin(VALID_SUBS)]

# ---------------------------------------------------------------------------
# Load impressions, compute person/situation composites, average across chars
# ---------------------------------------------------------------------------
print("Loading impression transcriptions …")
df_i = pd.read_csv(TRANSCRIBE)
df_i = df_i[df_i['subject'].isin(VALID_SUBS)].dropna(subset=['transcribe']).copy()

for cat, words in CATS.items():
    df_i[cat] = df_i['transcribe'].apply(lambda t: cat_rate(t, words))

df_i['person_score']    = df_i[PERSON_CATS].mean(axis=1)
df_i['situation_score'] = df_i[SITUATION_CATS].mean(axis=1)
total = df_i['person_score'] + df_i['situation_score']
df_i['person_prop']    = df_i['person_score']    / total.replace(0, np.nan)
df_i['situation_prop'] = df_i['situation_score'] / total.replace(0, np.nan)

imp_avg = (df_i.groupby(['subject', 'run'])[['person_prop', 'situation_prop']]
               .mean()
               .reset_index())

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
df = aha.merge(imp_avg, on=['subject', 'run'], how='inner')
print(f"Analysis dataset: {len(df)} subject × run observations, "
      f"{df['subject'].nunique()} subjects")

# ---------------------------------------------------------------------------
# Within-subject z-score (removes between-subject variance)
# ---------------------------------------------------------------------------
for col in ['n_char_aha', 'n_nonchar_aha', 'person_prop', 'situation_prop']:
    df[col + '_z'] = df.groupby('subject')[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

def spearman_z(x_col, y_col):
    xz, yz = x_col + '_z', y_col + '_z'
    valid = df[[xz, yz]].dropna()
    r, p = spearmanr(valid[xz], valid[yz])
    return r, p

# ---------------------------------------------------------------------------
# Combined scatter: char aha → person model, nonchar aha → situation model
# ---------------------------------------------------------------------------
print("\n--- Combined binned scatter plot ---")

fig, ax = plt.subplots(figsize=(6.5, 5.5))

pairs = [
    ('n_char_aha',    'Character aha count',     'person_prop',    'Person model proportion',       '#2166ac', 'Character aha → Person model'),
    ('n_nonchar_aha', 'Non-character aha count', 'situation_prop', 'Social-event model proportion', '#d6604d', 'Non-character aha → Social-event model'),
]

for aha_col, aha_label, imp_col, imp_label, color, legend_label in pairs:
    sub = df[[aha_col, imp_col]].dropna()

    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(sub))
    ax.scatter(sub[aha_col] + jitter, sub[imp_col],
               alpha=0.22, s=14, color=color, linewidths=0)

    binned = sub.groupby(aha_col)[imp_col].agg(['mean', 'sem']).reset_index()
    binned = binned[binned[aha_col] <= np.percentile(sub[aha_col], 95)]
    ax.plot(binned[aha_col], binned['mean'],
            color=color, linewidth=2.5, marker='o', markersize=5, zorder=5,
            label=legend_label)
    ax.fill_between(binned[aha_col],
                    binned['mean'] - binned['sem'],
                    binned['mean'] + binned['sem'],
                    alpha=0.18, color=color)

r1, p1 = spearman_z('n_char_aha', 'person_prop')
r2, p2 = spearman_z('n_nonchar_aha', 'situation_prop')
sig1 = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'n.s.'
sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'n.s.'
print(f"  Character aha → Person model:          r={r1:.3f}  p={p1:.4f}  {sig1}")
print(f"  Non-character aha → Social-event model: r={r2:.3f}  p={p2:.4f}  {sig2}")

# Mixed-effects model: nested random effects (run within subject within group)
df['group'] = (df['subject'] // 1000).astype(str)

def fit_adjusted_mixed_model(x_col, y_col):
    """
    Nested mixed model: (1 | group / subject / run).
    x and y are z-scored; the fixed-effect coefficient is a standardised beta.
    """
    sub = df[[x_col, y_col, 'run', 'group', 'subject']].dropna().copy()
    sub = sub.rename(columns={x_col: 'x', y_col: 'y'})
    sub['group'] = sub['group'].astype(str)
    sub['x_z'] = zscore(sub['x'], nan_policy='omit')
    sub['y_z'] = zscore(sub['y'], nan_policy='omit')
    sub = sub.dropna(subset=['x_z', 'y_z']).copy()

    sub['_group_sub']     = sub['group'] + ':' + sub['subject'].astype(str)
    sub['_group_sub_run'] = sub['_group_sub'] + ':' + sub['run'].astype(str)

    vcf = {
        'subject': '0 + C(_group_sub)',
        'run':     '0 + C(_group_sub_run)',
    }
    model = smf.mixedlm("y_z ~ x_z", data=sub, groups=sub['group'], vc_formula=vcf)
    try:
        fit = model.fit(reml=False, method='lbfgs', disp=False)
    except Exception:
        fit = model.fit(reml=False, method='powell', disp=False)
    return sub, fit

print("\n--- Mixed-effects model (nested: run within subject within group) ---")
for x_col, y_col, label in [
    ('n_char_aha',    'person_prop',    'Character aha → Person model'),
    ('n_char_aha',    'situation_prop', 'Character aha → Social-event model'),
    ('n_nonchar_aha', 'person_prop',    'Non-character aha → Person model'),
    ('n_nonchar_aha', 'situation_prop', 'Non-character aha → Social-event model'),
]:
    _, fit = fit_adjusted_mixed_model(x_col, y_col)
    beta = fit.fe_params['x_z']
    se   = fit.bse['x_z']
    pval = fit.pvalues['x_z']
    sig  = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
    print(f"  {label}: beta={beta:+.4f}  SE={se:.4f}  p={pval:.4f}  {sig}")

ax.set_xlabel('Aha count', fontsize=10)
ax.set_ylabel('Impression model proportion', fontsize=10)
ax.set_title('Aha count vs impression model proportions', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'aha_imp_combined.png'), dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved: aha_imp_combined.png")

print("\nDone.")
