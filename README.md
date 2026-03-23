# Socialaha

**Jin Ke, Rhea Madhogarhia, Marvin M. Chun, Monica D. Rosenberg, Yuan Chang Leong, and Hayoung Song (2026). Neural dynamics of social impression updating during narrative comprehension. _bioRxiv_**  

Correspondence to jin.ke@yale.edu and hayoung@wustl.edu.
please feel free to reach out with questions or inquiries.

**SocialAha Dataset**

36 participants watched a temporally scrambled version of the first episode of NBC's _This Is Us_ while undergoing fMRI. Over 10 runs, participants viewed the movies while indicating and explaining moments of insight, and verbally reported impressions of the characters.

* The raw and preprocessed MRI data are publicly available on OpenNeuro. Link: https://openneuro.org/datasets/ds005658
* The associated behavioral data are available here: https://github.com/jinke828/socialaha/tree/main/data/beh
* Intermediate brain data are available here for convenience in replicating the findings: https://github.com/jinke828/socialaha/tree/main/data/brain

This dataset was initially used in our earlier publication (Song et al., Nature Communications, 2026): https://www.biorxiv.org/content/biorxiv/early/2025/03/13/2025.03.12.642853.full.pdf
   
**Code**

We provide a step-by-step, very detailed instructions to run the scripts that replicate the key findings of this paper: 
[Code Guide](https://github.com/jinke828/socialaha/blob/main/Code%20guide_JK.pdf); Software's license: Apache License 2.0

* `.step01_load-brain.ipynb` — Loads fMRI data, extracts ROI time series, applies preprocessing, and segments signals into events and impression periods.
* `.step02_count_aha.ipynb` — Counts and summarizes the distribution of “aha” moments across participants.
* `.step03_impression-updates.py` — Quantifies how character impressions evolve over time using text embeddings and similarity analyses.
* `.step03_plot_impression_updates.R` — Models and visualizes how impression similarity changes with temporal distance.
* `.step04_IS-RSA` — Tests whether neural synchrony predicts shared impressions using IS-RSA with bootstrapping.
* `.step05a_IS-RSA-mediation.ipynb` — Constructs dataset linking pre/post impressions and neural synchrony for mediation analysis.
* `.step05b_mediation.R` — Runs ROI-wise mediation to test whether neural synchrony mediates impression updating.
* `.step05c_mediation_rSTS.R` — Performs high-precision mediation analysis specifically for right STS.
* `.step06_compute_mtm_neural-pattern-shift.ipynb` — Computes moment-to-moment neural pattern shifts across time (TR-by-TR dissimilarity).
* `.step07_neural-pattern-shift--social_insight.py` — Tests whether neural shifts increase around character-related “aha” moments.
* `.step08_neural_pattern_shift-impression_update.py` — Links neural shifts at “aha” moments to the magnitude of impression updates.


**Data Flow**
1. Preprocessed fMRI (`.nii.gz`) is loaded from `./data/brain/derivatives/` (sourced from OpenNeuro ds005658)
2. Event timing files (`.tsv`, BIDS format) from `./data/brain/events/` define movie-watching vs. impression-rating periods
3. Time series are extracted from 116 brain ROIs (100 cortical Schaefer + 16 subcortical Tian parcels)
4. Intermediate results are saved as pickle files in `./data/brain/loaded_BOLD/`, `./data/brain/similarity/`, and `./data/brain/pattern_shift/`
5. Final figures output to `./results/figures/`
