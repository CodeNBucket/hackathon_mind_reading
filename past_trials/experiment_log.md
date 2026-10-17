# Experiment log

Every change was judged on **held-out training videos**: 15% of the 1,000 training videos are hidden, the model is trained on the rest, and each hidden recording is ranked against the ~150 hidden videos (chance: top-1 0.7%, top-5 3.3%). From round 3 on, results are averaged over 3 different random hidden sets to reduce noise. The sub-01…04 **test sets** (102 candidates, chance 1.0% / 4.9%) were used only to report milestone scores, never to choose settings.

Scripts: `past_trials/exp_eeg.py`, `past_trials/exp_combo.py` (round 0 was run with the first version of the code, removed since; it is in git history at commit 8e5ae95).

## Round 0: base model (CLIP-B/32 card → ridge → predicted EEG, per-subject)
| setting | top-1 | top-5 |
|---|---|---|
| window 0–1 s (chosen) | 2.7% | 9.5% |
| window 0–0.6 / 0–2 / 0–3 s | | 8.6 / 8.1 / 6.8% |
| per-recording normalisation | | 7.8% |
| decoder direction (EEG → card) | | ~4.9% (chance) |

## Round 1: EEG preprocessing (CLIP-B/32, per-subject encoder)
| change | top-1 | top-5 |
|---|---|---|
| baseline | 2.7% | 9.5% |
| band-pass 0.1–30 Hz | 3.0% | 9.3% |
| low-pass 15 Hz | 2.7% | 10.0% |
| low-pass 8 Hz | 2.1% | 10.9% |
| **common average reference (CAR)** | 3.0% | **12.5%** |
| window 0.05–0.8 s | 2.7% | 9.5% |
| window 0–1.3 s | 2.4% | 9.4% |
| + model on all 6 subjects ("pooled") | 3.4% | 10.3% |
| + model on the grand-average EEG ("GA") | 3.1% | 9.9% |
| reliability-weighted correlation | 2.5% | 8.8% |

## Round 2: combining (CLIP-B/32)
| change | top-1 | top-5 |
|---|---|---|
| CAR | 3.0% | 12.5% |
| CAR + low-pass 12 Hz | 3.7% | 12.9% |
| CAR + band 0.5–8 Hz | 3.0% | 13.0% |
| CAR + subject + pooled | 3.8% | 12.8% |
| **CAR + subject + pooled + GA ensemble** | 3.8% | **13.7%** |
| ensemble + extra filtering | | 12.3–13.6% (no gain) |

## Round 3: sound (3 hidden sets averaged; encoder now only centres features so block weights work)
| features | top-1 | top-5 |
|---|---|---|
| CLIP-B/32 | 3.6% | 12.4% |
| audio alone (AST layer 4) | 1.6% | 8.0% |
| **CLIP + AST layer 4 (weight 0.5)** | 3.8% | **13.8%** |
| CLIP + AST layer 4 (weight 1.0) | 2.8% | 12.7% |

## Round 4: channel whitening (multivariate noise normalisation)
| change | top-1 | top-5 |
|---|---|---|
| CAR (reference) | 3.8% | 13.8% |
| CAR + whitening | 2.7–3.2% | 12.1–13.3% |
| whitening only, shrink 0.01 / 0.05 / 0.1 | | 13.9 / 15.0 / 15.3% |
| **whitening only, shrink 0.2–0.4** | 4.1–4.2% | **16.0–16.3%** |

## Round 5: more stimulus features (whitening 0.3, subject + pooled + GA)
| features | top-1 | top-5 |
|---|---|---|
| CLIP + AST4 | 4.3% | 16.2% |
| low-level time courses alone (brightness, contrast, motion, cuts, loudness, onsets) | 2.7% | 9.9% |
| **+ low-level (weight 0.6)** | 4.6% | **17.7%** |
| **+ separate ridge strength per EEG feature** | 5.1% | **19.3%** |
| VideoMAE alone | 4.8% | 16.0% |
| **+ VideoMAE layers 4/8/12 (weight 0.6)** | 5.2% | **20.9%** |
| VideoMAE layer 4 / 8 / 12 only, or other PCA sizes | | 18.7–20.8% (no gain) |

## Round 6: a bigger image model (CLIP ViT-L/14, 16 frames, final + layers 6/12/18/24)
| features | top-1 | top-5 |
|---|---|---|
| current best (CLIP-B/32 + AST4 + low-level + VideoMAE) | 5.2% | 20.9% |
| CLIP-L/14 embedding alone | 3.5% | 14.8% |
| CLIP-L/14 intermediate layers alone | 5.0% | 17.0% |
| swap CLIP-B/32 for CLIP-L/14 | 5.5% | 20.0% |
| best + CLIP-L/14 (weight 0.4 / 0.8, embedding or layers) | 5.1–5.5% | 20.3–20.7% |

No gain: a bigger image model does not explain the EEG better. Motion (VideoMAE), low-level timing and sound add more than a stronger object recogniser. Not used.

## Round 7: ensemble member weights (subject / pooled / grand average)
| weights | top-1 | top-5 | median rank |
|---|---|---|---|
| 1 / 0 / 0 (subject only) | 6.3% | 20.5% | 21.5 |
| **1 / 0.25 / 0.25 (chosen)** | 6.1% | 20.9% | 20.7 |
| 1 / 0.5 / 0.5 | 5.5% | 21.3% | 21.1 |
| 1 / 1 / 1 | 5.2% | 20.9% | 22.9 |
| 1 / 0 / 0.5 | 6.1% | 20.9% | 20.9 |
| 1 / 0.5 / 0 | 6.1% | 20.8% | 20.5 |

## Milestone scores on the sub-01…04 test sets (102 candidates)
| model | top-1 | top-5 | median rank |
|---|---|---|---|
| organisers' baseline (AlexNet) | ~4% | 11.7% | |
| base: CLIP + per-subject ridge | 5.1% | 23.3% | 18.3 |
| + CAR, 3-model ensemble, audio | 7.1% | 25.7% | 15.3 |
| + whitening, low-level features | 14.0% | 37.0% | 10.1 |
| + per-feature ridge strength | 13.7% | 36.5% | 9.1 |
| + VideoMAE | 15.2% | 41.7% | 8.3 |
| **+ member weights 1 / 0.25 / 0.25 (final)** | **16.7%** | **46.1%** | **6.3** |
| shuffled-label control (final model) | 0.7% | 2.7% | |
