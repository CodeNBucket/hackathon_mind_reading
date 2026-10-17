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

## Round 8: checks on the final model (17 Oct)
| check | top-1 | top-5 | median rank |
|---|---|---|---|
| EEG sampling rate 25 / 50 / 100 / 200 / 250 / 500 Hz | 5.6–6.4% | 20.7–20.9% | 20.6–21.4 |
| low-level window 0–1.0 / 0–1.2 / 0–3.0 s | 6.1% | 20.8–20.9% | 20.6–21.1 |
| PCA size VideoMAE / AST 64–512, no PCA; PCA on CLIP | 5.8–6.4% | 20.7–20.9% | 20.3–21.3 |

No differences beyond noise: 100 Hz, 0–1.2 s low-level and PCA 128 are kept.

## Round 9: using the EEG after the first second
| change | top-1 | top-5 | median rank |
|---|---|---|---|
| EEG window 0–1 s (current) | 6.1% | 20.9% | 20.7 |
| EEG window 0–1.5 / 0–2 / 0–3 s | 5.4 / 4.9 / 3.8% | 19.2 / 16.8 / 14.4% | 22.6 / 24.9 / 29.2 |
| only 1–2 s / only 2–3 s (alone) | 1.3 / 1.2% | 6.7 / 5.6% | 48.6 / 53.7 |
| time-resolved model (TRF, lags 0–0.6 s; low-level + CLIP frames [+ windowed audio]) on 1–3 s, alone | 1.1–1.4% | 6.3–6.9% | 52–55 |
| + TRF combined with the current model | 6.1–6.3% | 20.1–20.8% | 21.2–22.1 |
| 1–3 s wave averaged in 0.25 / 0.5 s bins, combined | 6.0% | 20.1–20.5% | 21.1 |
| **rhythm power (log band power, 5 bands, 0.5 s bins, 0–3 s), combined at weight 0.25** | **7.7%** | **24.4%** | **18.0** |
| rhythm power other variants (1–3 s only, 4 bands, 1 s bins, weights 0.5–1.0) | 5.8–7.7% | 19.9–24.0% | 17.8–23.0 |

After about 1 s the exact wave shape varies too much from viewing to viewing to be predicted (window, TRF and coarse-bin variants all fail). The slower brain "state", measured as the power of each rhythm, is predictable from the video and adds information. TRF (Crosse et al., 2016) is the standard method for continuous stimuli; here it stays weak.

## Milestone scores on the sub-01…04 test sets (102 candidates)
| model | top-1 | top-5 | median rank |
|---|---|---|---|
| organisers' baseline (AlexNet) | ~4% | 11.7% | |
| base: CLIP + per-subject ridge | 5.1% | 23.3% | 18.3 |
| + CAR, 3-model ensemble, audio | 7.1% | 25.7% | 15.3 |
| + whitening, low-level features | 14.0% | 37.0% | 10.1 |
| + per-feature ridge strength | 13.7% | 36.5% | 9.1 |
| + VideoMAE | 15.2% | 41.7% | 8.3 |
| + member weights 1 / 0.25 / 0.25 | 16.7% | 46.1% | 6.3 |
| **+ rhythm power, 0–3 s (final)** | **17.6%** | **45.6%** | **6.6** |
| shuffled-label control (final model) | 0.7% | 3.2% | |

On the test sets, rhythm power is neutral (+0.9 top-1, −0.5 top-5, within noise), although alone it scores 13.7% top-5 there (chance 4.9%). The test recordings are cleaner than the training recordings (test/train amplitude ratio ~0.65, consistent with averaging over more repetitions), so the wave model already dominates there. It was kept because the decision rule is the held-out training videos, where it adds +3.5 points.
