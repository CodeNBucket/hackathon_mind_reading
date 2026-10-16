# Insights plan: cognitive psychology and brain vs AI

The brief rewards "scientifically meaningful insights into the similarities and differences between how biological and artificial systems process visual stimuli". Psychology enters in three places: **before** the model (design), **inside** it (features), and **after** it (the brain vs AI comparisons).

## 1. Before: design choices from what we know about vision (EEG side)

- **Where:** back of the head = visual cortex. Side-back electrodes (P7/P8) = face and object areas.
- **When:** ~100 ms (P1/N1) light, edges, motion. ~170 ms (N170) faces. ~300 ms+ (P300) attention, surprise, meaning.
- **Rhythms:** alpha (8–13 Hz) drops when someone attends visually.

Slide line: "our features follow the known timeline of human vision."

## 2. Inside: human-readable video features (video side)

Next to CLIP / VideoMAE / V-JEPA 2 vectors, describe each video with features that each have a known brain response:
brightness, amount of motion, scene-cut times, whether a face is present, and the world-model surprise curve.

Each one becomes a checkable hypothesis, e.g. "does the N170 appear only for face videos?", "does a scene cut trigger a response ~100 ms later?"

## 3. After: brain vs AI comparisons (from the trained models)

| Analysis | Question | Who |
| --- | --- | --- |
| **Decoding over time** | Accuracy per 50 ms EEG window: when does the brain "know" what it saw? | EEG |
| **Head map** | Which electrodes carry the information? | EEG |
| **Time vs depth (RSA)** | Does early EEG match early model layers (edges) and later EEG match later layers (objects)? Same order = shared hierarchy. | EEG + video |
| **Which model is most brain-like (RSA)** | Correlate the brain's 20×20 video-similarity table with each model's table (simple features, CLIP, VideoMAE, V-JEPA 2). | EEG + video |
| **Same confusions?** | Does the brain-based decoder mix up the same videos the models find similar? | EEG + video |
| **Surprise** | Does the EEG react at the moments V-JEPA 2 is surprised (predictive coding)? | EEG + video |

## Differences worth showing

- **Repetition suppression:** the brain responds less on the 2nd viewing of a video; models give identical output every time. Measured directly from the repetitions.
- **Individual differences:** every brain differs, the model is the same for everyone. Shows in the accuracy drop on unseen people.
- **Motion:** if the brain leans on motion that CLIP (still frames) can't see, VideoMAE / V-JEPA 2 should be more brain-like.

## When

Add the design choices (part 1) to the baseline right away, since they're cheap. Add the simple video features with EEG steps 2–3. Run the part 3 analyses on the final models in the last stretch. They reuse what we've built, so the cost is mostly plotting.
