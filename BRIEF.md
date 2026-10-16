# Hackathon brief (official text)

Copied from the organizers' announcement. The data and template scripts haven't arrived yet.

> Electroencephalogram (EEG) data is a relatively inexpensive biological signal that can be easily collected using simple and non-invasive procedures. Despite a relatively low signal-to-noise ratio, EEG has proven useful for a wide range of research and real-world applications. This Hackathon will explore the use of state-of-the-art machine learning approaches (such as dimensionality reduction, video foundational models, and world models) to perform a simple for of "mind reading". Participants will work with EEG data recordings collected while viewing short three seconds videos. The goal is to be able to tell which video a participant was viewing based solely on their EEG activity. More specifically, given an EEG recording and a set of candidate videos, participants will develop a model that identifies which video most likely elicited the recorded brain activity. Beyond predictive performance, the challenge aims to encourage the integration of machine learning expertise and cognitive-psychology domain knowledge. We are particularly interested in approaches that are not only accurate, but can also provide scientifically meaningful insights into the similarities and differences between how biological and artificial systems process visual stimuli. To motivate scientifically grounded and domain-appropriate reasoning instead of over-engineering and dataset hacking, the challenge will comprise two datasets: one full training dataset with 4 subjects for model selection and fine-tuning, and a second test-set of 2 partially-held-out subjects to evaluate the generalizability and robustness of the solutions. All participating teams will be required to share the training pipelines and model weights to enable the reproduction and assessment of the proposed approaches. The solutions are required to be written in Python 3.11. Template scripts for loading and processing the EEG data, as well as training baseline models will be provided by the organizers.

## What it means for us

| The brief says | Our plan |
| --- | --- |
| Given EEG + candidate videos, pick the one that elicited it | EEG → predicted video vector → rank candidates by cosine similarity |
| Dimensionality reduction | PCA (and later CCA / spatial filters) on the EEG side |
| Video foundational models | CLIP, VideoMAE on the video side |
| World models | V-JEPA 2: video vectors and a "surprise over time" curve |
| Insights into biological vs artificial vision | Brain timing vs model depth, which electrodes carry the information, and whether brain and model confuse the same videos (see `WORK_SPLIT.md`) |
| 4 training subjects, 2 partially-held-out test subjects | Always validate leave-one-subject-out. "Partially" probably means a little data from the new people for calibration. |
| Share pipelines and weights, Python 3.11, templates provided | `requirements.txt`, saved weights, one `run.py`. The organizers' template layout wins once it arrives. |

**Open question for the template:** are the test candidates the same videos as in training, or new ones? With the same videos, a plain classifier (EEG → video ID) also works. With new videos, only the vector route works, because any new video can be turned into a vector.
