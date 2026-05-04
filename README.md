# ELLIIIT signal processing repo

This repository contains the post-processing code used to inspect the ELLIIIT acoustic and RF dataset.


## To start
On the top of the original project, RF part have some dependencies you can find in the requirements.txt

## RF signal prcessing
The RF code is in the [`post-processing/match_filter.py`](post-processing/match_filter.py)


## Impormtant for collabration
- for easy sharing, i put the venv file and irrelavant docs in the .gitignore file
- 


Key code paths remain in:

- `server/` for orchestration and control-plane logic
- `client/` for rover, RF, and auxiliary clients
- `acoustic/` for acoustic capture and processing
- `processing/dataset-download/` for published dataset download helpers
- `processing/parsing/` for RF and acoustic extraction/parsing scripts
- `processing/tutorials/` for xarray utilities and runnable notebook analysis
- `post-processing/` for user notebooks, scripts, figures, and follow-up analysis built on the dataset
