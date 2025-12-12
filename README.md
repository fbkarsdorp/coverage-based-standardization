# Coverage-Based Standardization for Cultural Data: Correcting Collection Bias in Comparative Research

Code and data accompanying the paper "Coverage-Based Standardization for Cultural Data: Correcting Collection Bias in Comparative Research". Below we describe the main purpose of each file and link them to the images in the manuscript. 

# File structure
## Notebooks
- Figure 1: notebooks/example-figure.ipynb
- Main simulation experiments (Figures 2, 3, 4, 5, S1, S6): notebooks/simulation.ipynb
- Coverage threshold analysis (Figure S5): notebooks/coverage_threshold.ipynb
- Population size regression analysis (Figure 6): notebooks/song-coverage.ipynb
- Diachronic regression analysis (Figure 7): notebooks/song-coverage-diachronic.ipynb
- Coverage accuracy analysis (Figures S2, S3, S4), coverage_accuracy.ipynb
- Construct Amsterdam song data: notebooks/make-data-amsterdam.ipynb
- Construct song data: notebooks/make-data-17th-century.ipynb

## Supporting Source Code
- Implementation of collection biases: src/simulation.py
- Wright-Fisher model implementation: src/model.py
- Main experiment functions: src/experiment.py
- Make data for coverage accuracy and ratio accuracy analysis: src/coverage_threshold.py
- Some utility functions: src/utils.py

## Data (empirical and generated)
- Dutch Song Database melody locations counts: data/dsd-melody-place-counts-census.csv
- Stratified sampling data: data/exp_stratified-new.pkl
- Productivity paradox sampling data: data/exp_productivity-new.pkl
- Stromer's Riddle paradox sampling data: data/exp_stromer-new.pkl
- Wright-Fisher populations: data/populations_list.pkl
- True diversity numbers from Wright-Fisher populations: data/S_true.npy
- Population diversity comparisons:
    - data/threshold_populations1-3.pkl
    - data/threshold_populations1-4.pkl
    - data/threshold_populations1-2.pkl
- Census data: data/population-estimates.csv
- Dutch Song Database melody counts Amsterdam: data/dsd-amsterdam-melody-decade-counts.csv
- Standardized Dutch place names: data/placenames.json

# Installation
1. `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. navigate to project directory
3. `uv sync`
4. `cd notebooks`
5. `uv run jupyter notebook simulation.ipynb`
