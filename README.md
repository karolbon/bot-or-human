# bot-or-human

This project is for the course TDT4310 Intelligent Text Analytics and Language Understanding spring 2020 at NTNU. The system predicts whether a tweets are authored by a bot or a human and compares the results of three well known and widely used classifiers, Support Vector Machine, Naïve Bayes and Random Forest.

## Dataset

The data used in this project is made available for the [PAN19 Author Profiling: Bots and Gender Profiling](https://pan.webis.de/clef19/pan19-web/author-profiling.html). You can request access at [Zenodo](https://zenodo.org/record/3692340).

## Prerequisites

- Pandas
- scikit-learn
- Pickle
- Matplotlib
- NLTK

Packages can be installed using [pip](https://pypi.org/project/pip/).

Example:

```python
pip install scikit-learn
```

## Architecture
![Overall system architecture](/Architecture.png)

## Configuration

The project is configured using the `config.py` file and enable step by step development creating checkpoints. Decreasing run time aspreprocessing and training can be saved, thus avoiding tedious latency. In the `config.py` file, set `read_and_save_raw_data_as_dataframe`, `load_preprocessed_dataframe`, `load_preprocessed_dataframe_of_test_data` and `load_trained_model` to `False` in order to do all processing from scratch.

## Results

The classification results for all the classifiers are printed to the console and confusion matrices are saved in `.png`-format to `results/`. Results for the project show that the SVM classifier and the Random Forest classifier perform clearly better than the Naïve Bayes classifier.  The Random Forest classifier scores highest on both the accuracy and the F1-score, with the SVM classifier scoring 2-3 percentage points lower on those metrics.  It seems that the Naïve Bayes classifier especially struggles in recalling all bot samples. 
