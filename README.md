# bot-or-human

This project is for the course TDT4310 Intelligent Text Analytics and Language Understanding spring 2020 at NTNU. The system predicts whether a tweets are authored by a bot or a human and compares the results of three well known and widely used classifiers, Support Vector Machine, Naïve Bayes and Random Forest.

## Dataset

The data used in this project is made available for the [PAN19 Author Profiling: Bots and Gender Profiling](https://pan.webis.de/clef19/pan19-web/author-profiling.html). You can request access at [Zenodo](https://zenodo.org/record/3692340).

## Prerequisites

- [Pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [NLTK](https://www.nltk.org/)

Packages can be installed using [pip](https://pypi.org/project/pip/).

Example:

```python
pip install scikit-learn
```

## Architecture
![Overall system architecture](/Architecture.png)

`preprocessing/`contains code for preprocsessing of the raw data. The models are written in `models.py`, where scikit-learn models are encapsulated providing potensial futher developend extending the system. `main.py` consists the main method and run the whole system by navigating to the root folder and execute following command:

```python
python main.py
```

## Configuration

The project is configured using the `config.py` file and enable step by step development creating checkpoints. Decreasing run time aspreprocessing and training can be saved, thus avoiding tedious latency. In the `config.py` file, set `read_and_save_raw_data_as_dataframe`, `load_preprocessed_dataframe`, `load_preprocessed_dataframe_of_test_data` and `load_trained_model` to `False` in order to do all processing from scratch.

## Results

The classification results for all the classifiers are printed to the console and confusion matrices are saved in `.png`-format to `results/`. Results for the project show that the SVM classifier and the Random Forest classifier perform clearly better than the Naïve Bayes classifier.  The Random Forest classifier scores highest on both the accuracy and the F1-score, with the SVM classifier scoring 2-3 percentage points lower on those metrics.  It seems that the Naïve Bayes classifier especially struggles in recalling all bot samples. 

## Acknowledgements
* Alex-just on GitHub for [this](https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1) snippet on how to strip emojis from a string. 
* MaxU on StackOverflow for answer on [this](https://stackoverflow.com/questions/43577590/adding-sparse-matrix-from-countvectorizer-into-dataframe-with-complimentary-info) question on how to add feature matrix to the dataframe.
* Alexandre Wrg for providing code on textual features in [this](https://towardsdatascience.com/how-i-improved-my-text-classification-model-with-feature-engineering-98fbe6c13ef3) blogpost. 
* Paolo Rosso and Francisco Rangel for assistance in accessing the dataset. 
