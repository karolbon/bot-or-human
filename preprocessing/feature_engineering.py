from read_data import load_dataframe, save_dataframe
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def textual_features(df):
    # Using code snippet from Alexandre Wrg on https://towardsdatascience.com/how-i-improved-my-text-classification-model-with-feature-engineering-98fbe6c13ef3
    # as starting point for textual features.
    df['word_count'] = df['Tweets'].apply(lambda x: len(x.split()))
    df['char_count'] = df['Tweets'].apply(lambda x: len(x.replace(" ", "")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['total_length'] = df['Tweets'].apply(len)
    df['capitals'] = df['Tweets'].apply(
        lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(
        row['capitals'])/float(row['total_length']), axis=1)
    df['num_exclamation_marks'] = df['Tweets'].apply(lambda x: x.count('!'))
    df['num_question_marks'] = df['Tweets'].apply(lambda x: x.count('?'))
    df['num_punctuation'] = df['Tweets'].apply(
        lambda x: sum(x.count(w) for w in '.,;:'))
    df['num_symbols'] = df['Tweets'].apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    df['num_unique_words'] = df['Tweets'].apply(
        lambda x: len(set(w for w in x.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['word_count']
    df["word_unique_percent"] = df["num_unique_words"]*100/df['word_count']
    return df


def feature_engineering(df):
    tfidf = TfidfVectorizer(stop_words='english', min_df=0.01)
    X = tfidf.fit_transform(df['Tweets'])

    # Code snippet from https://stackoverflow.com/questions/43577590/adding-sparse-matrix-from-countvectorizer-into-dataframe-with-complimentary-info
    for i, col in enumerate(tfidf.get_feature_names()):
        df[col] = pd.SparseSeries(X[:, i].toarray().ravel(), fill_value=0)

    df = textual_features(df)

    return df


df = load_dataframe('training_df')
with_features = feature_engineering(df)
save_dataframe(with_features, 'training_df_with_basic_features')
