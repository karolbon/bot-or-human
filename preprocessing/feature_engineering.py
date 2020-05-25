from preprocessing.read_data import load_dataframe, save_dataframe
import pandas as pd
import pickle
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
# https://en.wikipedia.org/wiki/Unicode_block
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+"
)


def strip_emoji(text):
    # http://stackoverflow.com/a/13752628/6762004
    return re.sub(EMOJI_PATTERN, '<EMOJI>', text)


def remove_stopwords(tweet):
    stopwords = nltk.corpus.stopwords.words('english')
    # NLTK chaper 4.1
    tweet_as_list = tweet.split(' ')
    without_stopwords = [
        w for w in tweet_as_list if w.lower() not in stopwords]
    return ' '.join(without_stopwords)


def process_tweet(tweet):
    processed = tweet.lower()
    processed = re.sub('https([^\s]+)', "<URL>", processed)
    processed = re.sub('rt @([^\s]+)', '<RT>', processed)
    processed = re.sub('@([^\s]+)', "<UserMention>", processed)
    processed = re.sub('#([^\s]+)', '<HashTag>', processed)
    processed = strip_emoji(processed)
    processed = processed.replace('\n', ' ')
    processed = remove_stopwords(processed)
    processed = processed.replace("[^0-9a-zA-Z:,.?!]+", "")
    return processed


def textual_features(df):
    # Using code snippet from Alexandre Wrg on https://towardsdatascience.com/how-i-improved-my-text-classification-model-with-feature-engineering-98fbe6c13ef3
    # as starting point for textual features.
    df['word_count'] = df['Tweets'].apply(lambda x: len(x.split()))
    df['char_count'] = df['Tweets'].apply(lambda x: len(x.replace(" ", "")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    df['total_length'] = df['Tweets'].apply(len)
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


def tfidf_features(df, is_training):

    if is_training:
        tfidf = TfidfVectorizer(stop_words='english', min_df=0.01)

        X = tfidf.fit_transform(df['Tweets'])

        # Code snippet from https://stackoverflow.com/questions/29788047/keep-tfidf-result-for-predicting-new-content-using-scikit-for-python
        pickle.dump(tfidf, open("tfidf1.pkl", "wb"))

    else:
        # Is testing data, have to load Tfidf corpus from what is generated from the training data
        # in order to get the same features.
        tfidf = pickle.load(open("tfidf1.pkl", 'rb'))
        print("corpus:", tfidf, type(tfidf))
        X = tfidf.transform(df['Tweets'])

    # Code snippet from https://stackoverflow.com/questions/43577590/adding-sparse-matrix-from-countvectorizer-into-dataframe-with-complimentary-info
    for i, col in enumerate(tfidf.get_feature_names()):
        df[col] = pd.SparseSeries(X[:, i].toarray().ravel(), fill_value=0)

    return df


def feature_engineering(df, is_training):
    # Lowercase, replace URLs, etc.
    df['Tweets'] = df['Tweets'].apply(lambda x: process_tweet(x))
    df_tfidf = tfidf_features(df, is_training)
    df_features = textual_features(df_tfidf)
    return df_features
