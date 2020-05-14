from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessing.read_data import load_dataframe, save_dataframe


def use_SVC(data):
    y = df['Label']
    X = df.drop(columns=['Label', 'User_ID', 'Tweets'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.50, random_state=42)

    clf = make_pipeline(StandardScaler(), SVC())

    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    acc = accuracy_score(predicted, y_test)

    print("Accuracy for SVM: ", acc)


def use_naive_bayes(data):
    y = df['Label']
    X = df.drop(columns=['Label', 'User_ID', 'Tweets'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    acc = accuracy_score(predicted, y_test)

    print("Accuracy for Naive Bayes: ", acc)


def use_random_forest(data):
    y = df['Label']
    X = df.drop(columns=['Label', 'User_ID', 'Tweets'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    acc = accuracy_score(predicted, y_test)

    print("Accuracy for Random Forest: ", acc)


df = load_dataframe('training_df_with_basic_features')

use_SVC(df)
use_naive_bayes(df)
use_random_forest(df)
