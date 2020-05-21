from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


class SVMClassifier():

    def __init__(self):
        self.classifier = make_pipeline(StandardScaler(), SVC())


class NaiveBayesClassifier():

    def __init__(self):
        self.classifier = MultinomialNB()


class RandomForestWrapperClassifier():

    def __init__(self):
        self.classifier = RandomForestClassifier()


class EnsembleClassifier():

    def __init__(self):
        clf1 = make_pipeline(StandardScaler(), SVC())
        clf2 = MultinomialNB()
        clf3 = RandomForestClassifier()

        self.voting_classifier = VotingClassifier(
            estimators=[('svm', clf1), ('nb', clf2), ('rf', clf3)], voting='hard')
