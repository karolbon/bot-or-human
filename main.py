from datetime import datetime
import pandas as pd
from config import config
from preprocessing.read_data import read_test_data, load_dataframe, save_dataframe, read_folder_of_xml_files_to_dataframe
from preprocessing.feature_engineering import feature_engineering
from models import SVMClassifier, NaiveBayesClassifier, RandomForestWrapperClassifier

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt


def main():
    if config['read_and_save_raw_data_as_dataframe']:
        df = read_folder_of_xml_files_to_dataframe(
            config['path_to_root_training_data'])
        save_dataframe(df, 'training_data_' + str(datetime.now()))
    else:
        df = load_dataframe(config['filename_existing_data_df'])

    if config['load_preprocessed_dataframe']:
        df = load_dataframe(config['filename_exisiting_preprocessed_df'])
    else:
        df = feature_engineering(df, is_training=True)
        save_dataframe(df, 'training_df_preprocessed_features_22_05')
    y = [1 if label == 'bot' else 0 for label in df['Label']]
    X = df.drop(columns=['Label', 'User_ID', 'Tweets'])

    for validation_size in config['validation_set_sizes']:
        print("Doing experiment with validation set size", validation_size)
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=validation_size, random_state=42)

        # Set up classifiers and fit to data
        # SVM
        print("Training SVM " + str(datetime.now()))
        svm_classifier = SVMClassifier()
        svm_classifier.classifier.fit(X_train, y_train)

        # Naive Bayes
        print("Training Naïve Bayes " + str(datetime.now()))
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.classifier.fit(X_train, y_train)

        # Random Forest
        print("Training Random Forest " + str(datetime.now()))
        rf_classifier = RandomForestWrapperClassifier()
        rf_classifier.classifier.fit(X_train, y_train)

        print("Done training model. Now doing predictions...")

        # Do predictions on test data and measure accuracy
        svm_y_predicted = svm_classifier.classifier.predict(X_validation)
        svm_accuracy = accuracy_score(y_validation, svm_y_predicted)

        nb_y_predicted = nb_classifier.classifier.predict(X_validation)
        nb_accuracy = accuracy_score(y_validation, nb_y_predicted)

        rf_y_predicted = rf_classifier.classifier.predict(X_validation)
        rf_accuracy = accuracy_score(y_validation, rf_y_predicted)

        print("SVM accuracy:", svm_accuracy)
        print("Naive Bayes accuracy:", nb_accuracy)
        print("Random Forest accuracy:", rf_accuracy)
        print("")

    if config['load_trained_model']:
        # Load pre trained models
        print("Loading saved models.", datetime.now())
        final_svm_classifier = pickle.load(open('final_svm_trained', 'rb'))
        final_nb_classifier = pickle.load(open('final_nb_trained', 'rb'))
        final_rf_classifier = pickle.load(open('final_rf_trained', 'rb'))

        print("Loaded trained models from file.", datetime.now())
    else:
        # Train models on all training data for final predictions on test data

        print("Training final SVM " + str(datetime.now()))
        final_svm_classifier = SVMClassifier()
        final_svm_classifier.classifier.fit(X, y)
        pickle.dump(final_svm_classifier, open('final_svm_trained', 'wb'))

        print("Training final Naïve Bayes " + str(datetime.now()))
        final_nb_classifier = NaiveBayesClassifier()
        final_nb_classifier.classifier.fit(X, y)
        pickle.dump(final_nb_classifier, open('final_nb_trained', 'wb'))

        print("Training final Random Forest " + str(datetime.now()))
        final_rf_classifier = RandomForestWrapperClassifier()
        final_rf_classifier.classifier.fit(X, y)
        pickle.dump(final_rf_classifier, open('final_rf_trained', 'wb'))

        print("Done training final models.")

    if config['load_preprocessed_dataframe_of_test_data']:
        # Load test data frame from file
        test_data = load_dataframe(
            'test_df_preprocessed_features')

        test_y = [1 if label == 'bot' else 0 for label in test_data['Label']]
        test_X = test_data.drop(columns=['User_ID', 'Tweets', 'Label'])

    else:
        # Read test data
        print("Reading test data")
        test_data, test_label = read_test_data()
        test = pd.merge(test_data, test_label, on='User_ID')
        save_dataframe(test, 'test_df')

        test_y = [1 if label == 'bot' else 0 for label in test['Label']]

        # Drop label before feature engineering
        test_X_pre_features = test.drop(columns=['Label'])

        print("Preprocessing test data " + str(datetime.now()))
        test_X = feature_engineering(test_X_pre_features, is_training=False)

        # Want to save dataframe with all columns for faster loading and processing
        test_df_for_saving = test_X.copy()
        test_df_for_saving['Label'] = test_y
        save_dataframe(test_df_for_saving, 'test_df_preprocessed_features')

        # Drop columns not used for classification
        test_X = test_X.drop(columns=['User_ID', 'Tweets'])

    print("Doing final predictions with SVM.", datetime.now())
    final_svm_predictions = final_svm_classifier.classifier.predict(test_X)
    final_svm_accuracy = accuracy_score(test_y, final_svm_predictions)

    class_names = ["Bot", "Human"]
    plot_svm = plot_confusion_matrix(final_svm_classifier.classifier,
                                     test_X, test_y, display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize='true')
    plot_svm.ax_.set_title("Support Vector Machine")
    plt.savefig('results/svm.png')

    print("Doing final predictions with Naïve Bayes.", datetime.now())
    final_nb_predictions = final_nb_classifier.classifier.predict(test_X)
    final_nb_accuracy = accuracy_score(test_y, final_nb_predictions)

    plot_nb = plot_confusion_matrix(final_nb_classifier.classifier,
                                    test_X, test_y, display_labels=class_names,
                                    cmap=plt.cm.Blues, normalize='true')
    plot_nb.ax_.set_title("Naïve Bayes")
    plt.savefig('results/nb.png')

    print("Doing final predictions with Random Forest.", datetime.now())
    final_rf_predictions = final_rf_classifier.classifier.predict(test_X)
    final_rf_accuracy = accuracy_score(test_y, final_rf_predictions)

    plot_rf = plot_confusion_matrix(final_rf_classifier.classifier,
                                    test_X, test_y, display_labels=class_names,
                                    cmap=plt.cm.Blues, normalize='true')
    plot_rf.ax_.set_title("Random Forest")
    plt.savefig('results/rf.png')

    # Accuracy for final model
    print("Final SVM accuracy:", final_svm_accuracy)
    print("Final Naive Bayes accuracy:", final_nb_accuracy)
    print("Final Random Forest accuracy:", final_rf_accuracy)
    print("")


if __name__ == "__main__":
    main()
