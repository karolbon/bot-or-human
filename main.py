from config import config
from preprocessing.read_data import read_folder_of_xml_files_to_dataframe, load_dataframe, save_dataframe

from sklearn.model_selection import train_test_split
from datetime import datetime


def main():
    if config['read_and_save_raw_data_as_dataframe']:
        df = read_folder_of_xml_files_to_dataframe(
            config['path_to_root_training_data'])
        save_dataframe(df, 'training_data_' + str(datetime.now()))
    else:
        df = load_dataframe(config['training_df'])

    if config['load_preprocessed_dataframe']:
        df = load_dataframe(config['filename_existing_preprocessed_df'])

    y = df['Label']
    X = df.drop(columns=['Label', 'User_ID', 'Tweets'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42)
