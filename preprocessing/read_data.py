import glob
import pandas as pd
from xml.dom import minidom


def read_txt_file_of_truth(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    rows = []
    for line in lines:
        splitted = line.strip().split(":::")
        row = {'User_ID': splitted[0], 'Label': splitted[1]}
        rows.append(row)
    df = pd.DataFrame(rows, columns=["User_ID", "Label"])
    return df


def read_one_xml_file(filepath):
    document = minidom.parse(filepath)
    tweets = document.getElementsByTagName('document')
    concatenated_tweets = ""
    for tweet in tweets:
        concatenated_tweets += tweet.firstChild.data + " "

    # Getting only the filename excluding path and fileending
    hash_id = filepath.split('/')[-1].split('.')[0]

    return {'User_ID': hash_id, 'Tweets': concatenated_tweets}


def read_folder_of_xml_files_to_dataframe(path_to_root_folder):
    filepaths = glob.glob(path_to_root_folder + "*.xml")
    rows = []
    for filepath in filepaths:
        row = read_one_xml_file(filepath)
        rows.append(row)
    df = pd.DataFrame(rows, columns=["User_ID", "Tweets"])
    return df


def save_dataframe(df, filename):
    df.to_pickle("data/"+filename+".pkl")
    print("DataFrame saved to file " + filename + ".pkl.")


def load_dataframe(filename):
    df = pd.read_pickle("data/"+filename+".pkl")
    print("DateFrame loaded from data/" + filename + ".pkl.")
    return df


def read_training_data():
    root_path = 'data/training/'
    training_data = read_folder_of_xml_files_to_dataframe(root_path)
    training_label = read_txt_file_of_truth(root_path + "/truth.txt")
    training = pd.merge(training_data, training_label, on='User_ID')
    save_dataframe(training, 'training_df')
    return training


def read_test_data():
    root_path = 'data/test/'
    test_data = read_folder_of_xml_files_to_dataframe(root_path)
    test_label = read_txt_file_of_truth(root_path + '/truth.txt')
    return test_data, test_label
