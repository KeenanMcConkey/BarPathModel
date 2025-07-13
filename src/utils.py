import os
import yaml

import pickle as pkl
import numpy as np
import pandas as pd

from ast import literal_eval
from pyairtable import Api as airtable_api
from datetime import datetime
from pymongo import MongoClient

DEFAULT_FORMULA = f"AND({{ValidData}} = 'true', {{StartRepTime}} != '', {{EndRepTime}} != '', {{Reps}} >= '1')"
ARRAY_COLUMNS = ['Counter', 'TimeBetweenSamples', 'HeartRate', 'AccX', 'AccY', 'AccZ', 'Pitch', 'Roll', 'Yaw']
OTHER_COLUMNS = ['Date', 'Exercise', 'Lifter', 'WorkoutTime', 'Reps', 'Weight', 'Intensity', 'Notes', 'StartRepTime', 'EndRepTime']

DATA_FOLDER = 'data'
CLASSIFIERS_FOLDER = 'classifiers'
REPS_FOLDER = 'reps'
EXERCISES_FOLDER = 'exercises'
RESULTS_FOLDER = 'results'

DATA_FILENAME = 'raw_data.csv'
CLASSIFIER_FILENAME = 'classifier.pkl'
FEATURES_FILENAME = 'features.csv'
LABELS_FILENAME = 'labels.csv'
REPORT_FILENAME = 'report.txt'
RESULTS_FILENAME = 'results.txt'
CONFIG_FILENAME = 'config.yaml'

CLASSIFIER_TYPES = {
    'Exercise': 'exercises',
    'Window': 'windows'
}

def load_config():
    with open(os.path.join("..", "config.yml"), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_airtable_data(formula=DEFAULT_FORMULA):    
    config = load_config()
    api = airtable_api(config['Airtable']['ApiKey'])
    table = api.table(config['Airtable']['BaseId'], config['Airtable']['TableName'])

    matches = table.all(formula=formula)

    def to_numpy(data): 
        return np.array(literal_eval(data))
    
    def map_data(data):
        return {**{col: data.get(col) for col in OTHER_COLUMNS}, **{col: to_numpy(data.get(col)) for col in ARRAY_COLUMNS}}
    
    data = [map_data(match['fields']) for match in matches]
    return pd.DataFrame(data)


def load_mongo_data():
    config = load_config()
    client = MongoClient(config['Mongo']['Url'])
    database = client.get_database(config['Mongo']['Database'])
    collection = database.get_collection(config['Mongo']['Collection'])
    data = pd.DataFrame(list(collection.find()))

    for column in ARRAY_COLUMNS:
        data[column] = data[column].apply(lambda x: np.array(x))

    # Filter out rows where 'startreptime' is blank
    data = data[data['StartRepTime'].notna() & (data['StartRepTime'] != '')]


    
    return data

def load_classifier(classifier_type):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_name = CLASSIFIER_TYPES[classifier_type]
    classifier_folder = os.path.join('..', CLASSIFIERS_FOLDER, classifier_name)
    classifier_folder = os.path.join(classifier_folder, os.listdir(classifier_folder)[-1])

    classifier_file = os.path.join(classifier_folder, CLASSIFIER_FILENAME)

    with open(classifier_file, 'rb') as f:
        classifier = pkl.load(f)
    
    return classifier


def save_data(data):
    data_folder = os.path.join('..', DATA_FOLDER)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    

    data_file = os.path.join(data_folder, DATA_FILENAME)
    data.to_csv(data_file)


def save_classifier(classifier, classifier_type, classification_report=None):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_folder = os.path.join('..', CLASSIFIERS_FOLDER, CLASSIFIER_TYPES[classifier_type])
    classifier_folder = os.path.join(classifier_folder, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if not os.path.exists(classifier_folder):
        os.makedirs(classifier_folder)
    
    classifier_file = os.path.join(classifier_folder, CLASSIFIER_FILENAME)

    with open(classifier_file, 'wb') as f:
        pkl.dump(classifier, f)
    
    config_file = os.path.join(classifier_folder, CONFIG_FILENAME)
    config = load_config()

    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    if classification_report is None:
        return
    
    report_file = os.path.join(classifier_folder, REPORT_FILENAME)

    with open(report_file, 'w') as f:
        f.write(classification_report)


def save_features(features, classifier_type):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_name = CLASSIFIER_TYPES[classifier_type]
    features_folder = os.path.join('..', DATA_FOLDER, classifier_name)

    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    
    features_file = os.path.join(features_folder, FEATURES_FILENAME)

    features.to_csv(features_file, index=False)


def save_labels(labels, classifier_type):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_name = CLASSIFIER_TYPES[classifier_type]
    labels_folder = os.path.join('..', DATA_FOLDER, classifier_name)

    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    labels_file = os.path.join(labels_folder, LABELS_FILENAME)

    labels.to_csv(labels_file, index=False)
