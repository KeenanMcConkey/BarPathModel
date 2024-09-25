import os
import yaml

import pickle as pkl
import numpy as np
import pandas as pd

from ast import literal_eval
from pyairtable import Api as airtable_api
from datetime import datetime


DEFAULT_FORMULA = f"AND({{ValidData}} = 'true', {{StartRepTime}} != '', {{EndRepTime}} != '', {{Reps}} >= '1')"

DATA_FOLDER = 'data'
CLASSIFIERS_FOLDER = 'classifiers'
REPS_FOLDER = 'reps'
EXERCISES_FOLDER = 'exercises'
RESULTS_FOLDER = 'results'

DATA_FILENAME = 'airtable_data.csv'
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
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_data(data):
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    data_file = os.path.join(DATA_FOLDER, DATA_FILENAME)
    data.to_csv(data_file)


def load_data(formula=DEFAULT_FORMULA):    
    config = load_config()
    api = airtable_api(config['ApiKey'])
    table = api.table(config['BaseId'], config['TableName'])

    matches = table.all(formula=formula)

    def to_numpy(data): 
        return np.array(literal_eval(data))
    
    def map_data(data):
        return {
            'Date': data.get('Date'),
            'Exercise': data.get('Exercise'),
            'Lifter': data.get('Lifter'),
            'WorkoutTime': data.get('WorkoutTime'),
            'Reps': data.get('Reps'),
            'Weight': data.get('Weight'),
            'Intensity': data.get('Intensity'),
            'Notes': data.get('Notes'),
            'StartRepTime': data.get('StartRepTime'),
            'EndRepTime': data.get('EndRepTime'),
            'Counter': to_numpy(data.get('Counter')),
            'TimeBetweenSamples': to_numpy(data.get('TimeBetweenSamples')),
            'AccX': to_numpy(data.get('AccX')),
            'AccY': to_numpy(data.get('AccY')),
            'AccZ': to_numpy(data.get('AccZ')),
            'Pitch': to_numpy(data.get('Pitch')),
            'Roll': to_numpy(data.get('Roll')),
            'Yaw': to_numpy(data.get('Yaw')),
            'HeartRate': to_numpy(data.get('HeartRate')),
        }
    
    data = [map_data(match['fields']) for match in matches]
    return pd.DataFrame(data)


def save_classifier(classifier, classifier_type, classification_report=None):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_folder = os.path.join(CLASSIFIERS_FOLDER, CLASSIFIER_TYPES[classifier_type])
    # with current_date time
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


def load_classifier(classifier_type):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_name = CLASSIFIER_TYPES[classifier_type]
    classifier_folder = os.path.join(CLASSIFIERS_FOLDER, classifier_name)
    classifier_folder = os.path.join(classifier_folder, os.listdir(classifier_folder)[-1])

    classifier_file = os.path.join(classifier_folder, CLASSIFIER_FILENAME)

    with open(classifier_file, 'rb') as f:
        classifier = pkl.load(f)
    
    return classifier


def save_features(features, classifier_type):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_name = CLASSIFIER_TYPES[classifier_type]
    features_folder = os.path.join(DATA_FOLDER, classifier_name)

    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    
    features_file = os.path.join(features_folder, FEATURES_FILENAME)

    features.to_csv(features_file, index=False)


def save_labels(labels, classifier_type):
    if classifier_type not in CLASSIFIER_TYPES:
        raise ValueError(f"Invalid classifier type: {classifier_type}")
    
    classifier_name = CLASSIFIER_TYPES[classifier_type]
    labels_folder = os.path.join(DATA_FOLDER, classifier_name)

    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    
    labels_file = os.path.join(labels_folder, LABELS_FILENAME)

    labels.to_csv(labels_file, index=False)