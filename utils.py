import numpy as np
import pandas as pd
import yaml

from ast import literal_eval
from pyairtable import Api as airtable_api

DEFAULT_FORMULA = f"AND({{ValidData}} = 'true', {{StartRepTime}} != '', {{EndRepTime}} != '', {{Reps}} >= '1')"

def load_config():
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_data(formula=DEFAULT_FORMULA):
    config = load_config()
    api = airtable_api(config['ApiKey'])
    table = api.table(config['BaseId'], config['TableName'])

    print("Fetching data from Airtable...")
    print(f"Formula: {formula}")

    matches = table.all(formula=formula)

    print(f"Found {len(matches)} matches")

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
