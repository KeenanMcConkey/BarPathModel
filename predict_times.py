import sys
sys.path.append("..")

from preprocessing import extract_window_features, preprocess_data
from postprocessing import predict_exercise_indices
from utils import load_config, load_data

from os import path
import pickle as pkl

FORMULA = f"AND({{ValidData}} = 'true', {{StartRepTime}} != '', {{EndRepTime}} != '', {{Reps}} >= '1', {{Date}} > \"2024-09-01\")"


def predict_times(record, config):
    record = preprocess_data(record, config['CutoffFreq'], config['SampleRate'])
    window_features = extract_window_features(record, config['WindowLength'], config['WindowStride'])
    
    clf_file = path.join(config['ExercisesFolder'], config['ClassifierFilename'])
    with open(clf_file, 'rb') as f:
        clf = pkl.load(f)

    predictions = clf.predict(window_features)
    print(f"Predictions: {predictions}")
    start_predict, end_predict = predict_exercise_indices(
        predictions, config['WindowLength'], config['WindowStride'], config['SampleRate'])
    start_actual = float(record['StartRepTime'].values[0])
    end_actual = float(record['EndRepTime'].values[0])

    print(f"Predicted start: {start_predict}, actual start: {start_actual}")
    print(f"Predicted end: {end_predict}, actual end: {end_actual}")

    start_diff = abs(start_predict - start_actual)
    end_diff = abs(end_predict - end_actual)

    print(f"Start difference: {start_diff:.3f}")
    print(f"End difference: {end_diff:.3f}")


if __name__ == "__main__":
    config = load_config()
    data = load_data(FORMULA)
    
    record = data.sample(1)
    predict_times(record, config)
    