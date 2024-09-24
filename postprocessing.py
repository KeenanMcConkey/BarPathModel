def predict_exercise_times(predictions, window_length, window_stride, sample_rate, min_window_counter=3):
    exercise_start = None
    exercise_end = None
    window_counter = 0
    
    for i in range(len(predictions) - 1):
        if predictions[1] == 1:
            window_counter += 1

        if predictions[1] == 0:
            window_counter = 0
        
        if exercise_start == None and predictions[i] == 0 and predictions[i + 1] == 1:
            exercise_start = i * window_stride

        if predictions[i] == 1 and predictions[i + 1] == 0:
            if window_counter >= min_window_counter:
                exercise_end = i * window_stride + window_length

    if exercise_start == None:
        exercise_start = 0

    if exercise_end == None:
        exercise_end = len(predictions) * window_stride + window_length
    
    return float(exercise_start) / float(sample_rate), float(exercise_end) / float(sample_rate)