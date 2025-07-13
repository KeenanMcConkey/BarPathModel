def predict_exercise_indices(predictions, window_length, window_stride, min_consq_windows):
    exercise_start = None
    exercise_end = None
    window_counter = 0
    
    for i in range(len(predictions) - 1):
        if predictions[i] == 1:
            window_counter += 1

        if predictions[i] == 0:
            window_counter = 0
        
        if exercise_start == None and predictions[i] == 0 and predictions[i + 1] == 1:
            if predictions[i + 1:i + min_consq_windows + 1].all():
                exercise_start = i * window_stride
                window_counter = 1

        if predictions[i] == 1 and predictions[i + 1] == 0:
            if window_counter >= min_consq_windows:
                exercise_end = i * window_stride
    
    if exercise_start == None:
        exercise_start = 0

    if exercise_end == None:
        exercise_end = len(predictions) * window_stride + window_length
    
    return exercise_start, exercise_end
