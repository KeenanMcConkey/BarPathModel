#!/usr/bin/env python
# coding: utf-8

# # Bar Path Model Training
# 
# Training an model for bar path tracking using exercise data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load data and config

# In[2]:


from utils import load_airtable_data, load_config, save_data

config = load_config()
data = load_airtable_data()
save_data(data)


# ## Train an exercise classifier
# 
# For each data column
# 
# - Extract features from the entire time window for each data set
# - Train model to predict exercise based on these feature
# 
# 
# #### To Do
# 
# - Add back FFT features
# - Feature selection
# - Feature normalization
# - Plot classifier results

# In[3]:


EXERCISE_CLASSIFIER = 'Exercise'


# ### Extract features and labels

# In[4]:


from preprocessing import preprocess_data, extract_features, extract_labels

data = preprocess_data(data, config['CutoffFreq'], config['SampleRate'])
features = extract_features(data)
labels = extract_labels(data, EXERCISE_CLASSIFIER)


# ### Train model

# In[5]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

exercise_classifier = XGBClassifier()
exercise_classifier.fit(X_train, y_train)

print("Classifier type: ", EXERCISE_CLASSIFIER)
print("Training accuracy: ", exercise_classifier.score(X_train, y_train))
print("Testing accuracy: ", exercise_classifier.score(X_test, y_test))

exercise_predictions = exercise_classifier.predict(X_test)
report = classification_report(y_test, exercise_predictions, zero_division=0)
print(report)


# ### Save classifer, features, and labels

# In[6]:


from utils import save_classifier, save_features, save_labels

save_classifier(exercise_classifier, EXERCISE_CLASSIFIER, report)
save_features(features, EXERCISE_CLASSIFIER)
save_labels(labels, EXERCISE_CLASSIFIER)


# ## Train a predictor for whether is in rep state or non rep state
# 
# For entries with a RepStartTime and RepEndTime
#  - Extract windows for each axes
#  - Label window with 0 or 1 depending on whether it is between RepStartTime and RepEndTime
#  - Create a classifier for this data
# 
# #### To Do
# 
# - Add back in FFT features
# - Feature selection
# - Feature normalization
# - Plot classifier results

# In[7]:


WINDOW_CLASSIFIER = 'Window'


# ### Extract windowed features and labels

# In[8]:


from preprocessing import extract_window_features, extract_window_labels

config = load_config()
window_features = extract_window_features(data, config['WindowLength'], config['WindowStride'])
window_labels = extract_window_labels(data, config['WindowLength'], config['WindowStride'], config['SampleRate'])


# ### Train model

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(window_features, window_labels, test_size=0.2)

window_classifier = XGBClassifier()
window_classifier.fit(X_train, y_train)

print("Classifier type: ", WINDOW_CLASSIFIER)
print("Training accuracy: ", window_classifier.score(X_train, y_train))
print("Testing accuracy: ", window_classifier.score(X_test, y_test))

y_pred = window_classifier.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
print(report)


# ### Save classifier, features and labels

# In[10]:


save_classifier(window_classifier, WINDOW_CLASSIFIER, report)
save_features(window_features, WINDOW_CLASSIFIER)
save_labels(window_labels, WINDOW_CLASSIFIER)


# ### Save this notebook to a script

# In[1]:


# Convert the notebook to a script
get_ipython().system('jupyter nbconvert --to script BarPathModelTraining.ipynb --output train_models')

