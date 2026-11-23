# CSC311 Final Project

## Prediction/Model done so far
- Sylvia: Added a preliminary random forest model in model.ipynb and a TFIDF vectorization on the text columns in data_cleaning.py. Validation accuracy shown in the notebook. Added preliminary logistic regression model + the validation. Validation accuracy is very high with random forest (~ 95%) and logistic is lower at ~92%. Can try an ensemble of models for different types of features (e.g. Naive Bayes for text features, logistic regression/random forest for remaining features) or a majority vote.