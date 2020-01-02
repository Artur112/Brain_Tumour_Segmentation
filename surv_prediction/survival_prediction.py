from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd


data_path = 'features.csv'
training_data = pd.read_csv(data_path, index_col=0)
features = training_data.drop(['Survival'], axis=1)
labels = training_data['Survival']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)

# Get accuracy in predicting short (<10m), medium (10m<x<15m) and long (>15m) survivors
test_labels_month = test_labels/31
test_labels_cl = np.zeros(test_labels.shape)
test_labels_cl[test_labels_month < 10] = 0
test_labels_cl[(test_labels_month >= 10) & (test_labels_month < 15)] = 1
test_labels_cl[test_labels_month >= 15] = 2

predictions_month = predictions / 31
predictions_cl = np.zeros(predictions.shape)
predictions_cl[predictions_month < 10] = 0
predictions_cl[(predictions_month >= 10) & (predictions_month < 15)] = 1
predictions_cl[predictions_month >= 15] = 2

accuracy = np.sum(predictions_cl == test_labels_cl)/len(test_labels_cl)*100
print('Short, Medium and Long survival accuracy: {:.2f}%'.format(accuracy))

errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'days')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



# # Get numerical feature importances
#importances = list(rf.feature_importances_)
# List of tuples with variable and importance
#feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
#feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

