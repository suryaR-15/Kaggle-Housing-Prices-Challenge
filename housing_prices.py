# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:26:32 2020

@author: surya

Kaggle housing prices challenge - https://www.kaggle.com/c/home-data-for-ml-course/overview

XGB Classifier has been used after feature engineering
Cross-validation approach used; best model used to predict labels on test data

Kaggle score (likely MSE) achieved using this version of the code - 16419.47 
"""
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate

from feature_engineering import feature_eng

train_data_path = 'data/train.csv'
test_data_path = 'data/test.csv'

# =============================================================================
# Read in and look at train data using summary descriptors - uncomment as
# required
# =============================================================================
train_data = pd.read_csv(train_data_path)
# print(train_data.columns)
# print(train_data.head())
# print(train_data.shape)

X, y = feature_eng(train_data)

model = XGBRegressor(learning_rate=0.01, n_estimators=1000,
                     max_depth=4, min_child_weight=2,
                     subsample=0.7, colsample_bytree=0.5,
                     objective='reg:squarederror', scale_pos_weight=1)

scores = cross_validate(model, X, y, cv=15, return_estimator=True,
                        scoring='r2')
print("Accuracy on validation data:")
print(scores['test_score'])
max_acc = np.argmax(scores['test_score']) + 1
print("Best r^2 value from estimator number {0} - {1:.2f} "
      .format(max_acc, scores['test_score'][max_acc-1]))
best_model = scores['estimator'][max_acc - 1]

test_data = pd.read_csv(test_data_path)
X_test, y_dummy = feature_eng(test_data)
predictions = best_model.predict(X_test)

# =============================================================================
# Save predictions in Kaggle format to be submitted to the competition
# =============================================================================
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your predictions were successfully saved!")

# =============================================================================
# Uncomment to submit to Kaggle competition - need Kaggle API installed
# =============================================================================
# os.system('kaggle competitions submit -c home-data-for-ml-course -f \
#          my_submission.csv -m "Submitted to Kaggle"')
