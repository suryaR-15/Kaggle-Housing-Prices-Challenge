# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:42:58 2020

@author: surya

Feature engineering module for the housing prices challenge in Kaggle.

feature_eng()
input: dataframe (can be train/validation data or test data)
output: X - dataframe containing features after feature engineering
        y - list containing 'Survived' labels for train/validation data
          - set to 0 for test data

Module contains two functions:

plot_features(X, selected_columns) - to visualise relationship between
different features and the target labels in bar plot form. Defined as a
separate function to enable use at various stages of the workflow and
support decision-making in feature engineering. Only features in 'selected_
columns' are plotted to avoid clutter.

feature_eng(data) - the main feature engineering module. Developed specifically
for the Kaggle housing prices challenge after close inspection of the training
data so as to perform as best as possible on the test data.
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def plot_features(X, selected_columns):
    """
    plot as bar graphs, each feature against the survival probability. Can
    be used to visualise their relationship and consequently, the importance
    in training.
    """
    features = selected_columns
    for feature in features:
        g = sns.catplot(x=feature, y="SalePrice", data=X, kind="bar",
                        height=6, palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("Sale Price")


def feature_eng(data):
    """
    Parameters
    ----------
    data : TYPE: Pandas dataframe
           DESCRIPTION: Contains training/validation data or test data

    Returns
    -------
    X : TYPE: Pandas dataframe
           DESCRIPTION: Contains features after feature engineering
    y : TYPE: Series object containing target labels
        DESCRIPTION: Contains labels if X is training or validation data;
                     else set to a dummy value of 0 for test data

    Due to the large number of features, only those having correlations with
    target labels greater than a minimum acceptable threshold are used for
    training and predictions.
    """

# =============================================================================
#   Plot correlations between features and target labels (Sale price) using a
#   heatmap. Set a threshold for a minimum acceptable correlation since there
#   are 81 features - would be hard to visualise all at once. Extract the
#   columns with correlation values greater than this threshold and use for
#   training and predictions. Plot bar graphs for these features to see how
#   they relate to the sale price (using plot_features()) - uncomment if
#   required as plotting several graphs may slow down the code a bit.
# =============================================================================
    min_corr = 0.3
    data = data.iloc[:, ::-1]
    if 'SalePrice' in data.columns:
        corr_series = data.iloc[:, 1:80].corrwith(data['SalePrice'])
        corrs = corr_series.loc[corr_series > min_corr]
        selected_features = corrs.index.tolist()
        # sns.heatmap(corrs.values.reshape(corrs.shape[0], 1), annot=True,
        #             fmt='.2f')
        # plot_features(data, selected_features)
        n = len(selected_features)
        print("The {0} important features in this dataset are: ".format(n))
        print(selected_features)

# =============================================================================
#   Look at the important features (selected_features) and combine if
#   appropriate to reduce the number of features and make them stronger. For
#   example, surface areas of different floors in sq.ft. may be combined to
#   give the total surface area of the house. Removing correlated features
#   such as GarageArea and GarageCars gave slightly worse performance than
#   when keeping it. So, correlated features have not been dropped here.
# =============================================================================
    houses = pd.DataFrame()
    houses['OutsideArea'] = data['OpenPorchSF'] + data['WoodDeckSF']
    houses['InsideArea'] = data['GrLivArea'] + data['2ndFlrSF'] + \
        data['1stFlrSF'] + data['TotalBsmtSF']
    houses['TotalRooms'] = data['TotRmsAbvGrd']
    houses['GarageArea'] = data['GarageArea']
    houses['TotalCars'] = data['GarageCars']
    houses['GarageYearBuilt'] = data['GarageYrBlt']
    houses['Fireplaces'] = data['Fireplaces']
    houses['Bathrooms'] = data['FullBath']
    houses['Type1Basement'] = data['BsmtFinSF1']
    houses['VeneerArea'] = data['MasVnrArea']
    houses['RemodelYear'] = data['YearRemodAdd']
    houses['YearBuilt'] = data['YearBuilt']
    houses['Quality'] = data['OverallQual']
    houses['Frontage'] = data['LotFrontage']
# =============================================================================
#   Visualise the new features and correlation with SalePrice
# =============================================================================
    if 'SalePrice' in data.columns:
        houses['SalePrice'] = data['SalePrice']
        # plot_features(houses, houses.columns)
        # sns.heatmap(houses.corr(), annot=True, fmt='.2f')
        houses.drop(['SalePrice'], axis=1, inplace=True)
    # print(houses.head())

# =============================================================================
#   Imputing missing entries separately for categorical and numerical columns
# =============================================================================
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')
    for feature in houses.columns:
        if houses[feature].dtype == 'object':
            houses[feature] = cat_imputer.fit_transform(
                houses[feature].values.reshape(houses.shape[0], 1))
        elif houses[feature].dtype in ['int64', 'float64']:
            if feature in ['YearBuilt', 'RemodelYear', 'GarageYearBuilt']:
                houses[feature] = cat_imputer.fit_transform(
                    houses[feature].values.reshape(houses.shape[0], 1))
            else:
                houses[feature] = num_imputer.fit_transform(
                    houses[feature].values.reshape(houses.shape[0], 1))

# =============================================================================
#   Return X and y
# =============================================================================
    if 'SalePrice' in data.columns:
        y = data['SalePrice']
    else:
        y = 0
    X = houses

    return X, y
