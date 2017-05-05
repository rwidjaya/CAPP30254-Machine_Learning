from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import re
import datetime as dt
import seaborn as sns


### READING DATA ###

def read_file(file_name):
	'''
	Given a 'xls', 'csv', or 'xlsx' file, read file into a pandas dataframe

	Input: (String) file name

	Output: (df) pandas dataframe 
	'''

    pattern = r'[(?!=.)]([a-z]*)'
    file_type = re.findall(pattern, file_name)[0]
    
    if file_type == 'csv':
        data = pd.read_csv(file_name)
      
    elif file_type in ['xls', 'xlsx']:
        data = pd.read_excel(file_name)
    
    return data



def colname_to_snake(data):
    
    def camelCase_to_snake_case(colname):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', colname)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    
    return [camelCase_to_snake_case(x) for x in data.columns.tolist()]



def num_feature_summary(data, num_features):

	summary_df = data[num_features].describe()

	return summary_df



def feature_corr(data, features, notebook = 0):
    feature_data = data.loc[:,features]
    feature_corr = feature_data.corr()
    
    if notebook == 1:
    	corr_heatmap = sns.heatmap(feature_corr,
                annot = True,
                xticklabels=feature_corr.columns.values,
                yticklabels=feature_corr.columns.values)

    	return corr_heatmap
    
    else:
    	return feature_corr


### Preprocess ###

def count_null_by(data, features, group_by):
	'''
	Create a data crosstab dataframe that counts the number of nulls of selected features grouped by a column

	Input:
		- (df) data
		- (list) list of features/column names
		- (string) column name to be grouped by 
	'''
	null_countby_df = pd.DataFrame()
	colname = []

	for f in features:
		colname += [f]
		null_count = data[data[f].isnull()].groupby(group_by).size().to_frame()

		null_countby_df = pd.concat([null_countby_df, null_count], axis=1)
	    
	null_countby_df.columns = colname
	    
	return null_countby_df



def preprocess_data(data, var_dict, group_by = None):
	for var_type in var_dict:
		if var_type == 'continuous_var':
			for var in var_dict[var_type]:
				if group_by:
					data[var] = data[var].fillna(data.groupby(group_by)[var].transform('mean'))
				else:
					data[var] = data[var].fillna(data[var].mean())

		if var_type == 'num_categorical_var':
			for var in var_dict[var_type]:
				data[var] = data[var].fillna(0)

		if var_type == 'categorical_var':
			for var in var_dict[var_type]:
				data[var] = data[var].fillna('None')
				data[var] = data[var].astype('category')

	return data


### VARIABLE TRANSFORMATION ###

def dicretize_var(data, colname, crit_dict, new_colname = None, inplace = False):
    
    bins = crit_dict[colname]['bins']
    group_names = crit_dict[colname]['group_names']

    new_cat_var = pd.cut(data[colname], bins, labels=group_names)

    if inplace:
    	assert new_colname, "New categorical column needs 'new_colname'."
    	
    	data[new_colname] = new_cat_var

    else:
    	return categorical_column



def dummy_var(data, colname, inplace = False):
    
    dummy_df = pd.get_dummies(data[colname])
    
    if inplace:

    	data = data.join(dummy_df)

    else:
    	return dummy_df


### PREDICTING FEATURE IMPORTANCE ###

def feature_importance_rf(data, features, y, notebook = 0):
	rdf = RandomForestClassifier()
	rdf.fit(data[features], data[y])

	feat_array = np.array(features)
	importances = rdf.feature_importances_

	sorted_idx = np.argsort(importances)

	if notebook == 1:
		padding = np.arange(len(feat_array)) + 0.5
		pl.barh(padding, importances[sorted_idx], align='center')
		pl.yticks(padding, feat_array[sorted_idx])
		pl.xlabel("Relative Importance")
		pl.title("Variable Importance")
		pl.show()

	else:
		return sorted_idx


