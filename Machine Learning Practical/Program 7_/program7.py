import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('heart.csv')

heartDisease = heartDisease.replace('?', np.nan)

print('Sample Instances from the dataset are given below:')
print(heartDisease.head())

print('\nAttributes and data types:')
print(heartDisease.dtypes)

heartDisease['ca'] = pd.to_numeric(heartDisease['ca'], errors='coerce')
heartDisease['thal'] = pd.to_numeric(heartDisease['thal'], errors='coerce')

heartDisease = heartDisease.fillna(heartDisease.mode().iloc[0])

model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence = restecg : 1')
q1 = HeartDisease_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

print('\n2. Probability of HeartDisease given evidence = cp : 2')
q2 = HeartDisease_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
