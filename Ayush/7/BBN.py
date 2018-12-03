import pandas as pd
import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

HDData = pd.read_csv('Data7.csv')
HDData = HDData.replace('?',np.nan)

model = BayesianModel([('age','fbs'),('age','trestbps'),('sex','trestbps'),('exang','trestbps'),('fbs','HeartDisease'),('trestbps','HeartDisease'),('HeartDisease','restecg'),('HeartDisease','chol'),('HeartDisease','thalach')])
model.fit(HDData, estimator=MaximumLikelihoodEstimator)
HDInfer = VariableElimination(model)

q = HDInfer.query(variables=['HeartDisease'],evidence={'age':40})
print(q['HeartDisease'])
