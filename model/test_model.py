import pickle
import pandas as pd

model = pickle.load(open("model/attrition_model.pkl","rb"))
columns = pickle.load(open("model/model_columns.pkl","rb"))

print(type(model))
print(len(columns))

import numpy as np

sample = pd.DataFrame([np.zeros(len(columns))], columns=columns)

print(model.predict(sample))
print(model.predict_proba(sample))