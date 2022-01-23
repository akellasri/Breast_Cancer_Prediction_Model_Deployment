# Importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

# Import dataset
df = pd.read_csv("Breast_Cancer_Prediction.csv")

 # we change the class values from B to 0 and from M to 1
df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)

# split the data
X = df[['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
       'texture_se', 'area_se', 'texture_worst', 'smoothness_worst',
       'compactness_worst', 'symmetry_worst']]

y = df["diagnosis"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Model building
from sklearn.linear_model import LogisticRegression

# loading the logistic regression model to the variable clf
clf = LogisticRegression() 

# training the model on training data
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Confusion Matrix : \n\n" , confusion_matrix(prediction,y_test))

print("Classification Report : \n\n" , classification_report(prediction,y_test),"\n")

# store this model in pickle file uisng clf 
pickle.dump(clf, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)
