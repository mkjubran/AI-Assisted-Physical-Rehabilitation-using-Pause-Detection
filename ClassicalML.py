import os
import pdb
import numpy as np
from scipy.stats import zscore, skew, kurtosis
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from multiprocessing import Process


import json

from datetime import datetime

start_time = datetime.now()

#Parameters
a = 10
l = 21
s = 4

for label in np.arange(10):
    print(label)
    tmp = np.load(f"../PDSaveData/SavedData_E{label}_l{l}_s{s}_a{a}.npy",allow_pickle=True)
    if label == 0:
       Zload=tmp.copy()
    else:
       Zload=np.concatenate((Zload,tmp),axis=0)

X=Zload[:,:-1]
y=Zload[:,-1]


if not np.isnan(X).any():
    print("The array does not contain any NaN values.")
else:
    exit("The array contains NaN values.")


#
# Initialize the imputer with the desired strategy (e.g., 'mean', 'median', or 'most_frequent')
#imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on your feature matrix
#X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-------------------------------
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("precision" , precision)
print("recall" , recall)
print("f1 score " , f1)
print("Classification Report:\n", class_report)

#--------------------------------------MLP --------------------------------
print("------------------------------------------------------------------------------")
print("                                MLP Classifier                              \n")
'''
# Create an MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

#pdb.set_trace()

# Train the MLP on the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_MLP = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy_MLP = accuracy_score(y_test, y_pred_MLP)
precision_MLP = precision_score(y_test, y_pred_MLP, average='weighted', zero_division=1)
recall_MLP = recall_score(y_test, y_pred_MLP, average='weighted', zero_division=1)
f1_MLP = f1_score(y_test, y_pred_MLP, average='weighted', zero_division=1)
class_report_MLP = classification_report(y_test, y_pred_MLP)

print("Accuracy:", accuracy_MLP)
print("precision" , precision_MLP)
print("recall" , recall_MLP)
print("f1 score " , f1_MLP)
print("Classification Report:\n", class_report_MLP)

with open('results.txt', 'w') as f:

    f.write(f"l={l}, s={s}, a={a}")
    f.write(f"Execution time: {(datetime.now() - start_time)} seconds ---")

    f.write(f" ----------- Random Forest ------------\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"precision: {precision}\n")
    f.write(f"recall: {recall}\n")
    f.write(f"f1 score: {f1}\n")
    f.write(f"Classification Report: {class_report}\n")

    f.write(f"----------- MLP -------------\n")
    f.write(f"Accuracy: {accuracy_MLP}\n")
    f.write(f"precision: {precision_MLP}\n")
    f.write(f"recall: {recall_MLP}\n")
    f.write(f"f1 score: {f1_MLP}\n")
    f.write(f"Classification Report: {class_report_MLP}\n")
'''

print("--- %s seconds ---" % (datetime.now() - start_time))
