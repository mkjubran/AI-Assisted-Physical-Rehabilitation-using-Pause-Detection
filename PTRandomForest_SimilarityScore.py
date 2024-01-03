import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from sklearn.model_selection import  cross_validate
from sklearn.model_selection import RepeatedKFold
import multiprocessing
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
import pdb

# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
dictionary2D = {}  # dictionary to store all 2D files
# files path for similarity score
path_score = '/AIARUPD/Dataset_CVDLPT_Videos_embbeddings_similarityScore_MoveNet_thunder/output_similarity'

#Exercise Type
Exercise="E0"

# Get the list of files in the folder; for specific exercise or all exercises
file_list = [file for file in os.listdir(path_score) if ((os.path.isfile(os.path.join(path_score, file))) and (Exercise in file))]
#file_list = [file for file in os.listdir(path_score) if ((os.path.isfile(os.path.join(path_score, file))) and (True))]

# loop through all files and store them in the dictionary
DataScore=np.array([])
for npzFile in tqdm(file_list, desc=f"Loading Similarity Score"):
    f = os.path.join(path_score, npzFile)
    if os.path.isfile(f):
        if "npz" in f:
            fdata = np.load(f)
            # load the files into the dictionary
            score = fdata['data'].astype(float)
            npzFileSplit=npzFile.split('_')
            Source=np.array([npzFileSplit[0][1],npzFileSplit[1][1],npzFileSplit[2][1],npzFileSplit[3][1],npzFileSplit[4][3:]],dtype=float)
            ArraySource=np.ones((score.shape[0],Source.shape[0]))
            Source_Score=np.concatenate((ArraySource*Source,score),axis=1)

            if not (DataScore.size > 0) :
               DataScore = Source_Score.copy()
            else:
               DataScore = np.concatenate((DataScore,Source_Score), axis=0)

npzOptions=np.array([[3,4,10],[3,8,10,],[3,12,10],[3,16,10],[11,4,5],[11,4,10],[11,4,20],[11,4,30],[11,4,50],[11,8,10],[6,8,10],[6,4,10],[21,4,10]])
cnt=4
print(npzOptions[cnt])

resultsFile=f"Results_Score_MoveNet_thunder_2D.txt"
with open(resultsFile, 'a') as f:
  f.write(f"{Exercise} - {npzOptions[cnt]}")


##Loading Features Vectors for all exercises
for label in tqdm(range(10), desc="Loading Features Vectors"):
   LoadedFV=np.load(f"../HandcraftedFeaturesVector/FeaturesVectors_MoveNet_thunder_2D_E{label}_l{npzOptions[cnt,0]}_s{npzOptions[cnt,1]}_a{npzOptions[cnt,2]}.npy",allow_pickle=True)
   if label == 0:
       featureVector=LoadedFV.copy()
   else:
       featureVector = np.concatenate((featureVector,LoadedFV),axis=0)

#FeatureVector_Score=np.ones((DataScore.shape[0],featureVector.shape[1]+1))
FeatureVector_Score=[]
for cnt in tqdm(range(featureVector.shape[0]), desc=f"Matching Features Vectors and Similarity Scores"):
    # return indices where exercise (featureVector[cnt,0:5]) match exercise (DataScore[:,5:10])
    true_indices = np.where((featureVector[cnt,0:5] == DataScore[:,5:10]).all(axis=1))[0]
    if len(true_indices) > 0:
       tmp = DataScore[true_indices,:]
       idx = np.where(tmp[:,10]==np.max(tmp[:,10]))
       FeatureVector_Score.append(np.concatenate((featureVector[cnt,:],tmp[idx,10].reshape(-1)),axis=0))

    # Add the feature vector to the matrix
    #FeatureVector_Score[true_indices,0:-1]=featureVector[cnt,:]

    # Add the similarity score to the matrix
    #FeatureVector_Score[true_indices,-1]=DataScore[true_indices,10]

FVS=FeatureVector_Score[np.where(FeatureVector_Score[:,0] == 0)[0],5:]
X=FVS[:,:-1]
y=FVS[:,-1]
#rfc = RandomForestRegressor(n_estimators=100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

## ----- Random Forest Regressor
# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the train set
y_train_pred = rf_regressor.predict(X_train)

# Make predictions on the test set
y_test_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse_test = mean_squared_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_test, y_test_pred)
r2_test = r2_score(y_train, y_train_pred)
print(f"RF - R2: Train {r2_train}, Test {r2_test}")
print(f"RF - Mean Squared Error: Train {mse_train}, Test {mse_test}")
with open(resultsFile, 'a') as f:
  f.write(f"RF - R2: Train {r2_train}, Test {r2_test}")
  f.write("RF - Mean Squared Error: Train {mse_train}, Test {mse_test}")

## ----- Multivariate Linear Regression
# Create a multivariate linear regression model using scikit-learn
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the train set
y_train_pred = model.predict(X_train)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_test = mean_squared_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_test, y_test_pred)
r2_test = r2_score(y_train, y_train_pred)
print(f"Multivariate Linear - R2: Train {r2_train}, Test {r2_test}")
print(f"Multivariate Linear - Mean Squared Error: Train {mse_train}, Test {mse_test}")
with open(resultsFile, 'a') as f:
  f.write(f"Multivariate Linear - R2: Train {r2_train}, Test {r2_test}")
  f.write(f"Multivariate Linear - Mean Squared Error: Train {mse_train}, Test {mse_test}")


## ----- Gradient Boosting Regressor
# Create a Gradient Boosting Regression model
# You can adjust hyperparameters such as learning_rate, n_estimators, max_depth, etc.
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the train set
y_train_pred = model.predict(X_train)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_test = mean_squared_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_test, y_test_pred)
r2_test = r2_score(y_train, y_train_pred)
print(f"Gradient Boosting - R2: Train {r2_train}, Test {r2_test}")
print(f"Gradient Boosting - Mean Squared Error: Train {mse_train}, Test {mse_test}")
with open(resultsFile, 'a') as f:
  f.write(f"Gradient Boosting - R2: Train {r2_train}, Test {r2_test}")
  f.write(f"Gradient Boosting - Mean Squared Error: Train {mse_train}, Test {mse_test}")

