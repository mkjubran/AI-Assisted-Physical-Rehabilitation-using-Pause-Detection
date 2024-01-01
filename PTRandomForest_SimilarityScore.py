import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
import pdb

# ------------------------------------------- LOAD .npz FILES --------------------------------------------- #
dictionary2D = {}  # dictionary to store all 2D files
# files path for similarity score
path_score = '/AIARUPD/Dataset_CVDLPT_Videos_embbeddings_similarityScore_MoveNet_thunder/output_similarity'

#Exercise Type
Exercise="E0"
# Get the list of files in the folder
file_list = [file for file in os.listdir(path_score) if ((os.path.isfile(os.path.join(path_score, file))) and (Exercise in file))]
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
cnt=1
label=Exercise[1]
featureVector=np.load(f"../HandcraftedFeaturesVector/SavedData_MoveNet_thunder_2D_E{label}_l{npzOptions[cnt,0]}_s{npzOptions[cnt,1]}_a{npzOptions[cnt,2]}.npy",allow_pickle=True)

resultsFile=f"{Exercise}_results_Score_MoveNet_thunder_2D_RF.txt"
pdb.set_trace()

'''
if __name__ == "__main__":
   npzOptions=np.array([[3,4,10],[3,8,10,],[3,12,10],[3,16,10],[11,4,5],[11,4,10],[11,4,20],[11,4,30],[11,4,50],[11,8,10],[6,8,10],[6,4,10],[21,4,10]])
   print(npzOptions)
   for cnt in range(npzOptions.shape[0]):
      print(f"l={npzOptions[cnt,0]}, s={npzOptions[cnt,1]}, a={npzOptions[cnt,2]}")

      for label in np.arange(10):
          print(f"loading {label} ...")
          tmp = np.load(f"../PDSaveData/SavedData_MoveNet_thunder_2D_E{label}_l{npzOptions[cnt,0]}_s{npzOptions[cnt,1]}_a{npzOptions[cnt,2]}.npy",allow_pickle=True)
          if label == 0:
              Zload = tmp.copy()
          else:
              Zload = np.concatenate((Zload, tmp), axis=0)

      # Find rows with any NaN values
      rows_without_nan = np.all(~np.isnan(Zload), axis=1)

      # Use boolean indexing to get rows without NaN values
      Zload = Zload[rows_without_nan]

      X = Zload[:, :-1]
      y = Zload[:, -1]

      with open(resultsFile, 'a') as f:
         f.write(f"\n\nl={npzOptions[cnt,0]}, s={npzOptions[cnt,1]}, a={npzOptions[cnt,2]}\n")

      for n_estimators in [5,10,20,30,50,100]:
         print(f"Fitting and Evaluating RF with {n_estimators} estimators")
         rfc = RandomForestClassifier(n_estimators=n_estimators)
         #cv = RepeatedKFold(n_splits=10, n_repeats=5)
         scores = cross_validate(rfc, X=X, y=y, cv=50, return_train_score=True)
         print(n_estimators,np.mean(scores['train_score']),np.mean(scores['test_score']),np.mean(scores['fit_time']),np.mean(scores['score_time']))

         with open(resultsFile, 'a') as f:
            f.write(f"Num Estimators = {n_estimators}, Train Acc.:{np.mean(scores['train_score'])}, Test Acc.: {np.mean(scores['test_score'])}, Train Time (Sec):{np.mean(scores['fit_time'])}, Test Time (Sec): {np.mean(scores['score_time'])}\n")
      
'''
