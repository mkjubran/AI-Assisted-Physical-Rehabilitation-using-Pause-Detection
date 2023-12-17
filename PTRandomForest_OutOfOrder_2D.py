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
import pdb

resultsFile='results_OutOfOrder_Fit_MoveNet_thunder_Eval_Gast_2D_RF.txt'

if __name__ == "__main__":
   npzOptions=np.array([[3,4,10],[3,8,10,],[3,12,10],[3,16,10],[11,4,5],[11,4,10],[11,4,20],[11,4,30],[11,4,50],[11,8,10],[6,8,10],[6,4,10],[21,4,10]])
   print(npzOptions)
   for cnt in range(npzOptions.shape[0]):
      print(f"l={npzOptions[cnt,0]}, s={npzOptions[cnt,1]}, a={npzOptions[cnt,2]}")

      for label in np.arange(10):
          print(f"loading MoveNet_thunder {label} ...")
          tmp = np.load(f"../PDSaveData/SavedData_MoveNet_thunder_2D_E{label}_l{npzOptions[cnt,0]}_s{npzOptions[cnt,1]}_a{npzOptions[cnt,2]}.npy",allow_pickle=True)
          if label == 0:
              Zload = tmp.copy()
          else:
              Zload = np.concatenate((Zload, tmp), axis=0)

      # Find rows with any NaN values
      rows_without_nan = np.all(~np.isnan(Zload), axis=1)

      # Use boolean indexing to get rows without NaN values
      Zload = Zload[rows_without_nan]

      X_train = Zload[:, :-1]
      y_train = Zload[:, -1]

      print(f"l={npzOptions[cnt,0]}, s={npzOptions[cnt,1]}, a={npzOptions[cnt,2]}")

      for label in np.arange(10):
          print(f"loading GAST {label} ...")
          tmp = np.load(f"../PDSaveData/SavedData_GAST_2D_E{label}_l{npzOptions[cnt,0]}_s{npzOptions[cnt,1]}_a{npzOptions[cnt,2]}.npy",allow_pickle=True);
          if label == 0:
              Zload = tmp.copy()
          else:
              Zload = np.concatenate((Zload, tmp), axis=0)

      # Find rows with any NaN values
      rows_without_nan = np.all(~np.isnan(Zload), axis=1)

      # Use boolean indexing to get rows without NaN values
      Zload = Zload[rows_without_nan]

      X_test = Zload[:, :-1]
      y_test = Zload[:, -1]
      #pdb.set_trace()

      with open(resultsFile, 'a') as f:
         f.write(f"\n\nl={npzOptions[cnt,0]}, s={npzOptions[cnt,1]}, a={npzOptions[cnt,2]}\n")

      pdb.set_trace

      for n_estimators in [5,10,20,30,50,100]:
            # Create a Random Forest Classifier
            rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            # Train the classifier on the training data
            rf_classifier.fit(X_train, y_train)

            # Make predictions on the testing data
            y_pred = rf_classifier.predict(X_test)

            # Evaluate the performance of the classifier
            train_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train))
            test_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
            report = classification_report(y_test, y_pred)

            print(f"Fitting and Evaluating RF with {n_estimators} estimators") 
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
            print("\nClassification Report:\n", report)

            with open(resultsFile, 'a') as f:
               f.write(f"Num Estimators = {n_estimators}, Train Acc.:{train_accuracy}, Test Acc.: {test_accuracy}\n")

