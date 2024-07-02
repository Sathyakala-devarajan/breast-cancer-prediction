# import libraries
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# import dataset from sklearn
from sklearn.datasets import load_breast_cancer

# Load the dataset
Data = load_breast_cancer()
X = pd.DataFrame(Data.data, columns= Data.feature_names)
Y = pd.Series(Data.target, name='target')


# Train & Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=10)

# No. of decision trees
Estimator = [10, 30, 40, 50, 60]

# No. of branches in each tree
Max_depth = [1, 3, 5, 6, 7]

for est in Estimator:
    for depth in Max_depth:
        RFC = RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_split=2, random_state=10)
        RFC.fit(X_train, Y_train)
        Y_train_Predict = RFC.predict(X_train)
        Y_test_Predict = RFC.predict(X_test)
                
        Train_accuracy = accuracy_score(Y_train, Y_train_Predict)
        Train_Precission = precision_score(Y_train, Y_train_Predict)
        Train_recall= recall_score(Y_train, Y_train_Predict)
        Train_f1= f1_score(Y_train, Y_train_Predict)

        Test_accuracy = accuracy_score(Y_test, Y_test_Predict)
        Test_Precission = precision_score(Y_test, Y_test_Predict)
        Test_recall= recall_score(Y_test, Y_test_Predict)
        Test_f1= f1_score(Y_test, Y_test_Predict)

        print(f'Estimator: {est}, Max_depth : {depth} ')
        print(Train_accuracy,Train_Precission, Train_recall, Train_f1)
        print(Test_accuracy,Test_Precission, Test_recall, Test_f1)

        with mlflow.start_run():
            mlflow.log_param('n_estimators', est)
            mlflow.log_param('Max_depth', depth)
            
            mlflow.log_metric('Trian_accuracy', Train_accuracy)
            mlflow.log_metric('Train_Precision', Train_Precission)
            mlflow.log_metric('Train_recall', Train_recall)
            mlflow.log_metric('Train_f1',Train_f1)
            
            mlflow.log_metric('Test_accuracy', Test_accuracy)
            mlflow.log_metric('Test_Precision', Test_Precission)
            mlflow.log_metric('Test_recall', Test_recall)
            mlflow.log_metric('Test_f1',Test_f1)
            
            mlflow.sklearn.log_model(RFC, 'RandomforestClassifier')
