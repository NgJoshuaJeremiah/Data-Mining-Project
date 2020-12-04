import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class XGB():
    def __init__(self,train_path,test_path):
        #initialize train and test datasets
        self.input_train, self.target_train, self.input_test, self.target_test = self.train_test_prep(train_path,test_path)
        self.XGB_classifier = None



    def train_test_prep(self,train_path,test_path):
        df_train = pd.read_csv(train_path)
        input_train = df_train.drop('LABELS', axis='columns')
        target_train = df_train['LABELS']

        df_test = pd.read_csv(test_path)
        input_test = df_test.drop('LABELS', axis='columns')
        target_test = df_test['LABELS']

        return input_train,target_train,input_test,target_test

    def XGB_trainer(self):
        self.XGB_classifier = xgb.XGBClassifier()
        self.XGB_classifier.fit(self.input_train,self.target_train)

    def XGB_test(self):
        predictions = self.XGB_classifier.predict(self.input_test)
        print("Accraucy of model ::", accuracy_score(self.target_test, predictions))
        return predictions

    def plot_results(self,predictions):
        print(classification_report(self.target_test, predictions))
        # Get and reshape confusion matrix data
        matrix = confusion_matrix(self.target_test, predictions)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        plt.figure(figsize=(16, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                    cmap=plt.cm.Greens, linewidths=0.2)

        # Add labels to the plot
        class_names = ['1', '2', '3',
                       '4', '5', '6']
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=25)
        plt.yticks(tick_marks2, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for XGBoost')
        plt.show()