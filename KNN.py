import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class KNN():

    def __init__(self, train_path, test_path):
        # initialize train and test datasets
        self.input_train, self.target_train, self.input_test, self.target_test = self.train_test_prep(train_path,
                                                                                                          test_path)

    def train_test_prep(self, train_path, test_path):
        df_train = pd.read_csv(train_path)
        input_train = df_train.drop('LABELS', axis='columns')
        target_train = df_train['LABELS']

        df_test = pd.read_csv(test_path)
        input_test = df_test.drop('LABELS', axis='columns')
        target_test = df_test['LABELS']

        return input_train, target_train, input_test, target_test

    def KNN_trainer(self):
        error_rate = []
        # Will take some time
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.input_train, self.target_train)
            pred_i = knn.predict(self.input_test)
            error_rate.append(np.mean(pred_i != self.target_test))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')

        optimal_n = min(error_rate)
        return error_rate.index(optimal_n) + 1



    def KNN_test(self,optimal_n):
        neigh = KNeighborsClassifier(n_neighbors=optimal_n)
        neigh.fit(self.input_train, self.target_train)
        neigh.score(self.input_test, self.target_test)
        return neigh.predict(self.input_test)

    def plot_results(self,target_pred_test):
        print("Accraucy of model ::", accuracy_score(self.target_test, target_pred_test))
        print(classification_report(self.target_test, target_pred_test))
        # Get and reshape confusion matrix data
        matrix = confusion_matrix(self.target_test, target_pred_test)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        plt.figure(figsize=(16, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(matrix, annot=True, annot_kws={'size': 15},
                    cmap=plt.cm.Purples, linewidths=0.2)

        # Add labels to the plot
        class_names = ['1', '2', '3',
                       '4', '5', '6']
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=25)
        plt.yticks(tick_marks2, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for K Nearest Neighbours Classification Model')
        plt.show()

