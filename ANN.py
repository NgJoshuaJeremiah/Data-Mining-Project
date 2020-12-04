from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

class ANN():
    def __init__(self, train_path, test_path):
        # initialize train and test datasets
        self.input_train, self.target_train, self.input_test, self.target_test = self.train_test_prep(train_path,test_path)
        self.model = None

    def train_test_prep(self, train_path, test_path):
        df_train = pd.read_csv(train_path)
        input_train = df_train.drop('LABELS', axis='columns')
        target_train = df_train['LABELS'] -1

        df_test = pd.read_csv(test_path)
        input_test = df_test.drop('LABELS', axis='columns')
        target_test = df_test['LABELS'] -1

        return input_train, target_train, input_test, target_test


    def model_init(self):

        #model creation
        #keras is batch X 58(timesteps) X 768 (dimensional vectors)
        input_shape = self.input_train.shape[1]
        outputs = 6
        model = keras.Sequential([

                layers.Dense(input_shape,activation='relu'),
                layers.Dense(64,activation='relu'),
                layers.Dense(32,activation='relu'),
                layers.Dense(outputs,activation='softmax')

            ])

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics='accuracy')
        return model

    def train_model(self):
        self.model = self.model_init()
        callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
        result = self.model.fit(self.input_train,
                           self.target_train,
                         batch_size = 8,
                         epochs= 300,
                         validation_data=(self.input_test,self.target_test),shuffle=True,callbacks=[callback],verbose=2)


        y_predict = self.model.predict_classes(self.input_test)
        print("Accraucy of model ::", accuracy_score(self.target_test, y_predict))
        # test_X
        con_matrix = confusion_matrix(self.target_test,y_predict)
        con_matrix = con_matrix.astype('float') / con_matrix.sum(axis=1)[:,np.newaxis]

        # Build the plot
        plt.figure(figsize=(16, 7))
        sns.set(font_scale=1.4)
        sns.heatmap(con_matrix, annot=True, annot_kws={'size': 10},
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
        plt.title('Confusion Matrix for ANN')
        plt.show()
        print(classification_report(self.target_test,y_predict))
