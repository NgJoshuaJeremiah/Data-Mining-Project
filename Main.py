import KNN as knn
import Random_Forest as rf
import SVM as svm
import XGBoost as xgb
import ANN as ann


# NuSVM_obj = svm.NuSVM(train_path, test_path)

def run_KNN_trainer():
    global KNN_obj
    optimal_n = KNN_obj.KNN_trainer()
    predicted = KNN_obj.KNN_test(optimal_n)
    KNN_obj.plot_results(predicted)

def run_RF_trainer():
    global RF_obj
    RF_obj.RF_trainer()
    predicted = RF_obj.RF_test()
    RF_obj.plot_results(predicted)

def run_SVM_trainer(option=1):
    global SVM_obj, NuSVM_obj
    if option == 1:
        SVM_obj.SVM_trainer()
        predicted = SVM_obj.SVM_test()
        SVM_obj.plot_result(predicted)
    elif option == 2:
        NuSVM_obj.SVM_trainer()
        predicted = NuSVM_obj.SVM_test()
        NuSVM_obj.plot_result(predicted)

def run_XGB_trainer():
    global XGB_obj
    XGB_obj.XGB_trainer()
    predicted = XGB_obj.XGB_test()
    XGB_obj.plot_results(predicted)


def run_ann_trainer():
    global ann_obj
    ann_obj.train_model()

#run_lenet_trainer()
sel = '''
Input number selection for training
raw dataset:
1) KNN
2) Random Forest
3) SVM
4) XGB
5) ANN
Preprocessed dataset without PCA
6) KNN
7) Random Forest
8) SVM
9) XGB
10) ANN
Preprocessed dataset with PCA
11) KNN
12) Random Forest
13) SVM
14) XGB
15) ANN
'''
usr_input = int(input(sel))
if usr_input == 1:
    # RAW
    train_path = './preprocessed_training_set/train_df.csv'
    test_path = './preprocessed_training_set/test_df.csv'
    KNN_obj = knn.KNN(train_path, test_path)
    run_KNN_trainer()
elif usr_input == 2:
    # RAW
    train_path = './preprocessed_training_set/train_df.csv'
    test_path = './preprocessed_training_set/test_df.csv'
    RF_obj = rf.RF(train_path, test_path)
    run_RF_trainer()
elif usr_input == 3:
    # RAW
    train_path = './preprocessed_training_set/train_df.csv'
    test_path = './preprocessed_training_set/test_df.csv'
    SVM_obj = svm.SVM(train_path, test_path)
    run_SVM_trainer()
elif usr_input == 4:
    # RAW
    train_path = './preprocessed_training_set/train_df.csv'
    test_path = './preprocessed_training_set/test_df.csv'
    XGB_obj = xgb.XGB(train_path,test_path)
    run_XGB_trainer()
elif usr_input == 5:
    # RAW
    train_path = './preprocessed_training_set/train_df.csv'
    test_path = './preprocessed_training_set/test_df.csv'
    ann_obj = ann.ANN(train_path, test_path)
    run_ann_trainer()
elif usr_input == 6:
    # #WITH Preprocessing
    train_path = './preprocessed_training_set/train_df_O_T_Smote.csv'
    test_path = './preprocessed_training_set/test_df_log.csv'
    KNN_obj = knn.KNN(train_path, test_path)
    run_KNN_trainer()
elif usr_input == 7:
    # #WITH Preprocessing
    train_path = './preprocessed_training_set/train_df_O_T_Smote.csv'
    test_path = './preprocessed_training_set/test_df_log.csv'
    RF_obj = rf.RF(train_path, test_path)
    run_RF_trainer()
elif usr_input == 8:
    # #WITH Preprocessing
    train_path = './preprocessed_training_set/train_df_O_T_Smote.csv'
    test_path = './preprocessed_training_set/test_df_log.csv'
    SVM_obj = svm.SVM(train_path, test_path)
    run_SVM_trainer()
elif usr_input == 9:
    # #WITH Preprocessing
    train_path = './preprocessed_training_set/train_df_O_T_Smote.csv'
    test_path = './preprocessed_training_set/test_df_log.csv'
    XGB_obj = xgb.XGB(train_path,test_path)
    run_XGB_trainer()
elif usr_input == 10:
    # #WITH Preprocessing
    train_path = './preprocessed_training_set/train_df_O_T_Smote.csv'
    test_path = './preprocessed_training_set/test_df_log.csv'
    ann_obj = ann.ANN(train_path, test_path)
    run_ann_trainer()
elif usr_input == 11:
    # WITH Preprocessing WITH PCA
    train_path = './preprocessed_training_set/train_df_O_T_Smote_PCA.csv'
    test_path = './preprocessed_training_set/test_df_log_PCA.csv'
    KNN_obj = knn.KNN(train_path, test_path)
    run_KNN_trainer()
elif usr_input == 12:
    # WITH Preprocessing WITH PCA
    train_path = './preprocessed_training_set/train_df_O_T_Smote_PCA.csv'
    test_path = './preprocessed_training_set/test_df_log_PCA.csv'
    RF_obj = rf.RF(train_path, test_path)
    run_RF_trainer()
elif usr_input == 13:
    # WITH Preprocessing WITH PCA
    train_path = './preprocessed_training_set/train_df_O_T_Smote_PCA.csv'
    test_path = './preprocessed_training_set/test_df_log_PCA.csv'
    SVM_obj = svm.SVM(train_path, test_path)
    run_SVM_trainer()
elif usr_input == 14:
    # WITH Preprocessing WITH PCA
    train_path = './preprocessed_training_set/train_df_O_T_Smote_PCA.csv'
    test_path = './preprocessed_training_set/test_df_log_PCA.csv'
    XGB_obj = xgb.XGB(train_path, test_path)
    run_XGB_trainer()

elif usr_input == 15:
    # WITH Preprocessing WITH PCA
    train_path = './preprocessed_training_set/train_df_O_T_Smote_PCA.csv'
    test_path = './preprocessed_training_set/test_df_log_PCA.csv'
    ann_obj = ann.ANN(train_path, test_path)
    run_ann_trainer()
else:
    print('wrong input')

