import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from numpy import VisibleDeprecationWarning, genfromtxt
import numpy as np

np.random.seed(13452346)

def Classify(clf, folds):
    best_model = [None, 0.]
    num_folds = len(folds)
    for fold in range(num_folds):
        mask = np.zeros(num_folds)
        mask[fold] = True
        training = np.ma.masked_array(folds, mask=mask)
        training = np.concatenate(training[~training.mask])
        validation = folds[fold]
        X = np.array(training[:, :-1])
        y = np.array(training[:, -1:].flatten())
        clf.fit(X, y)
        predictions = clf.predict(validation[:, :-1])
        correct_count = 0
        for i in range(len(predictions)):
            if predictions[i] == validation[i, -1]:
                correct_count += 1
        accuracy = correct_count / len(validation)
        if accuracy > best_model[1]:
            best_model = [clf, accuracy]
        
    return best_model

def Folds(data_location, dtype=int, train_percent=80, num_folds=10):
    dataframe = genfromtxt(data_location, delimiter=',', dtype=dtype)[1:]
    train_index = round(len(dataframe) * train_percent / 100)
    training, test = dataframe[:train_index], dataframe[train_index:]

    np.random.shuffle(dataframe)

    return (np.array_split(training, num_folds), test)

def Test(model_perf, test):

    final_output = []
    for item in model_perf:
        dt_clf = item[1][0]

        predictions = dt_clf.predict(test[:, :-1])

        correct_count = 0
        for i in range(len(predictions)):
            if predictions[i] == test[i, -1]:
                correct_count += 1
        test_perf = correct_count / len(test)

        final_output.append([item[0], test_perf])


    return final_output

def KNN(folds, k_min=1, k_max=5):
    model_perf = []
    for k in range(k_min, k_max + 1):
        # print(k)
        knn_clf = KNeighborsClassifier(n_neighbors=k)

        model_perf.append([k, Classify(clf=knn_clf, folds=folds)])

    return model_perf

def DecisionTree(folds, depth_min=1, depth_max=5):
    model_perf = []
    for depth in range(depth_min, depth_max + 1):
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        model_perf.append([depth, Classify(clf=dt_clf, folds=folds)])

    return model_perf

def BoostedDecisionTree(folds, estimators_min=1, estimators_max=5, depth=1, learning_rate = 1.0):
    model_perf = []
    for n_estimators in range(estimators_min, estimators_max + 1):
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        boosted_clf = AdaBoostClassifier(dt_clf, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
        model_perf.append([n_estimators, Classify(clf=boosted_clf, folds=folds)])

    return model_perf

def NeuralNet(folds, iter_min=1, iter_max=5):
    model_perf = []
    for iter in range(iter_min, iter_max + 1):
        mlp_clf = MLPClassifier(random_state=0, max_iter=iter)
        model_perf.append([iter, Classify(clf=mlp_clf, folds=folds)])

    return model_perf

def SVM(folds, iter_min=1, iter_max=5, function='rbf'):
    model_perf = []
    for iter in range(iter_min, iter_max + 1):
        svm_clf = SVC(random_state=0, max_iter=iter, kernel=function)
        model_perf.append([iter, Classify(clf=svm_clf, folds=folds)])

    return model_perf


###########################################################################################################
################################   Running Section   ######################################################
###########################################################################################################

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    folds, test = Folds('data/breast-cancer-dataset.csv')
    knn_output = KNN(folds=folds)
    print(Test(knn_output, test=test))
    dt_output = DecisionTree(folds=folds)
    print(Test(dt_output, test=test))
    boosted_output = BoostedDecisionTree(folds=folds)
    print(Test(boosted_output, test=test))
    mlp_output = NeuralNet(folds=folds)
    print(Test(mlp_output, test=test))
    svm_output = SVM(folds=folds)
    print(Test(svm_output, test=test))
    svm_output = SVM(folds=folds, function='poly')
    print(Test(svm_output, test=test))