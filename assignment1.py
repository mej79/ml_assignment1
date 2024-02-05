import warnings
from matplotlib import pyplot as plt
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
        training = np.copy(folds)
        np.delete(training, fold, 0)
        training = np.concatenate(training)
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
    dataframe = genfromtxt(data_location, delimiter=',', dtype=dtype)[1:, 1:]
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
        test_error = 1 - correct_count / len(test)

        final_output.append([item[0], test_error])


    return np.asarray(final_output)

def PlotError(model, data):
    x = data[:,0]
    y = data[:,1]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_ylim(bottom=0)
    ax.set_title(model)

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

    return (model_perf)


###########################################################################################################
################################   Running Section   ######################################################
###########################################################################################################

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    folds, test = Folds('data/apple_quality.csv')
    # print(folds)
    print('Starting KNN')
    knn_output = KNN(folds=folds, k_max=25)
    knn_test_results = Test(knn_output, test=test)
    print('Done')
    # print(knn_test_results)
    print('Starting DT')
    dt_output = DecisionTree(folds=folds, depth_max=20)
    dt_test_results = Test(dt_output, test=test)
    print('Done')
    # print(dt_test_results)
    print('Starting Boosted DT')
    boosted_output = BoostedDecisionTree(folds=folds, estimators_max=25)
    boosted_test_results = Test(boosted_output, test=test)
    print('Done')
    # print(boosted_test_results)
    print('Starting Neural Nets')
    mlp_output = NeuralNet(folds=folds, iter_max=25)
    mlp_test_results = Test(mlp_output, test=test)
    print('Done')
    # print(mlp_test_results)
    print('Starting SVM')
    svm_output = SVM(folds=folds, iter_max=25)
    svm_test_results = Test(svm_output, test=test)
    print('Done')
    # print(svm_test_results)
    # svm_output = SVM(folds=folds, function='poly')
    # print(Test(svm_output, test=test))


    PlotError('KNN', knn_test_results)
    PlotError('Decision Trees', dt_test_results)
    PlotError('Boosted Decision Trees', boosted_test_results)
    PlotError('Neural Network', mlp_test_results)
    PlotError('Support Vector Machine', svm_test_results)


    plt.show()
