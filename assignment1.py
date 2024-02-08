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

def Train(folds, knn=True, dt=True, boosted=True, mlp=True, svm=True, kernel_fns=['rbf', 'poly']):
    results = []
    if knn:
        print('Starting KNN')
        knn_output = KNN(folds=folds, k_max=75, to_print=True)
        results.append(['K Nearest Neighbors', 'Neighbors',  Test(knn_output, test=test), Test(knn_output, test=np.concatenate(folds))])
        print('Done')
    
    if dt:
        print('Starting DT')
        dt_output = DecisionTree(folds=folds, depth_max=80, to_print=True)
        # dt_test_results = Test(dt_output, test=test)
        results.append(['Decision Trees', 'Depth',  Test(dt_output, test=test), Test(dt_output, test=np.concatenate(folds))])
        print('Done')
    
    if boosted:
        print('Starting Boosted DT')
        boosted_output = BoostedDecisionTree(folds=folds, estimators_max=75, to_print=True)
        # boosted_test_results = Test(boosted_output, test=test)
        results.append(['Boosted Decision Trees', 'Estimators',  Test(boosted_output, test=test), Test(boosted_output, test=np.concatenate(folds))])
        print('Done')
    
    if mlp:
        print('Starting Neural Nets')
        mlp_output = NeuralNet(folds=folds, iter_max=60, to_print=True)
        # mlp_test_results = Test(mlp_output, test=test)
        results.append(['Multilevel Perceptron', 'Iterations',  Test(mlp_output, test=test), Test(mlp_output, test=np.concatenate(folds))])
        print('Done')
    
    if svm:
        for fn in kernel_fns:
            print(f'Starting SVM- {fn}')
            svm_output = SVM(folds=folds, iter_max=100, to_print=True, function=fn)
            # svm_test_results = Test(svm_output, test=test)
            results.append([f'Support Vector Machine- {fn}', 'Iterations', Test(svm_output, test=test), Test(svm_output, test=np.concatenate(folds))])
            print('Done')
        
    return np.asarray(results)

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

def Plot(data):
    name = data[0]
    xlabel = data[1]
    y1 = data[2][:,1]
    x1 = data[2][:,0]
    x2 = data[3][:,0]
    y2 = data[3][:,1]

    fig, ax = plt.subplots()
    ax.plot(x1, y1, label='Test')
    ax.plot(x2, y2, label='Train')
    ax.set_ylim(bottom=0)
    ax.set(title=name, xlabel=xlabel, ylabel='Error')
    ax.legend(title='Data')

def KNN(folds, k_min=1, k_max=5, to_print=False):
    model_perf = []
    for k in range(k_min, k_max + 1):
        if to_print:
            print(f'k = {k}')
        knn_clf = KNeighborsClassifier(n_neighbors=k)

        model_perf.append([k, Classify(clf=knn_clf, folds=folds)])

    return model_perf

def DecisionTree(folds, depth_min=1, depth_max=5, to_print=False):
    model_perf = []
    for depth in range(depth_min, depth_max + 1):
        if to_print:
            print(f'depth = {depth}')
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        model_perf.append([depth, Classify(clf=dt_clf, folds=folds)])

    return model_perf

def BoostedDecisionTree(folds, estimators_min=1, estimators_max=5, depth=1, learning_rate = 1.0, to_print=False):
    model_perf = []
    for n_estimators in range(estimators_min, estimators_max + 1):
        if to_print:
            print(f'n_estimators = {n_estimators}')
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        boosted_clf = AdaBoostClassifier(dt_clf, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
        model_perf.append([n_estimators, Classify(clf=boosted_clf, folds=folds)])

    return model_perf

def NeuralNet(folds, iter_min=1, iter_max=5, to_print=False):
    model_perf = []
    for iter in range(iter_min, iter_max + 1):
        if to_print:
            print(f'iteration: {iter}')
        mlp_clf = MLPClassifier(random_state=0, max_iter=iter)
        model_perf.append([iter, Classify(clf=mlp_clf, folds=folds)])

    return model_perf

def SVM(folds, iter_min=1, iter_max=5, function='rbf', to_print=False):
    model_perf = []
    for iter in range(iter_min, iter_max + 1):
        if to_print:
            print(f'iteration: {iter}')
        svm_clf = SVC(random_state=0, max_iter=iter, kernel=function)
        model_perf.append([iter, Classify(clf=svm_clf, folds=folds)])

    return (model_perf)


###########################################################################################################
################################   Running Section   ######################################################
###########################################################################################################

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    folds, test = Folds('data/diabetes_binary.csv')
    training_results = Train(folds)

    for result in training_results:
        Plot(data=result)

    plt.show()
