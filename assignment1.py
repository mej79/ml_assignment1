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
    training = folds[0]
    X = np.array(training[:, :-1])
    y = np.array(training[:, -1:].flatten())
    clf.fit(X, y)
        
    return clf

def Folds(dataframe, train_percent=80, num_folds=10):
    if train_percent > 80:
        raise ValueError('train_percent should be no greater than 80 to account for at least 20% test data')
    train_index = round(len(dataframe) * train_percent / 100)
    test_index = round(len(dataframe) * 20 / 100)
    training, test = dataframe[:train_index], dataframe[-test_index:]

    np.random.shuffle(dataframe)

    return (np.array_split(training, num_folds), test)

def TrainHyperparameters(folds, test, knn=True, dt=True, boosted=True, mlp=True, svm=True, kernel_fns=['rbf', 'poly']):
    results = []
    if knn:
        print('Starting KNN')
        knn_output = KNN(folds=folds, k_max=50, to_print=False)
        results.append(['K Nearest Neighbors', 'Neighbors',  Test(knn_output, test=test), Test(knn_output, test=np.concatenate(folds))])
        print('Done')
    
    if dt:
        print('Starting DT')
        dt_output = DecisionTree(folds=folds, depth_max=20, to_print=False)
        results.append(['Decision Trees', 'Depth',  Test(dt_output, test=test), Test(dt_output, test=np.concatenate(folds))])
        print('Done')
    
    if boosted:
        print('Starting Boosted DT')
        boosted_output = BoostedDecisionTree(folds=folds, estimators_max=40, to_print=False)
        results.append(['Boosted Decision Trees', 'Estimators',  Test(boosted_output, test=test), Test(boosted_output, test=np.concatenate(folds))])
        print('Done')
    
    if mlp:
        print('Starting Neural Nets')
        mlp_output = NeuralNet(folds=folds, iter_max=60, to_print=False)
        results.append(['Multilevel Perceptron', 'Iterations',  Test(mlp_output, test=test), Test(mlp_output, test=np.concatenate(folds))])
        print('Done')
    
    if svm:
        for fn in kernel_fns:
            print(f'Starting SVM- {fn}')
            svm_output = SVM(folds=folds, iter_max=600, to_print=False, function=fn)
            print('Done Training')
            results.append([f'Support Vector Machine- {fn}', 'Iterations', Test(svm_output, test=test), Test(svm_output, test=np.concatenate(folds))])
            
        
    return np.asarray(results)

def TrainDataForTraining(dataframe, test, knn=1, dt=1, boosted=1, mlp=1, svm=[['rbf', 1], ['poly', 1]]):
    results = []
    if knn:
        print('Starting KNN')
        knn_output = KNN_TrainingPercentage(dataframe=dataframe, k=knn, to_print=False)
        results.append(['K Nearest Neighbors', 'Training Data (%)',  Test(knn_output, test=test), Test(knn_output, test=np.concatenate(folds))])
        print('Done')
    
    if dt:
        print('Starting DT')
        dt_output = DecisionTree_TrainingPercentage(dataframe=dataframe, depth=dt, to_print=False)
        results.append(['Decision Trees', 'Training Data (%)',  Test(dt_output, test=test), Test(dt_output, test=np.concatenate(folds))])
        print('Done')
    
    if boosted:
        print('Starting Boosted DT')
        boosted_output = BoostedDecisionTree_TrainingPercentage(dataframe=dataframe, n_estimators=boosted, to_print=False)
        results.append(['Boosted Decision Trees', 'Training Data (%)',  Test(boosted_output, test=test), Test(boosted_output, test=np.concatenate(folds))])
        print('Done')
    
    if mlp:
        print('Starting Neural Nets')
        mlp_output = NeuralNet_TrainingPercentage(dataframe=dataframe, iter=mlp, to_print=False)
        results.append(['Multilevel Perceptron', 'Training Data (%)',  Test(mlp_output, test=test), Test(mlp_output, test=np.concatenate(folds))])
        print('Done')
    
    if svm:
        for fn in svm:
            print(f'Starting SVM- {fn[0]}')
            svm_output = SVM_TrainingPercentage(dataframe=dataframe, iter=fn[1], to_print=False)
            print('Done Training')
            results.append([f'Support Vector Machine- {fn[0]}', 'Training Data (%)', Test(svm_output, test=test), Test(svm_output, test=np.concatenate(folds))])
            
        
    return np.asarray(results)

def Test(model_perf, test):

    final_output = []
    for item in model_perf:
        clf = item[1]

        predictions = clf.predict(test[:, :-1])

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

def KNN_TrainingPercentage(dataframe, k=1, min_perc=10, max_perc=80, to_print=False):
    model_perf = []
    for perc in range(min_perc, max_perc + 1):
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        folds, test = Folds(dataframe, num_folds=1, train_percent=perc)
        if to_print:
            print(f'perc = {perc}')
        model_perf.append([perc, Classify(knn_clf, folds)])

    return model_perf

def DecisionTree(folds, depth_min=1, depth_max=5, to_print=False):
    model_perf = []
    for depth in range(depth_min, depth_max + 1):
        if to_print:
            print(f'depth = {depth}')
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        model_perf.append([depth, Classify(clf=dt_clf, folds=folds)])

    return model_perf

def DecisionTree_TrainingPercentage(dataframe, depth=1, min_perc=10, max_perc=80, to_print=False):
    model_perf = []
    for perc in range(min_perc, max_perc + 1):
        folds, test = Folds(dataframe, num_folds=1, train_percent=perc)
        if to_print:
            print(f'perc = {perc}')
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        model_perf.append([perc, Classify(clf=dt_clf, folds=folds)])

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

def BoostedDecisionTree_TrainingPercentage(dataframe, n_estimators=1, min_perc=10, max_perc=80, depth=1, learning_rate = 1.0, to_print=False):
    model_perf = []
    for perc in range(min_perc, max_perc + 1):
        folds, test = Folds(dataframe, num_folds=1, train_percent=perc)
        if to_print:
            print(f'perc = {perc}')
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)
        boosted_clf = AdaBoostClassifier(dt_clf, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
        model_perf.append([perc, Classify(clf=boosted_clf, folds=folds)])

    return model_perf

def NeuralNet(folds, iter_min=1, iter_max=5, to_print=False):
    model_perf = []
    for iter in range(iter_min, iter_max + 1):
        if to_print:
            print(f'iteration: {iter}')
        mlp_clf = MLPClassifier(random_state=0, max_iter=iter)
        model_perf.append([iter, Classify(clf=mlp_clf, folds=folds)])

    return model_perf

def NeuralNet_TrainingPercentage(dataframe, iter=1, min_perc=10, max_perc=80, to_print=False):
    model_perf = []
    for perc in range(min_perc, max_perc + 1):
        folds, test = Folds(dataframe, num_folds=1, train_percent=perc)
        if to_print:
            print(f'perc: {perc}')
        mlp_clf = MLPClassifier(random_state=0, max_iter=iter)
        model_perf.append([perc, Classify(clf=mlp_clf, folds=folds)])

    return model_perf

def SVM(folds, iter_min=1, iter_max=5, function='rbf', to_print=False):
    model_perf = []
    for iter in range(iter_min, iter_max + 1):
        if to_print:
            print(f'iteration: {iter}')
        svm_clf = SVC(random_state=0, max_iter=iter, kernel=function)
        model_perf.append([iter, Classify(clf=svm_clf, folds=folds)])

    return (model_perf)

def SVM_TrainingPercentage(dataframe, iter=1, min_perc=10, max_perc=80, function='rbf', to_print=False):
    model_perf = []
    for perc in range(min_perc, max_perc + 1):
        folds, test = Folds(dataframe, num_folds=1, train_percent=perc)
        if to_print:
            print(f'perc: {perc}')
        svm_clf = SVC(random_state=0, max_iter=iter, kernel=function)
        model_perf.append([perc, Classify(clf=svm_clf, folds=folds)])

    return (model_perf)


###########################################################################################################
################################   Running Section   ######################################################
###########################################################################################################

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    # First Dataset

    # First, run each algorithm with small training percent on relevant hyperparameters to judge what values are optimal
    dataframe = genfromtxt('data/diabetes_binary.csv', delimiter=',', dtype=int)[1:, 1:]
    folds, test = Folds(dataframe, num_folds=1, train_percent=20)
    training_results = TrainHyperparameters(folds, test)
    # training_results = TrainHyperparameters(folds, knn=False, dt=False, boosted=False, mlp=True, svm=False, kernel_fns=['poly'])


    # Plot the model complexity graphs for the various hyperparameters
    for result in training_results:
        Plot(data=result)
        # Print info about minimum error hyperparameter values
        minimum_error = result[2][np.argmin(result[2][:,1])]
        print(f'For {result[0]}, lowest test error of {minimum_error[1]} found with hyperparameter of {minimum_error[0]} {result[1]}')
    plt.show()


    # Second, run each algorithm with the optimized hyperparameters on different amounts of training data
    percentage_training_results = []
    dataframe = genfromtxt('data/diabetes_binary.csv', delimiter=',', dtype=int)[1:, 1:]

    folds, test = Folds(dataframe, num_folds=1)
    percentage_training_results = TrainDataForTraining(dataframe, test, knn=20, dt=4, boosted=18, mlp=59, svm=[['rbf', 361], ['poly', 294]])

    # Plot the model complexity graphs for the various hyperparameters
    for result in percentage_training_results:
        Plot(data=result)
        # Print info about minimum error hyperparameter values
        minimum_error = result[2][np.argmin(result[2][:,1])]
        print(f'For {result[0]}, lowest test error of {minimum_error[1]} found with hyperparameter of {minimum_error[0]} {result[1]}')
    plt.show()



    # Second Dataset

    # First, run each algorithm with small training percent on relevant hyperparameters to judge what values are optimal
    dataframe = genfromtxt('data/apple_quality.csv', delimiter=',', dtype=int)[1:, 1:]
    folds, test = Folds(dataframe, num_folds=1, train_percent=20)
    training_results = TrainHyperparameters(folds, test)
    # training_results = TrainHyperparameters(folds, knn=False, dt=False, boosted=False, mlp=True, svm=False, kernel_fns=['poly'])


    # Plot the model complexity graphs for the various hyperparameters
    for result in training_results:
        Plot(data=result)
        # Print info about minimum error hyperparameter values
        minimum_error = result[2][np.argmin(result[2][:,1])]
        print(f'For {result[0]}, lowest test error of {minimum_error[1]} found with hyperparameter of {minimum_error[0]} {result[1]}')
    plt.show()


    # Second, run each algorithm with the optimized hyperparameters on different amounts of training data
    percentage_training_results = []
    dataframe = genfromtxt('data/apple_quality.csv', delimiter=',', dtype=int)[1:, 1:]

    folds, test = Folds(dataframe, num_folds=1)
    percentage_training_results = TrainDataForTraining(dataframe, test, knn=34, dt=20, boosted=22, mlp=56, svm=[['rbf', 187], ['poly', 565]])

    # Plot the model complexity graphs for the various hyperparameters
    for result in percentage_training_results:
        Plot(data=result)
        # Print info about minimum error hyperparameter values
        minimum_error = result[2][np.argmin(result[2][:,1])]
        print(f'For {result[0]}, lowest test error of {minimum_error[1]} found with hyperparameter of {minimum_error[0]} {result[1]}')
    plt.show()
