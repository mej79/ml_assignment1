import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from numpy import VisibleDeprecationWarning, genfromtxt
import numpy as np

np.random.seed(12)

def KNN(dataset, dtype=int, k_min=1, k_max=5, train_percent=80, num_folds=10):
    dataframe = genfromtxt(dataset, delimiter=',', dtype=dtype)[1:]

    train_index = round(len(dataframe) * train_percent / 100)

    np.random.shuffle(dataframe)

    folds = np.array_split(dataframe[:train_index], num_folds)

    model_perf = []
    for k in range(k_min, k_max + 1):
        # print(k)
        model_perf.append([k, []])
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        for fold in range(num_folds):

            # print(f"    {fold}")
            mask = np.zeros(num_folds)
            mask[fold] = True
            training = np.ma.masked_array(folds, mask=mask)
            training = np.concatenate(training[~training.mask])
            validation = folds[fold]

            X = np.array(training[:, :-1])
            y = np.array(training[:, -1:].flatten())
            

            
            knn_clf.fit(X, y)

            predictions = knn_clf.predict(validation[:, :-1])
            correct_count = 0
            for i in range(len(predictions)):
                if predictions[i] == validation[i, -1]:
                    correct_count += 1
            model_perf[k - k_min][1].append(correct_count / len(validation))

    # print(model_perf)
    best_perf = 0
    best_index = 0
    for item in model_perf:
        perf = np.sum(item[1]) / num_folds
        if perf > best_perf:
            best_perf = perf
            best_index = item[0]

    training, test = dataframe[:train_index], dataframe[train_index:]
    knn_clf = KNeighborsClassifier(n_neighbors=best_index)
    X = np.array(training[:, :-1])
    y = np.array(training[:, -1:].flatten())
    
    knn_clf.fit(X, y)

    predictions = knn_clf.predict(test[:, :-1])
    correct_count = 0
    for i in range(len(predictions)):
        if predictions[i] == test[i, -1]:
            correct_count += 1
    test_perf = correct_count / len(test)



    print(f'Best accuracy with {num_folds}-fold validation is for {best_index}NN = {test_perf}')

    return


def DecisionTree(dataset, dtype=int, depth_min=1, depth_max=10, train_percent=80, num_folds=10):
    dataframe = genfromtxt(dataset, delimiter=',', dtype=dtype)[1:]

    train_index = round(len(dataframe) * train_percent / 100)

    np.random.shuffle(dataframe)

    folds = np.array_split(dataframe[:train_index], num_folds)

    model_perf = []
    for depth in range(depth_min, depth_max + 1):
        # print(depth)
        model_perf.append([depth, []])
        dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=depth)

        for fold in range(num_folds):

            # print(f"    {fold}")
            mask = np.zeros(num_folds)
            mask[fold] = True
            training = np.ma.masked_array(folds, mask=mask)
            training = np.concatenate(training[~training.mask])
            validation = folds[fold]

            X = np.array(training[:, :-1])
            y = np.array(training[:, -1:].flatten())
            

            
            dt_clf.fit(X, y)

            predictions = dt_clf.predict(validation[:, :-1])
            correct_count = 0
            for i in range(len(predictions)):
                if predictions[i] == validation[i, -1]:
                    correct_count += 1
            model_perf[depth - depth_min][1].append(correct_count / len(validation))

    # print(model_perf)
    best_perf = 0
    best_index = 0
    for item in model_perf:
        perf = np.sum(item[1]) / num_folds
        if perf > best_perf:
            best_perf = perf
            best_index = item[0]
        print(f"Accuracy with depth of {item[0]} is {perf}")

    training, test = dataframe[:train_index], dataframe[train_index:]
    dt_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=best_index)
    X = np.array(training[:, :-1])
    y = np.array(training[:, -1:].flatten())
    
    dt_clf.fit(X, y)

    predictions = dt_clf.predict(test[:, :-1])
    correct_count = 0
    for i in range(len(predictions)):
        if predictions[i] == test[i, -1]:
            correct_count += 1
    test_perf = correct_count / len(test)



    print(f'Best accuracy with {num_folds}-fold validation is for DT with max depth of {best_index} = {test_perf}')

    return

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)
    KNN('data/breast-cancer-dataset.csv')
    DecisionTree('data/breast-cancer-dataset.csv')