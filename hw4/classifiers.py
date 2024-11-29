from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifiers():
    def __init__(self,data):
        ''' 
        TODO: Write code to convert the given pandas dataframe into training and testing data 
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry 
        '''
        
        feature = data.drop(columns=['label']).values
        label = data['label'].values
 
        row_train, row_test, col_train, col_test = train_test_split(feature, label, test_size= 0.4, random_state= 0)

        self.training_data = row_train
        self.training_labels = row_test
        self.testing_data = col_train
        self.testing_labels = col_test
        self.outputs = []
    
    def test_clf(self, clf, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        pass
        # Use the following line to plot the results
        # self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=name)

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        param_grid = {
            'n_neighbors': np.arange(1,20, 2),
            'leaf_size': np.arange(5,35, 5)
            }
        
        nneighbor = KNeighborsClassifier()
        grid = GridSearchCV(nneighbor, param_grid, cv = 5)
        grid.fit(self.training_data, self.testing_data)

        params = grid.best_params_
        accuracy = grid.best_score_

        best_neighbor = grid.best_estimator_
        best_neighbor.fit(self.training_data, self.testing_data)

        acc = best_neighbor.score(self.training_labels, self.testing_labels)

        return params, accuracy, acc

        
    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        param_grid = {
            'C' : [0.1, 0.5, 1, 5, 10, 50, 100]
        }

        logisticModel = LogisticRegression()
        grid = GridSearchCV(logisticModel, param_grid, cv=5)
        grid.fit(self.training_data, self.testing_data)

        modelScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = modelScore.score(self.training_labels, self.testing_labels)

        return params, accuracy, acc
    
    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        param_grid = {
            'max_depth' : np.arange(1,51),
            'min_samples_split' : np.arange(2,11)
        }
        dtModel = DecisionTreeClassifier()
        grid = GridSearchCV(dtModel, param_grid, cv=5)
        grid.fit(self.training_data, self.testing_data)

        dtScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = dtScore.score(self.training_labels, self.testing_labels)

        return params, accuracy, acc

    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        #pass
        param_grid = {
            'max_depth' : [1,2,3,4,5],
            'min_samples_split': [2,3,4,5,6,7,8,9,10]
        }
        rfModel = RandomForestClassifier()
        grid = GridSearchCV(rfModel, param_grid, cv=5)
        grid.fit(self.training_data, self.testing_data)

        rfScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = rfScore.score(self.training_labels, self.testing_labels)

        return params, accuracy, acc
    
    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        param_grid = {
            'n_estimators': np.arange(10, 80, 10)
        }

        adaboostModel = AdaBoostClassifier(algorithm='SAMME')
        grid = GridSearchCV(adaboostModel, param_grid, cv=5)
        grid.fit(self.training_data, self.testing_data)

        adaboostScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = adaboostScore.score(self.training_labels, self.testing_labels)

        return params, accuracy, acc
    
    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images
        # plt.savefig(f'{classifier_name}.png')
        plt.show()

    
if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    params, acc, acc2  = models.classifyNearestNeighbors()
    print(params)
    print(acc)
    print(acc2)
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    params, acc, acc2  = models.classifyLogisticRegression()
    print(params)
    print(acc)
    print(acc2)
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    params, acc, acc2  = models.classifyDecisionTree()
    print(params)
    print(acc)
    print(acc2)
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    params, acc, acc2  = models.classifyRandomForest()
    print(params)
    print(acc)
    print(acc2)
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    params, acc, acc2  = models.classifyAdaBoost()
    print(params)
    print(acc)
    print(acc2)
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)
