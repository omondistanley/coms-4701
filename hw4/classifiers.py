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
 
        row_train, row_test, col_train, col_test = train_test_split(feature, label, test_size=0.4, random_state=0)

        self.training_data = row_train  # Features for training
        self.training_labels = col_train  # Labels for training
        self.testing_data = row_test  # Features for testing
        self.testing_labels = col_test  # Labels for testing

        self.outputs = []
    
    def test_clf(self, clf, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        clf.fit(self.training_data, self.training_labels)

        params = clf.best_params_
        accuracy = clf.best_score_

        best_neighbor = clf.best_estimator_
        best_neighbor.fit(self.training_data, self.training_labels)

        acc = best_neighbor.score(self.testing_data, self.testing_labels)
        
        name = classifier_name
        self.outputs.append(f"{name}, {accuracy}, {acc}")
        # Use the following line to plot the results
        self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=name)
        return params, accuracy, acc
    
    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        param_grid = {
            'n_neighbors': np.arange(1,20, 2),
            'leaf_size': np.arange(5,35, 5)
            }
        
        nneighbor = KNeighborsClassifier()
        grid = GridSearchCV(nneighbor, param_grid, cv = 5)
        '''grid.fit(self.training_data, self.training_labels)

        params = grid.best_params_
        accuracy = grid.best_score_

        best_neighbor = grid.best_estimator_
        best_neighbor.fit(self.training_data, self.training_labels)

        acc = best_neighbor.score(self.testing_data, self.testing_labels)
        
        print(params)
        print(accuracy)
        print(acc)'''
        #return params, accuracy, acc
        return grid

        
    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        param_grid = {
            'C' : [0.1, 0.5, 1, 5, 10, 50, 100]
        }

        logisticModel = LogisticRegression()
        grid = GridSearchCV(logisticModel, param_grid, cv=5)
        '''grid.fit(self.training_data, self.training_labels)

        modelScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = modelScore.score(self.testing_data, self.testing_labels)

        print(params)
        print(accuracy)
        print(acc)'''
        #return params, accuracy, acc
        return grid
    
    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        param_grid = {
            'max_depth' : np.arange(1,51),
            'min_samples_split' : np.arange(2,11)
        }
        dtModel = DecisionTreeClassifier()
        grid = GridSearchCV(dtModel, param_grid, cv=5)
        '''grid.fit(self.training_data, self.training_labels)

        dtScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = dtScore.score(self.testing_data, self.testing_labels)

        print(params)
        print(accuracy)
        print(acc)'''
        #return params, accuracy, acc
        return grid

    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        #pass
        param_grid = {
            'max_depth' : [1,2,3,4,5],
            'min_samples_split': [2,3,4,5,6,7,8,9,10]
        }
        rfModel = RandomForestClassifier()
        grid = GridSearchCV(rfModel, param_grid, cv=5)
        '''grid.fit(self.training_data, self.training_labels)

        rfScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = rfScore.score(self.testing_data, self.testing_labels)

        print(params)
        print(accuracy)
        print(acc)'''
        #return params, accuracy, acc
        return grid
    
    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        param_grid = {
            'n_estimators': np.arange(10, 80, 10)
        }

        adaboostModel = AdaBoostClassifier(algorithm='SAMME')
        grid = GridSearchCV(adaboostModel, param_grid, cv=5)
        '''grid.fit(self.training_data, self.training_labels)

        adaboostScore = grid.best_estimator_
        params = grid.best_params_
        accuracy = grid.best_score_

        acc = adaboostScore.score(self.testing_data, self.testing_labels)

        print(params)
        print(accuracy)
        print(acc)'''
        #return params, accuracy, acc
        return grid
    
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
    models.classifyNearestNeighbors()
    models.test_clf(models.classifyNearestNeighbors(), 'Nearest Neighbor')
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    models.test_clf(models.classifyLogisticRegression(), 'Logistic Regression')
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    models.test_clf(models.classifyDecisionTree(), 'Decision Tree')
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    models.test_clf(models.classifyRandomForest(), 'Random Forest')
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()
    models.test_clf(models.classifyAdaBoost(), 'Adaboost')
    #print('Testing test_clf for classifyNearestNeightbors()')
    
    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)
