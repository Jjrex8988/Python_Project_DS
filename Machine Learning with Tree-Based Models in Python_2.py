#--------------------------------------------------------------------------------#
## Chapter 3: Bagging and Random Forests
## Bagging is an ensemble method involving training the same algorithm many times using
## different subsets sampled from the training data. In this chapter, you'll understand
## how bagging can be used to create a tree ensemble. You'll also learn how the random
## forests algorithm can lead to further ensemble diversity through randomization at
## the level of each split in the trees forming the ensemble.

## Bagging
## Define the bagging classifier

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("indian_liver_patient_preprocessed.csv")
X = df.drop('Liver_disease', axis=1)
y = df['Liver_disease']



# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate Bagging performance
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)
## Decision Tree classifier
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc_test = accuracy_score(y_pred, y_test)
print("Test set accuracy of dt: {:.2f}".format(acc_test))

# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate acc_test
acc_test = accuracy_score(y_pred, y_test)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Out of Bag Evaluation
## Prepare the ground


# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt,
                       n_estimators=50,
                       oob_score=True,
                       random_state=1)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## OOB Score vs Test Set Score
# Fit bc to the training set
bc.fit(X_train, y_train)

# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
acc_test = accuracy_score(y_pred, y_test)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Random Forests (RF)
## Train an RF regressor

bike = pd.read_csv("bikes.csv")
X = bike.drop("cnt", axis=1)
y = bike['cnt'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)

# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
                           random_state=2)

# Fit rf to the training set
rf.fit(X_train, y_train)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Train an RF regressor

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=1)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
mse_dt = MSE(y_pred, y_test)
rmse_dt = mse_dt ** (1/2)
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))


# Predict the test set labels
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_pred, y_test) ** (1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Visualizing features importances
import matplotlib.pyplot as plt

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index=X_train.columns)

# importances = pd.Series(dt.feature_importances_, index=X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Chapter 4: Boosting
## Boosting refers to an ensemble method in which several models are trained
## sequentially with each model learning from the errors of its predecessors.
## In this chapter, you'll be introduced to the two boosting methods of AdaBoost
## and Gradient Boosting.

## Adaboost
## Define the AdaBoost classifier


df = pd.read_csv("indian_liver_patient_preprocessed.csv")
X = df.drop('Liver_disease', axis=1)
y = df['Liver_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

# ada = AdaBoostClassifier(algorithm='SAMME.R',
#                          base_estimator=
#                          DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
#                                                 max_features=None, max_leaf_nodes=None,
#                                                 min_impurity_decrease=0.0, min_impurity_split=None,
#                                                 min_samples_leaf=1, min_samples_split=2,
#                                                 min_weight_fraction_leaf=0.0,
#                                                 random_state=1,
#                                                 splitter='best'),
#                          learning_rate=1.0, n_estimators=180, random_state=1)

print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Train the AdaBoost classifier
# Fit ada to the training set
ada.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_proba = ada.predict_proba(X_test)[:, 1]
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the AdaBoost classifier
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Evaluate test-set roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print roc_auc_score
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Gradient Boosting (GB)
## Define the GB regressor

bike = pd.read_csv("bikes.csv")
X = bike.drop('cnt', axis=1)
y = bike['cnt'].values

# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate gb
gb = GradientBoostingRegressor(max_depth=4,
                               n_estimators=200,
                               random_state=2)
# gb = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
#                                learning_rate=0.1, loss='ls', max_depth=4, max_features=None,
#                                max_leaf_nodes=None, min_impurity_decrease=0.0,
#                                min_impurity_split=None, min_samples_leaf=1,
#                                min_samples_split=2, min_weight_fraction_leaf=0.0,
#                                n_estimators=200, n_iter_no_change=None,
#                                random_state=2, subsample=1.0, tol=0.0001,
#                                validation_fraction=0.1, verbose=0, warm_start=False)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Train the GB regressor
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)


# Fit gb to the training set
gb.fit(X_train, y_train)


# Predict test set labels
y_pred = gb.predict(X_test)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the GB regressor
from sklearn.metrics import mean_squared_error as MSE
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute MSE
mse_test = MSE(y_test, y_pred)

# Compute RMSE
rmse_test = mse_test ** (1/2)

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Stochastic Gradient Boosting (SGB)
## Regression with SGB
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4,
                                 subsample=0.9,
                                 max_features=0.75,
                                 n_estimators=200,
                                 random_state=2)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Train the SGB regressor
# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the SGB regressor
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute test set MSE
mse_test = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_test = mse_test ** (1/2)

# Print rmse_test
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Chapter 5: Model Tuning
## The hyperparameters of a machine learning model are parameters
## that are not learned from data. They should be set prior to fitting the model
## to the training set. In this chapter, you'll learn how to tune the hyperparameters
## of a tree-based model using grid search cross validation.

## Tuning a CART's Hyperparameters
## Tree hyperparameters

## (Q) Which of the following is not a hyperparameter of dt?
## (A) min_features

print(dt.get_params())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Set the tree's hyperparameter grid

# Define params_dt
params_dt = {
                'max_depth': [2, 3, 4],
                'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]
}
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Search for the optimal tree

dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                            max_features=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, random_state=1,
                            splitter='best')


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV



## EXTRA ##
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=25, random_state=2)

bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)


ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

from sklearn.ensemble import GradientBoostingClassifier


gb = GradientBoostingClassifier(max_depth=4,
                                n_estimators=200,
                                random_state=2)


from sklearn.ensemble import GradientBoostingClassifier


sgbr = GradientBoostingClassifier(max_depth=4,
                                  subsample=0.9,
                                  max_features=0.75,
                                  n_estimators=200,
                                  random_state=2)





# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the optimal tree

liver = pd.read_csv("indian_liver_patient_preprocessed.csv")
X = liver.drop('Liver_disease', axis=1)
y = liver['Liver_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=1)

grid_dt.fit(X_train, y_train)

# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Tuning a RF's Hyperparameters
## Random forests hyperparameters

## (Q) Which of the following is not a hyperparameter of rf?
## (A) learning_rate


bike = pd.read_csv("bikes.csv")
X = bike.drop("cnt", axis=1)
y = bike['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=1)

# rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#                            max_features='auto', max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=1, min_samples_split=2,
#                            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
#                            oob_score=False, random_state=2, verbose=0, warm_start=False)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Set the hyperparameter grid of RF
# Define the dictionary 'params_rf'
params_rf = {
    'n_estimators': [100, 350, 500],
    'max_features': ['log2', 'auto', 'sqrt'],
    'min_samples_leaf': [2, 10, 30]
}
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Search for the optimal forest
rf.fit(X_train, y_train)

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the optimal forest
from sklearn.metrics import mean_squared_error as MSE

grid_rf.fit(X_train, y_train)

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred) ** 0.5

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#



