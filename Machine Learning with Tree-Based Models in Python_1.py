#--------------------------------------------------------------------------------#
## Chapter 1: Classification and Regression Trees
## Classification and Regression Trees (CART) are a set of supervised learning
## models used for problems involving classification and regression. In this chapter,
## you'll be introduced to the CART algorithm.

## Decision Tree for Classification
## Train your first classification tree
from itertools import cycle


from mlxtend.plotting.decision_regions import get_feature_range_mask
from mlxtend.utils import check_Xy

import pandas as pd
from sklearn.model_selection import train_test_split

wbc = pd.read_csv("wbc.csv")
print(wbc.head(10))

# y = wbc['diagnosis'].values
# X = wbc.drop('diagnosis', axis=1)
# X = X[['radius_mean', 'concave points_mean']]

y = wbc['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
X = wbc[['radius_mean', 'concave points_mean']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

SEED = 1

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

# dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=6,
#                             max_features=None, max_leaf_nodes=None,
#                             min_impurity_decrease=0.0, min_impurity_split=None,
#                             min_samples_leaf=1, min_samples_split=2,
#                             min_weight_fraction_leaf=0.0, random_state=SEED,
#                             splitter='best')

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the classification tree
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Predict test set labels
y_pred = dt.predict(X_test)

# Compute test set accuracy
acc = accuracy_score(y_pred, y_test)
print("Test set accuracy: {:.2f}".format(acc))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Logistic regression vs classification tree
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, clf,
                          feature_index=None,
                          filler_feature_values=None,
                          filler_feature_ranges=None,
                          ax=None,
                          X_highlight=None,
                          res=0.02, legend=1,
                          hide_spines=True,
                          markers='s^oxv<>',
                          colors='red,blue,limegreen,gray,cyan'):
    """Plot decision regions of a classifier.

    Please note that this functions assumes that class labels are
    labeled consecutively, e.g,. 0, 1, 2, 3, 4, and 5. If you have class
    labels with integer labels > 4, you may want to provide additional colors
    and/or markers as `colors` and `markers` arguments.
    See http://matplotlib.org/examples/color/named_colors.html for more
    information.
    Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Feature Matrix.
        y : array-like, shape = [n_samples]
            True class labels.
        clf : Classifier object.
            Must have a .predict method.
        feature_index : array-like (default: (0,) for 1D, (0, 1) otherwise)
            Feature indices to use for plotting. The first index in
            `feature_index` will be on the x-axis, the second index will be
            on the y-axis.
        filler_feature_values : dict (default: None)
            Only needed for number features > 2. Dictionary of feature
            index-value pairs for the features not being plotted.
        filler_feature_ranges : dict (default: None)
            Only needed for number features > 2. Dictionary of feature
            index-value pairs for the features not being plotted. Will use the
            ranges provided to select training samples for plotting.
        ax : matplotlib.axes.Axes (default: None)
            An existing matplotlib Axes. Creates
            one if ax=None.
        X_highlight : array-like, shape = [n_samples, n_features] (default: None)
            An array with data points that are used to highlight samples in `X`.
        res : float or array-like, shape = (2,) (default: 0.02)
            Grid width. If float, same resolution is used for both the x- and
            y-axis. If array-like, the first item is used on the x-axis, the
            second is used on the y-axis. Lower values increase the resolution but
            slow down the plotting.
        hide_spines : bool (default: True)
            Hide axis spines if True.
        legend : int (default: 1)
            Integer to specify the legend location.
            No legend if legend is 0.
        markers : str (default 's^oxv<>')
            Scatterplot markers.
    colors : str (default 'red,blue,limegreen,gray,cyan')
        Comma separated list of colors.

    Returns
    ---------
    ax : matplotlib.axes.Axes object

    """

    check_Xy(X, y, y_int=True)  # Validate X and y arrays
    dim = X.shape[1]

    if ax is None:
        ax = plt.gca()

    if isinstance(res, float):
        xres, yres = res, res
    else:
        try:
            xres, yres = res
        except ValueError:
            raise ValueError('Unable to unpack res. Expecting '
                             'array-like input of length 2.')
    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_highlight must be a NumPy array or None')
        else:
            plot_testdata = False
    elif len(X_highlight.shape) < 2:
        raise ValueError('X_highlight must be a 2D array')

    if feature_index is not None:
        # Unpack and validate the feature_index values
        if dim == 1:
            raise ValueError(
                'feature_index requires more than one training feature')
        try:
            x_index, y_index = feature_index
        except ValueError:
            raise ValueError(
                'Unable to unpack feature_index. Make sure feature_index '
                'only has two dimensions.')
        try:
            X[:, x_index], X[:, y_index]
        except IndexError:
            raise IndexError(
                'feature_index values out of range. X.shape is {}, but '
                'feature_index is {}'.format(X.shape, feature_index))
    else:
        feature_index = (0, 1)
        x_index, y_index = feature_index

        # Extra input validation for higher number of training features
    if dim > 2:
        if filler_feature_values is None:
            raise ValueError('Filler values must be provided when '
                             'X has more than 2 training features.')

        if filler_feature_ranges is not None:
            if not set(filler_feature_values) == set(filler_feature_ranges):
                raise ValueError(
                    'filler_feature_values and filler_feature_ranges must '
                    'have the same keys')

        # Check that all columns in X are accounted for
        column_check = np.zeros(dim, dtype=bool)
        for idx in filler_feature_values:
            column_check[idx] = True
        for idx in feature_index:
            column_check[idx] = True
        if not all(column_check):
            missing_cols = np.argwhere(~column_check).flatten()
            raise ValueError(
                'Column(s) {} need to be accounted for in either '
                'feature_index or filler_feature_values'.format(missing_cols))

    marker_gen = cycle(list(markers))

    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')

    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]

    # Get minimum and maximum
    x_min, x_max = X[:, x_index].min() - 1, X[:, x_index].max() + 1
    if dim == 1:
        y_min, y_max = -1, 1
    else:
        y_min, y_max = X[:, y_index].min() - 1, X[:, y_index].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, xres),
                         np.arange(y_min, y_max, yres))

    if dim == 1:
        X_predict = np.array([xx.ravel()]).T
    else:
        X_grid = np.array([xx.ravel(), yy.ravel()]).T
        X_predict = np.zeros((X_grid.shape[0], dim))
        X_predict[:, x_index] = X_grid[:, 0]
        X_predict[:, y_index] = X_grid[:, 1]
        if dim > 2:
            for feature_idx in filler_feature_values:
                X_predict[:, feature_idx] = filler_feature_values[feature_idx]
    Z = clf.predict(X_predict)
    Z = Z.reshape(xx.shape)
    # Plot decisoin region
    ax.contourf(xx, yy, Z,
                alpha=0.3,
                colors=colors,
                levels=np.arange(Z.max() + 2) - 0.5)

    # ax.axis(xmin=xx.min(), xmax=xx.max(), y_min=yy.min(), y_max=yy.max())
    ax.axis(xmin=xx.min(), xmax=xx.max())

    # Scatter training data samples
    for idx, c in enumerate(np.unique(y)):
        if dim == 1:
            y_data = [0 for i in X[y == c]]
            x_data = X[y == c]
        elif dim == 2:
            y_data = X[y == c, y_index]
            x_data = X[y == c, x_index]
        elif dim > 2 and filler_feature_ranges is not None:
            class_mask = y == c
            feature_range_mask = get_feature_range_mask(
                X, filler_feature_values=filler_feature_values,
                filler_feature_ranges=filler_feature_ranges)
            y_data = X[class_mask & feature_range_mask, y_index]
            x_data = X[class_mask & feature_range_mask, x_index]
        else:
            continue

        ax.scatter(x=x_data,
                   y=y_data,
                   alpha=0.8,
                   c=colors[idx],
                   marker=next(marker_gen),
                   edgecolor='black',
                   label=c)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if dim == 1:
        ax.axes.get_yaxis().set_ticks([])

    if legend:
        if dim > 2 and filler_feature_ranges is None:
            pass
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels,
                      framealpha=0.3, scatterpoints=1, loc=legend)

    if plot_testdata:
        if dim == 1:
            x_data = X_highlight
            y_data = [0 for i in X_highlight]
        elif dim == 2:
            x_data = X_highlight[:, x_index]
            y_data = X_highlight[:, y_index]
        else:
            feature_range_mask = get_feature_range_mask(
                X_highlight, filler_feature_values=filler_feature_values,
                filler_feature_ranges=filler_feature_ranges)
            y_data = X_highlight[feature_range_mask, y_index]
            x_data = X_highlight[feature_range_mask, x_index]

        ax.scatter(x_data,
                   y_data,
                   c='',
                   edgecolor='black',
                   alpha=1.0,
                   linewidths=1,
                   marker='o',
                   s=80)

    return ax


def plot_labeled_decision_regions(X, y, models):
    '''
    Function producing a scatter plot of the instances contained
    in the 2D dataset (X,y) along with the decision
    regions of two trained classification models contained in the
    list 'models'.

    Parameters
    ----------
    X: pandas DataFrame corresponding to two numerical features
    y: pandas Series corresponding the class labels
    models: list containing two trained classifiers

    '''
    if len(models) != 2:
        raise Exception('''
        Models should be a list containing only two trained classifiers.
        ''')
    if not isinstance(X, pd.DataFrame):
        raise Exception('''
        X has to be a pandas DataFrame with two numerical features.
        ''')
    if not isinstance(y, pd.Series):
        raise Exception('''
        y has to be a pandas Series corresponding to the labels.
        ''')
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.7), sharey=True)
    for i, model in enumerate(models):
        plot_decision_regions(X.values, y.values, model, legend=2, ax=ax[i])
        ax[i].set_title(model.__class__.__name__)
        ax[i].set_xlabel(X.columns[0])
        if i == 0:
            ax[i].set_ylabel(X.columns[1])
        ax[i].set_ylim(X.values[:, 1].min(), X.values[:, 1].max())
        ax[i].set_xlim(X.values[:, 0].min(), X.values[:, 0].max())
    plt.tight_layout()
    plt.show()


# Import LogisticRegression from sklearn.linear_model
from sklearn.linear_model import LogisticRegression

# Instatiate logreg
logreg = LogisticRegression(random_state=1)


# Fit logreg to the training set
logreg.fit(X_train, y_train)

# Define a list called clfs containing the two classifiers logreg and dt
clfs = [logreg, dt]

# Review the decision regions of the two classifiers
plot_labeled_decision_regions(X_test, y_test, clfs)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Classification tree Learning
## Growing a classification tree

## (Q) In the video, you saw that the growth of an unconstrained classification
## tree followed a few simple rules. Which of the following is not one of these rules?

## (A) When an internal node is split, the split is performed in such a way so
## that information gain is minimized.
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
X = wbc[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
         'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
         'fractal_dimension_se', 'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst', 'smoothness_worst',
         'compactness_worst', 'concavity_worst', 'concave points_worst',
         'symmetry_worst', 'fractal_dimension_worst']]
y = wbc['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


## Using entropy as a criterion
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
# dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)


dt_entropy = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=8,
                                    max_features=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0, random_state=1,
                                    splitter='best')

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)


y_pred_entropy = dt_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_pred_entropy, y_test)

print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Entropy vs Gini index
# dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)

dt_gini = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=8,
                                 max_features=None, max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, random_state=1,
                                 splitter='best')


dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
accuracy_gini = accuracy_score(y_pred_gini, y_test)


# Import accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_pred, y_test)

# Print accuracy_entropy
print('Accuracy achieved by using entropy: ', accuracy_entropy)

# Print accuracy_gini
print('Accuracy achieved by using the gini index: ', accuracy_gini)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Decision tree for regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE

auto = pd.read_csv("auto.csv")
y = auto['mpg'].values
X = auto.drop('mpg', axis=1)
X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)
# Instantiate dt
# dt = DecisionTreeRegressor(max_depth=8,
#                            min_samples_leaf=0.13,
#                            random_state=3)

dt = DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, min_samples_leaf=0.13,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           random_state=3, splitter='best')

# Fit dt to the training set
dt.fit(X_train, y_train)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the regression tree
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Compute y_pred
y_pred = dt.predict(X_test)

# Compute mse_dt
mse_dt = MSE(y_pred, y_test)

# Compute rmse_dt
rmse_dt = mse_dt ** (1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Linear regression vs regression tree
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict test set labels
y_pred_lr = lr.predict(X_test)

# Compute mse_lr
mse_lr = MSE(y_pred_lr, y_test)

# Compute rmse_lr
rmse_lr = mse_lr ** (1/2)

# Print rmse_lr
print('Linear Regression test set RMSE: {:.2f}'.format(rmse_lr))

# Print rmse_dt
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_dt))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Chapter 2: The Bias-Variance Tradeoff
## The bias-variance tradeoff is one of the fundamental concepts in supervised
## machine learning. In this chapter, you'll understand how to diagnose the
## problems of overfitting and underfitting. You'll also be introduced to the
## concept of ensembling where the predictions of several models are aggregated
## to produce predictions that are more robust.

## Generalization Error
## (Q) Which of the following correctly describes the relationship between 's complexity and
## 's bias and variance terms?

## (A) As the complexity of F hat increases, the bias term decreases while the variance term increases
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Overfitting and underfitting

## (A) Which of the following statements is true?

## (B) B suffers from high bias and underfits the training set.
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Diagnose bias and variance problems
## Instantiate the model

auto = pd.read_csv("auto.csv")
y = auto['mpg'].values
X = auto.drop('mpg', axis=1)
X = pd.get_dummies(X)

# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the 10-fold CV error
from sklearn.model_selection import cross_val_score

# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10,
                                  scoring='neg_mean_squared_error',
                                  n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(0.5)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate the training error
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# y_pred_test = dt.predict(X_test)
# print('CV MSE: {:.2f}'.format(MSE_CV_scores.mean()))
# print('Train MSE: {:.2f}'.format(MSE(y_train, y_pred_train)))
# print('Test MSE: {:.2f}'.format(MSE(y_test, y_pred_test)))

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## High bias or high variance?

## Here baseline_RMSE serves as the baseline RMSE above which a model is
## considered to be underfitting and below which the model is considered 'good enough'.

## (Q) Does dt suffer from a high bias or a high variance problem?
## (A) dt suffers from high bias because RMSE_CV  RMSE_train and both scores are greater than baseline_RMSE.
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Ensemble Learning
## Define the ensemble

# wbc = pd.read_csv("wbc.csv")
#
# X = wbc.drop('diagnosis', axis=1);
# X = X[['radius_mean', 'concave points_mean']]
#
# # X = wbc[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
# #          'smoothness_mean', 'compactness_mean', 'concavity_mean',
# #          'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
# #          'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
# #          'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
# #          'fractal_dimension_se', 'radius_worst', 'texture_worst',
# #          'perimeter_worst', 'area_worst', 'smoothness_worst',
# #          'compactness_worst', 'concavity_worst', 'concave points_worst',
# #          'symmetry_worst', 'fractal_dimension_worst']]
#
# y = wbc['diagnosis'].apply(lambda x: 1 if x == "M" else 0)


liver = pd.read_csv("indian_liver_patient_preprocessed.csv", index_col=0)
y = liver['Liver_disease']
X = liver.drop('Liver_disease', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score


# Set seed for reproducibility
SEED = 1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Evaluate individual classifiers
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:
    # Fit clf to the training set
    clf.fit(X_train, y_train)

    # Predict y_pred
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_pred, y_test)

    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))


# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10,
                                  scoring='neg_mean_squared_error',
                                  n_jobs=-1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())** (0.5)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Better performance with a Voting Classifier
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_pred, y_test)
print('Voting Classifier: {:.3f}'.format(accuracy))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#