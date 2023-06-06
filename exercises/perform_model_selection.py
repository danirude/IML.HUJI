from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)

    indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0],
                               replace=False)


    X =X[indices]
    y =y[indices]
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], \
        X[n_samples:], y[n_samples:]


    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    indices = np.random.choice(np.arange(train_X.shape[0]), size=train_X.shape[0],
                              replace=False)
    train_X =X[indices]
    train_y =y[indices]
    ridge_train_scores=np.zeros(n_evaluations)
    ridge_validation_scores=np.zeros(n_evaluations)
    lasso_train_scores=np.zeros(n_evaluations)
    lasso_validation_scores=np.zeros(n_evaluations)
    lams_arr_ridge = np.linspace(0, 1, n_evaluations)
    lams_arr_lasso = np.linspace(0, 3, n_evaluations)
    for i in range(n_evaluations):
        ridge_train_scores[i],ridge_validation_scores[i] = cross_validate(
            RidgeRegression(lams_arr_ridge[i]),train_X,train_y,mean_square_error,5)
        lasso_train_scores[i],lasso_validation_scores[i] = cross_validate(
            Lasso(alpha=lams_arr_lasso[i],max_iter=6000),train_X,train_y,
            mean_square_error,5)

    fig = make_subplots(1, 2, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"], shared_xaxes=True)\
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$", width=750, height=300)\
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$")\
        .add_traces([go.Scatter(x=lams_arr_ridge, y=ridge_train_scores, name="Ridge Train Error"),
                    go.Scatter(x=lams_arr_ridge, y=ridge_validation_scores, name="Ridge Validation Error"),
                    go.Scatter(x=lams_arr_lasso, y=lasso_train_scores, name="Lasso Train Error"),
                    go.Scatter(x=lams_arr_lasso, y=lasso_validation_scores, name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    fig.show()


    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    




if __name__ == '__main__':
    np.random.seed(0)

    select_regularization_parameter()
