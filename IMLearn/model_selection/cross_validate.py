from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    train_score_list = []
    validation_score_list= []
    #for now moved randm shuffle to perform model selection
    # randomly shuffle
    #indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0],
    #                           replace=False)
   # X =X[indices]
    #y =y[indices]

    folder_size = int(X.shape[0]/cv)
    for i in range(0,X.shape[0],folder_size):
        if i +folder_size>=X.shape[0]:
            #should effect reakky the step in range of for loop
            folder_size= X.shape[0]-i
        folder_indices= np.arange(i, i+folder_size)
        current_train_X =np.delete(X, folder_indices, axis=0)
        current_train_y = np.delete(y, folder_indices)

        current_validation_X = X[folder_indices]
        current_validation_y = y[folder_indices]

        estimator.fit(current_train_X,current_train_y)
        train_score_list.append(scoring(current_train_y,estimator.predict(current_train_X)))
        validation_score_list.append(scoring(current_validation_y,estimator.predict(current_validation_X)))

    train_score= sum(train_score_list)/len(train_score_list)
    validation_score= sum(validation_score_list)/len(validation_score_list)
    return train_score,validation_score


