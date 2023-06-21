from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        sign_1_thresholds_and_their_err= np.apply_along_axis(
            self._find_threshold, axis=0,arr=X,labels=y,sign=1)
        print(1)
        sign_minus_1_thresholds_and_their_err = np.apply_along_axis(
            self._find_threshold, axis=0, arr=X,labels=y,sign=-1)
        print(2)
        if (np.min(sign_1_thresholds_and_their_err[1])<=np.min(
                sign_minus_1_thresholds_and_their_err[1])):
            self.j_ = np.argmin(sign_1_thresholds_and_their_err[1])
            self.threshold_ = sign_1_thresholds_and_their_err[0][self.j_]
            self.sign_=1
        else:
            self.j_ = np.argmin(sign_minus_1_thresholds_and_their_err[1])
            self.threshold_ = sign_minus_1_thresholds_and_their_err[0][self.j_]
            self.sign_=-1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_predicted = np.full(X.shape[0], -self.sign_)
        y_predicted[X[:, self.j_] >= self.threshold_] = self.sign_
        return y_predicted

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        sorted_indices_for_values = np.argsort(values)
        #this way the rows will be sorted by the values column

        values= values[sorted_indices_for_values]
        labels = labels[sorted_indices_for_values]

        min_thr_err =1
        best_thr=None
        for value in values:

            y_given_sign_indices = np.where(values >= value)[0]
            y_given_minus_sign_indices = np.where(values < value)[0]
            l_b_plus = (np.count_nonzero(labels[y_given_sign_indices] !=
                                         sign))/(labels[
                y_given_sign_indices].size )
            if labels[y_given_minus_sign_indices].size >0:
                l_b_minus = np.count_nonzero(
                    labels[y_given_minus_sign_indices] !=-sign)/(labels[
                            y_given_minus_sign_indices].size )
            else:
                l_b_minus=0
            current_thr_err = l_b_plus+l_b_minus

            if (current_thr_err<min_thr_err):
                min_thr_err =current_thr_err
                best_thr =value
        return best_thr,min_thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        from ...metrics import misclassification_error
        return misclassification_error(y,self._predict(X))