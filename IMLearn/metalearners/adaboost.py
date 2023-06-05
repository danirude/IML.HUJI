import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn

np.random.seed(1)

class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """


        samples_num = X.shape[0]

        self.D_ = np.full(samples_num, 1 / samples_num)
        self.models_ = []
        self.weights_ =[]

        for t in range(self.iterations_):

            sampled_rows = np.random.choice(samples_num, size=samples_num,
                                            p=self.D_)
            random_sample = X[sampled_rows]
            self.models_ = self.models_.append(self.wl_().fit(random_sample,y))
            y_predicted= self.models_[t].predict(X)
            error =  np.dot(self.D_, np.where(y!=y_predicted, 1, 0))
            current_weight = 0.5* np.log((1-error)/error)
            self.weights_.append(current_weight)

            #create new D and normalize it
            self.D_ =self.D_ * np.exp(-y*current_weight*y_predicted)
            self.D_ = self.D_ /(np.sum(self.D_ ))


    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        return self.partial_loss( X, y, T=self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        prediction_matrix =  np.empty((X.shape[0], 0))
        for t in range(min(T,self.iterations_)):
            current_weighted_prediction= self.weights_[t]*\
                                         self.models_[t].predict(X)
            prediction_matrix = np.append(prediction_matrix,
                                          current_weighted_prediction, axis=1)

        return np.sign(np.sum(prediction_matrix,axis=0))

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        from ..metrics import misclassification_error
        return misclassification_error(y,self._predict(X))
