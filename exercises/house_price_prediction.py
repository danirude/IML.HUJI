from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """



    X=X.drop('price', axis=1)
    # print(y.head())
    X=X.assign(price=y['price'])
    # print(X.head())
    X= X.dropna()
    X=X.drop_duplicates()
    #remove
    # rows in wchich price is zero
    X = X.drop(X[X['price'] == 0].index)
    X=X.drop(['id','date','lat','long','sqft_living15','sqft_lot15'],axis=1)

    X['zipcode'] = X['zipcode'].astype('category')
    X['zipcode'] = X['zipcode'].cat.add_categories(['OTHER'])
    X['waterfront'] = X['waterfront'].astype('category')
    X['waterfront'] = X['waterfront'].cat.add_categories(['UNKNOWN'])
    X=pd.get_dummies(data=X, drop_first=True)
    pd.set_option('display.max_columns', None)
    print(X.head())

    return X,y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """




if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    y= pd.DataFrame({'price': df['price']})

    print(df.head())
    print(y.head())

    # Question 1 - split data into train and test sets
    train_X,train_y,test_X,test_y= split_train_test(df,y)

    print(train_X.head())
    print(train_X.dtypes)
    print(train_y.head())
    print(test_X.head())
    print(test_y.head())
    # Question 2 - Preprocessing of housing prices dataset
    train_X,train_y =preprocess_data(train_X,train_y)

    # Question 3 - Feature evaluation with respect to response


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

