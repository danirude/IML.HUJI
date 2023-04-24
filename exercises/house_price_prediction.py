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




    if(y is None):
        return preprocess_data_test(X)
    else:
        return  preprocess_data_training(X, y)




def preprocess_data_training(X: pd.DataFrame, y: Optional[pd.Series] = None):
    X=X.drop('price', axis=1)
    # print(y.head())
    X=X.assign(price=y['price'])
    # print(X.head())
    X= X.dropna()
    X=X.drop_duplicates()

    X=X.drop(['id','date','lat','long','sqft_living15','sqft_lot15'],axis=1)
    print(X.dtypes)
    # removes rows in wchich price is zero
    X = X.drop(X[X['price'] <= 0].index)



    for col_name in ['yr_renovated','sqft_basement','sqft_above','bathrooms','floors']:
        X=X[X[col_name]>=0]
    for col_name in ['sqft_lot','sqft_living','sqft_above','yr_built','zipcode']:
        X=X[X[col_name]>0]
    for col_name in ['yr_renovated','sqft_basement','sqft_above','sqft_lot','sqft_living','sqft_above','yr_built','zipcode']:
        X=X[X[col_name]%1==0]



    X=X[(X['grade'].isin(range(1,14)))&
        (X['waterfront'].isin(range(2)))&
        (X['condition'].isin(range(1,6)))&
        (X['view'].isin(range(5)))]
    X['zipcode'] = X['zipcode'].astype('category')
    X['waterfront'] = X['waterfront'].astype('category')
    X['waterfront'] = X['waterfront'].cat.add_categories(['UNKNOWN'])
    X=pd.get_dummies(data=X, drop_first=True)
    pd.set_option('display.max_columns', None)
    print(X.head())
    y= pd.DataFrame({'price': X['price']})
    X=X.drop('price', axis=1)
    return X,y
def preprocess_data_test(X: pd.DataFrame):

    y = X['price']
    X=X.drop('price', axis=1)
    # print(y.head())
    X=X.assign(price=y['price'])
    train_X_means= train_X.mean()

    X=X.drop(['id','date','lat','long','sqft_living15','sqft_lot15'],axis=1)

    X = X.drop(X[X['price'] <= 0].index)

    for col in X.columns:
        X=X.loc[X[col].isnull(),col] =  train_X_means[col]
    for col in ['yr_renovated','sqft_basement','sqft_above','bathrooms','floors']:
        X=X.loc[X[col]<0,col] =  train_X_means[col]
    for col in ['sqft_lot','sqft_living','sqft_above','yr_built','zipcode']:
        X.loc[X[col]<=0,col] =  train_X_means[col]


    X.loc[(X['grade']<1) |(X['grade']>13) ,'grade'] = train_X_means['grade']
    X.loc[(X['waterfront'] != 0) & (X['waterfront'] !=1), 'grade'] = 2
    X.loc[(X['condition']<1) |(X['condition']>5) ,'condition'] = train_X_means['condition']
    X.loc[(X['view']<1) |(X['view']>4) ,'view'] = train_X_means['view']

    for col_name in ['yr_renovated', 'sqft_basement', 'sqft_above', 'sqft_lot', 'sqft_living', 'sqft_above', 'yr_built']:
        X.loc[X[col_name] % 1 == 0, col] = train_X_means[col]

    X['zipcode'] = X['zipcode'].astype('category')

    X['waterfront'] = X['waterfront'].astype('category')
    X.loc[X['waterfront']  == '2', col] = train_X_means[col]

    X=pd.get_dummies(data=X, drop_first=True)
    X.rename(columns={'waterfront_2': 'waterfront_UNKNOWN'}, inplace=True)
    print(X.head())
    y= X.DataFrame({'price': X['price']})
    X=X.drop('price', axis=1)

    X= X.reindex(columns=train_X.columns,fill_value=0)

    return X

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
    global train_X

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

