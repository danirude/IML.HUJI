from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from plotly.subplots import make_subplots

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



    X=X.assign(price=y)
    X= X.dropna()
    X=X.drop_duplicates()

    X=X.drop(['id','date','lat','long','sqft_living15','sqft_lot15'],axis=1)

    for col_name in ['yr_renovated','sqft_basement','sqft_above','bathrooms','floors','bedrooms']:
        X=X[X[col_name]>=0]
    for col_name in ['sqft_lot','sqft_living','sqft_above','yr_built','zipcode']:
        X=X[X[col_name]>0]
    for col_name in ['yr_renovated','sqft_basement','sqft_above','sqft_lot','sqft_living','sqft_above',
                     'yr_built','zipcode']:
        X=X[X[col_name]%1==0]


    X=X[X['sqft_living']>=25]

    X=X[X['bathrooms']<=10]
    X=X[X['floors']<=10]
    X=X[X['bedrooms']<=10]

    X=X[(X['grade'].isin(range(1,14)))&
        (X['waterfront'].isin(range(2)))&
        (X['condition'].isin(range(1,6)))&
        (X['view'].isin(range(5)))]
    X['zipcode'] = X['zipcode'].astype('category')
    X['zipcode'] = X['zipcode'].cat.add_categories(['0'])
    X['waterfront'] = X['waterfront'].astype('category')
    X['waterfront'] = X['waterfront'].cat.add_categories(['UNKNOWN'])

    X=pd.get_dummies(data=X, drop_first=True)
    pd.set_option('display.max_columns', None)
    y= X['price']
    X=X.drop('price', axis=1)
    return X,y
def preprocess_data_test(X: pd.DataFrame):



    train_X_means= train_X.mean()

    X=X.drop(['id','date','lat','long','sqft_living15','sqft_lot15'],axis=1)

    for col in X.columns:
        if((col =='waterfront') | (col =='zipcode') |(col =='price')):
            continue
        X.loc[X[col].isnull(),col] =  train_X_means[col]
    for col in ['yr_renovated','sqft_basement','sqft_above','bathrooms','floors','bedrooms']:
        X.loc[X[col]<0,col] =  train_X_means[col]
    for col in ['sqft_lot','sqft_living','sqft_above','yr_built']:
        X.loc[X[col]<=0,col] =  train_X_means[col]


    X.loc[(X['grade']<1) |(X['grade']>13) ,'grade'] = train_X_means['grade']
    X.loc[(X['waterfront'].isnull()) | ((X['waterfront'] != 0) & (X['waterfront'] !=1)), 'waterfront'] = 2
    X.loc[(X['condition']<1) |(X['condition']>5) ,'condition'] = train_X_means['condition']
    X.loc[(X['view']<1) |(X['view']>4) ,'view'] = train_X_means['view']


    X.loc[X['sqft_living']<25 ,'sqft_living'] = train_X_means['sqft_living']
    X.loc[X['bathrooms']>10 ,'bathrooms'] = train_X_means['bathrooms']
    X.loc[X['floors']>10 ,'floors'] = train_X_means['floors']
    X.loc[X['bedrooms']>10 ,'bedrooms'] = train_X_means['bedrooms']



    for col_name in ['yr_renovated', 'sqft_basement', 'sqft_above',
                     'sqft_lot', 'sqft_living', 'sqft_above', 'yr_built']:
        X.loc[X[col_name] % 1 != 0, col] = train_X_means[col]


    X.loc[(X['zipcode']<0)|(X['zipcode'] % 1 != 0), 'zipcode'] = 0
    X['zipcode'] = X['zipcode'].astype('category')

    X['waterfront'] = X['waterfront'].astype('category')

    X['waterfront'] = X['waterfront'].cat.add_categories(['UNKNOWN'])
    X.loc[X['waterfront'] == '2', 'waterfront'] = 'UNKNOWN'

    X=pd.get_dummies(data=X, drop_first=True)


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

    zip_cols = [col for col in X.columns if 'zipcode' in col]
    waterfront_cols  = [col for col in X.columns if 'waterfront' in col]
    # drop the dummy variable columns
    X = X.drop(zip_cols, axis=1)
    X = X.drop(waterfront_cols , axis=1)
    for feature in X.columns:
        feature_vector = X[feature]
        response_vector =y
        pears_corr = np.cov(feature_vector, response_vector)[0, 1] / (np.std(feature_vector)*
                                                                               np.std(response_vector))

        fig = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=feature_vector, y=response_vector,
                                                                   mode='markers', marker=dict(color="blue"))])

        fig.update_layout(title_text= f"Pearson Correlation between {feature} and price is : {pears_corr}").\
            update_yaxes(title_text="price", secondary_y=False, row=1,col=1).update_xaxes \
            (showgrid=False, title_text=f"{feature}", row=1, col=1)


        fig.write_image (output_path+ f"scatterPlot{feature}AndPrice.png")


if __name__ == '__main__':

    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    df=df.drop(df[df['price'].isnull()].index)
    df=df.drop(df[df['price'] <= 0].index)
    y= df['price']
    X = df.drop('price', axis=1)

    print("Linear Regression Model")

    # Question 1 - split data into train and test sets
    global train_X

    train_X,train_y,test_X,test_y= split_train_test(df,y)


    # Question 2 - Preprocessing of housing prices dataset

    train_X,train_y =preprocess_data(train_X,train_y)
    test_X =preprocess_data(test_X)


    print("Q2")
    print("*I decided not to keep the features : id,date,lat,long,sqft_living15 and sqft_lot15."
          "I kept the rest of the features. \n I didn't keep id because in my opinion it is a "
          "categorical value that is too specific because there cant cant be two houses with the same id. \n"
          "I didn't keep 'lat'  because in my opinion it is too generic and doesn't tell me enough about "
          "the house's location.I didn't keep 'long' for the same reason.\n I didn't keep date,sqft_living15  and "
          "sqft_lot15 because I dont think that they are relevant enough to the price of the house.")

    print("*zipcode and waterfront are categorical features.I treated them by using dummy variable encoding.\nI "
          "also added to zipcode a categorical variable 0 and added a categorical variable UNKNOWN in order"
          " to deal with problematic values in\nthe zipcode and waterfront columns of the test dataset.")
    print("* I didn't create any other features.")

    print("*For features that are not categorical, I treated invalid/missing values be replacing them with the mean "
          "of the feature in the training data set.\nFor features that are categorical, if the feature is zipcode I "
          "replaced the value with 0, and if the feature is waterfront then I replaced the value with UNKNOWN.")
    # Question 3 - Feature evaluation with respect to response
    print("Q3")
    feature_evaluation(train_X, train_y)
    print("A feature that seems to me to be beneficial for the model is sqft_living.\n"
          "A feature that seems to me to not be beneficial for the model is yr_built.\n"
          "I conclude if they are beneficial "
          "or not based on their Pearson Correlation with price.\n"
          "For sqft_living, it is around 0.7 which suggests that sqft_living is beneficial for the model.\n"
          "For yr_built it is around 0 which suggests that yr_built is not beneficial for the model.")
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    print("Q4")
    print("As the percentage increases, the loss and confidence interval both decreases.\n"
          "The confidence interval seems to be converging to 0 and the loss seems to be converging to "
          "somewhere near 25B.\nin terms of estimator properties, the estimator seems to be a biased estimator "
          "because the loss doesn't seem to be converging to 0")





    rows_num = 100-10+1
    loss_arr = np.zeros((rows_num,10))
    for p in range(10,101):
        current_row = p-10
        for j in range(10):
            sample =pd.DataFrame.sample(train_X.assign(price=train_y),frac = p/100)
            sample_y = sample['price']
            sample_X = sample.drop('price', axis=1)
            linearRegressor = LinearRegression()
            linearRegressor.fit(sample_X.to_numpy(),sample_y.to_numpy())
            test_X_arr = test_X.to_numpy()
            test_y_arr=test_y.to_numpy()
            current_loss = linearRegressor.loss(test_X_arr,test_y_arr)
            loss_arr[current_row,j] = current_loss

    loss_mean = np.mean(loss_arr,axis = 1)
    loss_std = np.std(loss_arr,axis = 1)
    p_arr = np.array(range(10,101))
    fig = go.Figure([go.Scatter(x=p_arr, y=loss_mean, mode="markers+lines",
                                line=dict(dash="dash"), marker=dict(color="blue"),showlegend=False),
                          go.Scatter(x=p_arr, y=loss_mean-2*loss_std, fill=None, mode="lines",
                                     line=dict(color="lightgrey"), showlegend=False),
                          go.Scatter(x=p_arr, y=loss_mean+2*loss_std, fill='tonexty',
                                     mode="lines", line=dict(color="lightgrey"), showlegend=False)],
                    layout = go.Layout(title="MSE of test data set as a function of percentage of training data set",
                                       xaxis=dict(title="Percentage of training data set"),
                                       yaxis=dict(title="MSE with test data set")))

    fig.show()