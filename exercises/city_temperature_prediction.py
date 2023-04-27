import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,parse_dates=['Date'])
    df = df.dropna()
    df = df[(df['Temp']>(-30))&(df['Temp']<(50))]
    df = df.drop(df[(df['Country']=='Israel')&((df['Temp']<0)|(df['Temp']>50))].index)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Year'] =df['Year'].astype(str)

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    print("Polynomial Fitting")
    df = load_data('C:\Program Files (x86)\IMLprojects\IML.HUJI\datasets\City_Temperature.csv')


    # Question 2 - Exploring data for specific country
    print("Q2")
    print("*The scarlett resembles in my opinion a wave graph so a polynomial of degree "
          "3 might be suitable for this data")



    df_israel = df[df['Country']=='Israel']


    fig1 = px.scatter(df_israel, x='DayOfYear', y='Temp',color='Year')
    #fig1.show()
    fig1.write_image(".fig1.png")


    df_israel_month_temp_sd = df_israel.groupby('Month',as_index =False).agg({'Temp':'std'})

    fig2 = px.bar( x=df_israel_month_temp_sd['Month'], y=df_israel_month_temp_sd['Temp'],
                  title='temp std for each month ',labels={'x': 'Month k', 'y':'temp std'})
    #fig2.show()
    fig2.write_image(".fig2.png")

    print("*Based on this graph, I don't expect the model to succeed equally over all months.\nThe reason being that"
          "in months with a relatively low std,such as months 7 and 8, the model will probably give a more accurate"
          " prediction than \nit's prediction on months with  relatively high std,such as months 3 and 4")

    # Question 3 - Exploring differences between countries
    print("Q3")
    df_month_country_temp = df.groupby(['Country','Month'],as_index =False).agg({'Temp':['mean','std']})


    fig3 = px.line( x=df_month_country_temp[('Month','')], y=df_month_country_temp[('Temp', 'mean')],
                    color=df_month_country_temp[('Country','')],
                    error_y=df_month_country_temp[('Temp', 'std')])
    fig3.write_image(".fig3.png")

    print("Based on this graph, different countries have different patterns.\n"
          "The model fitted for Israel is likely to work well on Jordan because it's pattern is "
          "similar enough to Israel's.\nThe model is not likely to work well on South Africa and The Netherlands "
          "because their patterns are too different from the Israel's pattern. ")

    #fig3.show()

    # Question 4 - Fitting model for different values of `k`
    print("Q4")
    y= df_israel['Temp']
    training_data_set,training_y,test_data_set,test_y=split_train_test(df_israel.drop('Temp', axis=1),y)
    training_X = training_data_set['DayOfYear']
    test_X = test_data_set['DayOfYear']
    loss_arr= np.zeros(10)
    for k in range(1,11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(training_X,training_y.to_numpy())
        loss_arr[k-1]= np.round(poly_model.loss(test_X,test_y.to_numpy()),2)

    print(loss_arr)
    fig4 = px.bar( x=np.arange(1,11), y=loss_arr, title='loss for each k',labels={'x': 'degree k', 'y':'loss'},
                   text_auto=True)
    fig4.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig4.write_image(".fig4.png")
    print("k=5 got us the lowest loss, because of this k with a value of 5 best fits the data")

    # Question 5 - Evaluating fitted model on different countries
    print("Q5")

    degree=5
    poly_model = PolynomialFitting(degree)
    poly_model.fit(df_israel['DayOfYear'],df_israel['Temp'])

    other_countries = ['South Africa','Jordan', 'The Netherlands']
    other_countries_loss_arr = np.zeros(3)
    for i, country in enumerate(other_countries):
        df_country= df[df['Country'] == country]

        other_countries_loss_arr[i] =  poly_model.loss(df_country['DayOfYear'],df_country['Temp'])
    fig5 = px.bar( x=other_countries, y=other_countries_loss_arr,
                   title='loss for other countries',labels={'x': 'Countries', 'y':'Loss'},text_auto=True)
    fig5.write_image(".fig5.png")

    print("Out of the three countries the model performed best on Jordan, probably because Jordan's Pattern on question 3"
          "is relatively similar to Israel's pattern.\nThe model performed poorly on South Africa and The Netherlands,"
          "probably because their patterns in question 3 are too different from Israel's pattern ")