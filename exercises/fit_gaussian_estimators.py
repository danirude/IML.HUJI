from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    real_mu =10
    real_var =1
    V = np.random.normal(real_mu,real_var,1000)
    gaussian_estimation = UnivariateGaussian().fit(V)
    print(gaussian_estimation.mu_,gaussian_estimation.var_)

    # Question 2 - Empirically showing sample mean is consistent
    X= np.arange(10,1010,10,dtype=np.int)
    Y = np.zeros((X.size,))
    for i in range(X.size):
        Y[i] = np.abs(real_mu-UnivariateGaussian().fit(V[:X[i]]).mu_)

    fig = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=X, y=Y, mode='markers', marker=dict(color="blue"))])
    fig.update_layout(title_text=" informative title text").update_yaxes\
        (title_text="diff", secondary_y=False, row=1, col=1).update_xaxes\
        (showgrid=False, title_text="distance from real expectation", row=1, col=1)
    fig.show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_X = gaussian_estimation.pdf(V)
    fig2 = make_subplots(rows=1, cols=1).add_traces([go.Scatter(x=V, y=pdf_X, mode='markers', marker=dict(color="blue"))])
    fig2.update_layout(title_text=" informative title text").update_yaxes\
        (title_text="probability", secondary_y=False, row=1, col=1).update_xaxes\
        (showgrid=False, title_text="samples", row=1, col=1)
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
