from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
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
    X= np.arange(10,1010,10)
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
    real_mu =np.array([0,0,4,0])
    real_cov =np.array(
    [[1, 0.2, 0, 0.5],
     [0.2, 2, 0, 0],
     [0, 0, 1, 0],
     [0.5, 0, 0, 1]])

    X=np.random.multivariate_normal(real_mu,real_cov,1000)
    multivariate_gaussian_estimation = MultivariateGaussian().fit(X)
    print(multivariate_gaussian_estimation.mu_)
    print(multivariate_gaussian_estimation.cov_)


    # Question 5 - Likelihood evaluation
    f= np.linspace(-10, 10, 200)
    log_likelihood_results = np.zeros([200,200])
    for i in range (200):
        for j in range(200):
            f1 = f[i]
            f3= f[j]
            log_likelihood_results[i,j] = \
                MultivariateGaussian.log_likelihood(np.array([f1,0,f3,0]),real_cov,X)
    plt.imshow(log_likelihood_results, cmap='viridis',extent=[-10,10,-10,10])
    plt.colorbar()
    plt.xlabel("f3")
    plt.ylabel("f1")
    plt.title("a good title")
    plt.show()



    # Question 6 - Maximum likelihood

    indexes_max_likelihood =np.unravel_index(log_likelihood_results.argmax(),
                                             log_likelihood_results.shape)
    # best_pair=np.around(f[indexes_max_likelihood],3)
    print(np.around(f[indexes_max_likelihood[0]],3))
    print(np.around(f[indexes_max_likelihood[1]], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
