import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    a=descent_path[:, 0]
    b = descent_path[:, 1]
    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """


    values = list()
    weights = list()

    def callback(gradient_descent,weight,val,grad,t,eta,delta):
        values.append(val)
        weights.append(weight)


    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):


    l1_values_for_etas = list()
    l2_values_for_etas = list()
    for l_num in range(1,3):
        for i in range(len(etas)):
            if (l_num == 1):
                f = L1(init)
            else:
                f = L2(init)

            curr_eta=etas[i]
            fixed_lr = FixedLR(curr_eta)
            callback,values,weights  =get_gd_state_recorder_callback()
            best_weights_for_curr_eta = GradientDescent(fixed_lr,out_type="best",callback=callback).fit(f,None,None)
            if l_num==1:
                l1_values_for_etas.append(values)
                fig =plot_descent_path(L1,np.array(weights),f"Descent path for module L1 and eta {curr_eta}")
                fig.write_image(f'L1{curr_eta}.png')
            else:
                l2_values_for_etas.append(values)
                fig= plot_descent_path(L2, np.array(weights), f"Descent path for module L2 and eta {curr_eta}")
                fig.write_image(f'L2{curr_eta}.png')



    l1_iterations = np.arange(len(max(l1_values_for_etas, key=len)))
    l1_fig = go.Figure(layout=go.Layout(   title="L1 Norm As a function of The GD iteration",
                                           xaxis_title="Iteration", yaxis_title="L1 Norm"))
    l2_iterations = np.arange(len(max(l2_values_for_etas, key=len)))
    l2_fig = go.Figure(layout=go.Layout(  title="L2 Norm As a function of The GD iteration", xaxis_title="Iteration",
        yaxis_title="L2 Norm"))

    for i in range(len(etas)):
        l1_fig.add_trace(go.Scatter(x=l1_iterations, y=l1_values_for_etas[i],mode='markers+lines',
                                    name=f"eta={etas[i]}"))

        l2_fig.add_trace(go.Scatter(x=l2_iterations,y=l2_values_for_etas[i],mode='markers+lines',
                                    name=f"eta={etas[i]}"))

    l1_fig.show()
    l2_fig.show()

    l1_min_value =np.min(l1_values_for_etas)
    print(f"L1 best loss is: {l1_min_value}")

    l2_min_value =np.min(l2_values_for_etas)

    print(f"L2 best loss is: {l2_min_value}")

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    X_train_arr, y_train_arr, X_test_arr, y_test_arr= X_train.to_numpy(), \
                y_train.to_numpy(),X_test.to_numpy(), y_test.to_numpy()
    logistic_regression = LogisticRegression(solver =GradientDescent())
    logistic_regression.fit(X_train_arr,y_train_arr)
    y_prob = logistic_regression.predict_proba(X_train_arr)
    fpr, tpr, thresholds = roc_curve(y_train_arr, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash')),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         showlegend=False, marker_size=3,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.update_layout(
        width=600,
        height=400
    )
    fig.show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter



if __name__ == '__main__':
    np.random.seed(0)
    #compare_fixed_learning_rates()
    #compare_exponential_decay_rates()
    fit_logistic_regression()
