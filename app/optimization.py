import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

# repair function to ensure that the weights add up to one, and that the allocation of a single stock is not below 1e-3
class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)

class PortfolioOptimizationWithSentiment(Problem):
    def __init__(self, returns, cov_matrix, sentiment_vector):
        """
        Portfolio optimization problem considering returns, risk, and sentiment

        Attributes
        ==========
        * returns (np.ndarray): array of expected returns for each stock
        * cov_matrix (np.ndarray): covariance matrix of stock returns
        * sentiment_vector (np.ndarray): sentiment scores for each stock
        """
        super().__init__(n_var=len(returns), n_obj=3, n_constr=1, xl=0, xu=1)
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.sentiment_vector = sentiment_vector

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the portfolio optimization problem

        Parameters
        ==========
        * X (np.ndarray): portfolio allocation matrix
        * out (dict): dictionary to store the evaluation results
        """
        returns = X @ self.returns
        risk = np.sqrt(np.einsum('ij,jk,ik->i', X, self.cov_matrix, X))
        sentiment = X @ self.sentiment_vector
        out["F"] = np.column_stack([-returns, risk, -sentiment])
        out["G"] = np.sum(X, axis=1) - 1

def select_optimal_portfolio(res, risk_appetite):
    """
    Select the optimal portfolio based on the user's risk appetite

    Parameters
    ==========
    * res (pymoo.optimize.Result): result object from the optimization algorithm
    * risk_appetite (str): user's risk appetite

    Returns
    =======
    * np.ndarray: optimal portfolio allocation
    """
    returns = -res.F[:, 0]  # Maximized returns (negative because we minimized in the optimization)
    risk = res.F[:, 1]  # Minimized risk
    sentiment = -res.F[:, 2]  # Maximized sentiment (negative because we minimized in the optimization)

    # Define weights based on risk appetite, with sentiment always considered
    if risk_appetite == 'aggressive':
        weights = {'returns_weight': 0.5, 'risk_weight': 0.2, 'sentiment_weight': 0.3}
    elif risk_appetite == 'conservative':
        weights = {'returns_weight': 0.2, 'risk_weight': 0.5, 'sentiment_weight': 0.3}
    elif risk_appetite == 'moderate':
        weights = {'returns_weight': 0.3, 'risk_weight': 0.4, 'sentiment_weight': 0.3}
    elif risk_appetite == 'risk-averse':
        weights = {'returns_weight': 0.2, 'risk_weight': 0.5, 'sentiment_weight': 0.3}
    elif risk_appetite == 'risk-seeking':
        weights = {'returns_weight': 0.5, 'risk_weight': 0.2, 'sentiment_weight': 0.3}
    elif risk_appetite == 'uncertain':
        weights = {'returns_weight': 0.3, 'risk_weight': 0.4, 'sentiment_weight': 0.3}
    else:
        raise ValueError("Invalid risk appetite")

    composite_scores = compute_composite_score(returns, risk, sentiment, weights)
    optimal_index = np.argmax(composite_scores)
    return res.X[optimal_index]

def compute_composite_score(returns, risk, sentiment, weights):
    """
    Compute a composite score based on returns, risk, and sentiment

    Parameters
    ==========
    * returns (np.ndarray): array of returns
    * risk (np.ndarray): array of risk values
    * sentiment (np.ndarray): array of sentiment scores
    * weights (dict): dictionary of weights for returns, risk, and sentiment

    Returns
    =======
    * np.ndarray: composite score for each portfolio allocation
    """
    normalized_returns = normalize(returns)
    normalized_risk = normalize(risk)
    normalized_sentiment = normalize(sentiment)
    composite_score = (
        weights['returns_weight'] * normalized_returns -
        weights['risk_weight'] * normalized_risk +
        weights['sentiment_weight'] * normalized_sentiment
    )
    return composite_score

def normalize(values):
    """
    Normalize an array of values to the range [0, 1]

    Parameters
    ==========
    * values (np.ndarray): array of values to normalize

    Returns
    =======
    * np.ndarray: normalized array of values
    """
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)
