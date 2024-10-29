from fastapi import FastAPI
from models import RecommendationRequest, RecommendationResponse
from sentiment import get_sentiment_score
from finance import fetch_historical_data, calculate_metrics, fetch_current_prices
from optimization import PortfolioOptimizationWithSentiment, PortfolioRepair, select_optimal_portfolio
from utils import calculate_total_value, calculate_optimal_shares, determine_actions
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

@app.get("/")
def read_root():
    return {"message": "Welcome to the 182Chefs application!"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """
    Endpoint to recommend portfolio actions based on user's portfolio and profile

    Parameters
    ==========
    * request (RecommendationRequest): request body containing user data, risk appetite, and expected return

    Returns
    =======
    * RecommendationResponse: response containing the recommended trades for the user
    """

    user_data = request.user_data
    risk_appetite = request.risk_appetite

    tickers = [stock.ticker for stock in user_data.current_portfolio]

    # Get historical data and calculate expected returns and covariance matrix
    historical_data = fetch_historical_data(tickers, period='1y')
    expected_returns, cov_matrix = calculate_metrics(historical_data)

    # Get sentiment for each stock
    sentiment_vector = [get_sentiment_score(ticker) for ticker in tickers]
    
    # Get current prices for each stock
    current_prices = fetch_current_prices(tickers)

    problem = PortfolioOptimizationWithSentiment(expected_returns, cov_matrix, sentiment_vector)
    algorithm = NSGA2(pop_size=100, repair = PortfolioRepair())
    res = minimize(problem, algorithm, ('n_gen', 200), verbose=True)

    optimal_portfolio = select_optimal_portfolio(res, risk_appetite)

    total_value = calculate_total_value(user_data.current_portfolio, current_prices)
    optimal_shares = calculate_optimal_shares(total_value, optimal_portfolio, current_prices)

    recommended_trades = determine_actions(user_data.current_portfolio, optimal_shares)

    response = RecommendationResponse(username=user_data.username, recommended_trades=recommended_trades)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
