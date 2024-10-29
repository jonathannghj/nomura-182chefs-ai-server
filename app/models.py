from pydantic import BaseModel
from typing import List, Dict

class Stock(BaseModel):
    """
    Represents an individual stock holding in the user's portfolio

    Attributes
    ==========
    * ticker (str): stock ticker
    * shares (int): number of shares each user hold for this particular stock
    * average_purchase_price (float): average price for this stock that the individual holds
    """
    ticker: str
    shares: int
    average_purchase_price: float

class UserPortfolio(BaseModel):
    """
    Represents the comprehensive portfolio of a user

    Attributes
    ==========
    * username (str): username of the portfolio owner
    * current_portfolio (List[Stock]): list of stocks that this particular user holds    
    """
    username: str
    current_portfolio: List[Stock]

class TradeRecommendation(BaseModel):
    """
    Represents a recommendation of a trade action that a individual should take on a particular stock with reasoning

    Attributes
    ==========
    * action (str): recommended action the user should perform
    * ticker (str): stock symbol or ticker
    * reasoning (str): reasoning behind the action recommended
    * shares (float): number of shares involved in the recommendation
    """
    ticker: str
    action: str
    shares: float
    reasoning: str

class RecommendationRequest(BaseModel):
    """
    Represents the input data consisting of the user's portfolio and their profile
    
    Attributes
    ==========
    * user_data (UserPortfolio): user's portfolio details.
    * risk_appetite (str): user's risk appetite (e.g., "aggressive", "conservative").
    * expected_return (float): user's expected return in percentage over a year.
    """
    user_data: UserPortfolio
    risk_appetite: str
    expected_return: float

class TradeDetails(BaseModel):
    """
    Represents the details of a trade recommendation.

    Attributes
    ==========
    * action (str): The action to take (e.g., 'buy', 'sell').
    * shares (float): The number of shares to trade.
    * reasoning (str): The reasoning behind the trade recommendation.
    """
    action: str
    shares: float
    reasoning: str

class RecommendationResponse(BaseModel):
    """
    Represents the output data, the trade recommendations for the user.

    Attributes
    ==========
    * username (str): username of the portfolio owner.
    * trades (Dict[str, TradeDetails]): A dictionary of trade recommendations where the ticker is the key.
    """
    username: str
    recommended_trades: Dict[str, TradeDetails]