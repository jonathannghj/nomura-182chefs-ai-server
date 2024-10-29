def calculate_optimal_shares(total_value, optimal_weights, current_prices):
    """
    Calculate the optimal number of shares based on the portfolio allocation, current prices, and total portfolio value

    Parameters
    ==========
    * total_value (float): total value of the portfolio
    * optimal_weights (np.ndarray): array of optimal weights for each stock
    * current_prices (dict): dict with stock symbols as keys and current prices as values

    Returns
    =======
    * dict: dict with stock symbols as keys and optimal number of shares as values
    """
    optimal_shares = {}
    for symbol, weight in zip(current_prices.keys(), optimal_weights):
        optimal_value = total_value * weight
        optimal_shares[symbol] = optimal_value / current_prices[symbol]
    return optimal_shares

def calculate_total_value(current_portfolio, current_prices):
    """
    Calculate the total value of the current portfolio

    Parameters
    ==========
    * current_portfolio (list): list of Stock objects
    * current_prices (dict): dict of current stock prices 

    Returns
    =======
    * float: total value of the portfolio
    """
    total_value = sum(stock.shares * current_prices[stock.ticker] for stock in current_portfolio)
    return total_value

def determine_actions(current_portfolio, optimal_shares):
    """
    Determine actions to take based on the current portfolio and optimal shares

    Parameters
    ==========
    * current_portfolio (list): list of dictionaries with stock holdings
    * optimal_shares (dict): dictionary with stock symbols as keys and optimal number of shares as values

    Returns
    =======
    * list: list of dictionaries with recommended actions for each stock
    """
    actions = {}
    for stock in current_portfolio:
        current_shares = stock.shares
        optimal_shares_count = optimal_shares[stock.ticker]
        action_shares = optimal_shares_count - current_shares

        if action_shares > 0:
            action = 'Buy'
        elif action_shares < 0:
            action = 'Sell'
        else:
            action = 'Hold'

        actions[stock.ticker] = {
            'action': action,
            'shares': abs(action_shares),
            'reasoning': f'{action} shares to match optimal allocation of {optimal_shares_count:.2f} shares'
        }
    print(actions)
    

    return actions
