'''
Usage:
python train_buy_strategy.py --config config.yaml

"--config" is optional. it loads config.yaml by default.

'''


def gen_price_trajectory(
    init_price: float,
    simulate_time_interval: int,
    pred_k: float,
    pred_b: float,
    pred_std: float,
    T: int,
    batch_size: int | None = None,
):
    """
    Generate price trajectory for M stocks fromt time t=0 to t=T-1.
    init_price: tensor of shape (1,)
    pred_k, pred_b, pred_std: float numbers
    T: simulation_T in the config.yaml
    
    Return:
    price_trajectory: tensor of shape (batch_size, T)
      price_trajectory[..., 0] = init_price ;
      price_trajectory[..., t] = pred_k * simulate_time_interval * t + pred_b + N(0, pred_std) for t>=1
      where the noise term is sampled from a Gaussian with mean 0 and std pred_std.
    """
    pass # implement this function

def get_final_asset_value(price_trajectory, buy_strategy_a, buy_strategy_b, buy_strategy_c, pred_k, pred_b, pred_std):
    """
    Return the final asset value when applying buying strategy {a,b} on price_trajectory.
    
    price_trajectory: tensor of shape (B,T) where B is the batch size, T is the time length
    buy_strategy_a, buy_strategy_b: float numbers of buying strategy
    
    for t=0,1, ... , T-1, set predicted_price=k*t+b . Set r = price_trajectory[:,t] / predicted_price
    
    Assume that at t=0, we have 1 dollar as our total budget.
    At time t, we spend p_t of our current remaining cash to buy the stock.
    That is, at time t, we have remaing cash C_t = (1-p_0) * (1-p_1) * ... * (1 - p{t-1}).
    We spend C_t * p_t to buy stock, therefore, we will buy num_shares_t = C_t * p_t / price_trajectory[:,t]
    
    We use the following rules to determine p_t:
    p_t is a piecewise linear function in r. x-axis is the value of r, y-axis is the value of p_t:
    - When 0 <= r <= buy_strategy_a , p_t is a linear segment from (0,1) to (buy_strategy_a, buy_strategy_b)
    - When buy_strategy_a <= r <= (buy_strategy_a + buy_strategy_c), p_t is a linear segment from (buy_strategy_a, buy_strategy_b) to (buy_strategy_a + buy_strategy_c, 0)
    - For other r values, p_t takes 0.
    
    Return:
    the total asset value we have at time t=T-1. The total asset value is the cash we have at hand and the stock_shares * stock_price at time T-1.
    Return this dict:
    {
        "stock_share": stock_share, # tensor of shape (B,) shares ofstock we have at time T-1
        "stock_price": stock_price, # tensor of shape (B,) price of stock we have at time T-1
        "stock_value": stock_value, # tensor of shape (B,) total value of stock we have have at time T-1
        "cash": cash, # tensor of shape (B,) the cash we have at hand at time T-1
        "total_asset_value": total_asset_value , # cash + stock_value  at time T-1
    }
    """
    pass # implement this function


def find_optimal_buy_strategy(cfg):
    """
    
    Find optimal buy strategy for stock.
    
    cfg: configuration dict, by default loaded form ./config.yaml
    
    For each stock X in config.yaml "stock_symbols":
    - Its current price can be read from <output_dir>/data/X.csv file. The last line (date) "avg_price" is its current price. Use this as init_price.
    - Its forecasting parameter pred_k/b/std is stored in <output_dir>/forecast/X.json
    
    Do the following:
    - Use gen_price_trajectory to get price_trajectory of shape (batch_size, T)    
    - split buy_strategy_a/b/c into grid_size intervals. a/b/c from/to value can be read from cfg. Say grid_size=10, there are 10*10*10 combinations of a/b/c
    - For each {a,b,c} triplet:
        + apply {a,b,c} on price_trajectory, get get_final_asset_value as Y
        + compute the mean value of Y and std of Y
        + compute buy_strategy_score=Y - buy_strategy_alpha * Y        
    - write these variables to  <output_dir>/train_buy_strategy/X.csv, each variable is a column. Sort by buy_strategy_score from large to small:
        + {a,b,c}
        + mean and std of Y, 
        + buy_strategy_score
        + mean value of number of stock shares, stock values, cashes we have at time T-1.
        + At t=0, how much money we need to spend to buy the stock
        + At t=0, how much shares of stock we should buy
        + At t=0, the price of the stock
    
    Finally, print the first row of  <output_dir>/train_buy_strategy/X.csv in a human-friendly way.
    """
    
