# DCA-Investment-Advisor

A simple machine learning based tool to help me on Dollar-Cost Averaging (DCA) investment.


## Warning

This is a personal project just for fun. Use this tool at your own risk. **You might loose all your money if you trust this tool blindly.**

## Installation

```
# [optional] create conda virtual environment
conda create --prefix=./venv python=3.11 -y
conda activate ./venv
pip install -r requirements.txt
```

## ## Usage

Set your configuration in `config.yaml`, particularly, `stock_symbols` is the stocks your want to buy.

In bash shell:

```
bash run.sh
```

This will download real-time stock price data to `outputs/data` and generate a price prediction model in the `outputs/forecast`. Then it will train the buying strategy model in `outputs/train_buy_strategy`.

The final buying suggestion will be printed to the screen, for example:

```
QQQ: best a=0.8889, b=1.0000, c=0.2222 -> total=1.1816, std=0.1304, score=1.1164, initial spend=0.1993, initial shares=0.0003, price=608.1367
```

This means:

- You should be QQQ today with 19.93% of your total cash budget. That is, if you have $1 to invest today, you should spend  $0.1993.
- The price of QQQ today is  $608.1367.
- The model assumes that you will repeat the buying action monthly
- The total asset value after 12 months is expected to be $1.1816 if your total cash budget is $1 today.
- The alpha-sigma lower bound of your 12-month predicted asset value of $1.1164. The alpha value is specified in `buy_strategy_alpha` variable.


## How it works

help me write this session. tell the user how it download and impute data, train price prediction model and the buy strategy model. explain the machine learning models i used in price prediction and buying strategy.
