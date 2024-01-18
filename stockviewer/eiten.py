import math, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load our modules
from stockviewer.data_loader import DataEngine
from stockviewer.simulator import MontoCarloSimulator
from stockviewer.backtester import BackTester
from stockviewer.strategy_manager import StrategyManager

import plotly.io as pio
import plotly.graph_objects as go

from stockviewer import ROOT_DIR


def save_obj(obj, name):
    '''Saves object in the results folder'''
    with open(ROOT_DIR + "/" + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name, pandas_pkl = True):
    '''Loads object from the results folder'''
    if pandas_pkl :
        return pd.read_pickle(ROOT_DIR + "/" + name +'.pkl')
    else:
        with open(ROOT_DIR + "/" + name + '.pkl', 'rb') as f:
            return pickle.load(f)
        
pio_template = load_obj("./assets/pio_template_stockviewer")
pio.templates["stockviewer"] = pio_template
pio.templates.default = "stockviewer"

class Args(argparse.Namespace):
    is_test = 0
    future_bars = 90
    data_granularity_minutes=3600
    history_to_use = "all"
    apply_noise_filtering = 1
    market_index = "QQQ"
    only_long = 1
    eigen_portfolio_number = 3
    stocks_file_path = "../stocks/stocks.txt"

class Eiten:
    def __init__(self, args):
        #plt.style.use('seaborn-white')
        plt.rc('grid', linestyle="dotted", color='#a0a0a0')
        plt.rcParams['axes.edgecolor'] = "#04383F"
        plt.rcParams['figure.figsize'] = (12, 6)

        print("\n--* Eiten has been initialized...")
        self.args = args

        # Create data engine
        self.dataEngine = DataEngine(args)

        # Monto carlo simulator
        self.simulator = MontoCarloSimulator()

        # Strategy manager
        self.strategyManager = StrategyManager()

        # Back tester
        self.backTester = BackTester()

        # Data dictionary
        self.data_dictionary = {}

        print('\n')

    def calculate_percentage_change(self, old, new):
        """
        Calculate percentage change
        """
        return ((new - old) * 100) / old

    def create_returns(self, historical_price_info):
        """
        Create log return matrix, percentage return matrix, and mean return 
        vector
        """

        returns_matrix = []
        returns_matrix_percentages = []
        predicted_return_vectors = []
        for i in range(0, len(historical_price_info)):
            close_prices = list(historical_price_info[i]["Close"])
            log_returns = [math.log(close_prices[i] / close_prices[i - 1])
                           for i in range(1, len(close_prices))]
            percentage_returns = [self.calculate_percentage_change(
                close_prices[i - 1], close_prices[i]) for i in range(1, len(close_prices))]

            total_data = len(close_prices)

            # Expected returns in future. We can either use historical returns as future returns on try to simulate future returns and take the mean. For simulation, you can modify the functions in simulator to use here.
            future_expected_returns = np.mean([(self.calculate_percentage_change(close_prices[i - 1], close_prices[i])) / (
                total_data - i) for i in range(1, len(close_prices))])  # More focus on recent returns

            # Add to matrices
            returns_matrix.append(log_returns)
            returns_matrix_percentages.append(percentage_returns)

            # Add returns to vector
            # Assuming that future returns are similar to past returns
            predicted_return_vectors.append(future_expected_returns)

        # Convert to numpy arrays for one liner calculations
        predicted_return_vectors = np.array(predicted_return_vectors)
        returns_matrix = np.array(returns_matrix)
        returns_matrix_percentages = np.array(returns_matrix_percentages)

        return predicted_return_vectors, returns_matrix, returns_matrix_percentages

    def unpack_data(self):
        # Add data to lists
        symbol_names = list(sorted(self.data_dictionary.keys()))
        historical_price_info, future_prices = [], []
        for symbol in symbol_names:
            historical_price_info.append(self.data_dictionary[symbol]["historical_prices"])
            future_prices.append(self.data_dictionary[symbol]["future_prices"])

        # Get return matrices and vectors
        predicted_return_vectors, returns_matrix, returns_matrix_percentages = self.create_returns(
            historical_price_info)

        return historical_price_info, future_prices, symbol_names, predicted_return_vectors, returns_matrix, returns_matrix_percentages

    def load_data(self):
        """
        Loads data needed for analysis
        """
        # Gather data for all stocks in a dictionary format
        # Dictionary keys will be -> historical_prices, future_prices
        self.data_dictionary = self.dataEngine.collect_data_for_all_tickers()
        return 

        

    def get_strategies_figs(self):
        """
        Run strategies, back and future test them, and simulate the returns.
        """

        figs = []
        historical_price_info, future_prices, symbol_names, predicted_return_vectors, returns_matrix, returns_matrix_percentages = self.unpack_data()
        historical_price_market, future_prices_market = self.dataEngine.get_market_index_price()

        # Calculate covariance matrix
        covariance_matrix = np.cov(returns_matrix)

        # Use random matrix theory to filter out the noisy eigen values
        if self.args.apply_noise_filtering:
            print(
                "\n** Applying random matrix theory to filter out noise in the covariance matrix...\n")
            covariance_matrix = self.strategyManager.random_matrix_theory_based_cov(
                returns_matrix)

        # Get weights for the portfolio
        eigen_weights = self.strategyManager.calculate_eigen_portfolio(symbol_names, covariance_matrix, self.args.eigen_portfolio_number)
        mvp_weights = self.strategyManager.calculate_minimum_variance_portfolio(symbol_names, covariance_matrix)
        msr_weights = self.strategyManager.calculate_maximum_sharpe_portfolio(symbol_names, covariance_matrix, predicted_return_vectors)
        ga_weights = self.strategyManager.calculate_genetic_algo_portfolio(symbol_names, returns_matrix_percentages)

        fig = go.Figure(data=[
            go.Bar(name=strategy, x=symbol_names, y=[strategy_weights[stock] for stock in symbol_names])
            for (strategy, strategy_weights) in zip(["Eigen", "MVP", "MSR", "Genetic"], [eigen_weights, mvp_weights, msr_weights, ga_weights])
        ])
        # Change the bar mode
        fig.update_layout(barmode='group', xaxis_title = "Stock symbols", yaxis_title = "Weight")
        figs.append( (fig, "Strategy weights"))

        fig = go.Figure()
        for (strategy, strategy_weights) in zip(["Eigen", "MVP", "MSR", "Genetic"], [eigen_weights, mvp_weights, msr_weights, ga_weights]):
            # Get market returns during the backtesting time
            historical_price_market_close = list(historical_price_market["Close"])
            market_returns = [self.backTester.calculate_percentage_change(historical_price_market_close[i - 1], historical_price_market_close[i]) for i in range(1, len(historical_price_market_close))]
            market_returns_cumulative = np.cumsum(market_returns)

            # Get invidiual returns for each stock in our portfolio
            normal_returns_matrix = []
            for symbol in symbol_names:
                symbol_historical_prices = list(self.data_dictionary[symbol]["historical_prices"]["Close"])
                symbol_historical_returns = [self.backTester.calculate_percentage_change(symbol_historical_prices[i - 1], symbol_historical_prices[i]) for i in range(1, len(symbol_historical_prices))]
                normal_returns_matrix.append(symbol_historical_returns)

            # Get portfolio returns
            normal_returns_matrix = np.array(normal_returns_matrix).transpose()
            portfolio_weights_vector = np.array([self.backTester.portfolio_weight_manager(strategy_weights[symbol], self.args.only_long) for symbol in strategy_weights]).transpose()
            portfolio_returns = np.dot(normal_returns_matrix, portfolio_weights_vector)
            portfolio_returns_cumulative = np.cumsum(portfolio_returns)
            
            fig.add_trace(go.Scatter(x = historical_price_market["Date"], y = portfolio_returns_cumulative, mode = "lines", name = strategy))

        fig.add_trace(go.Scatter(x = historical_price_market["Date"], y = market_returns_cumulative, mode = "lines", line_color = 'white', name = 'Market Index'))
        fig.add_hline(y = 0, line_dash="dot", line_color = "white")
        fig.update_layout(xaxis_title = "Date", yaxis_title = "Return (%)")
        figs.append((fig, "Strategy backtest (in sample !)"))

        return figs
    
    def get_stocks_figs(self):
        figs = []

        historical_price_info, future_prices, symbol_names, predicted_return_vectors, returns_matrix, returns_matrix_percentages = self.unpack_data()

        fig = go.Figure()
        for i, symbol in enumerate(symbol_names):
            symbol_prices = historical_price_info[i]
            fig.add_trace(go.Scatter(x = symbol_prices["Date"], y = symbol_prices["Close"], mode = "lines", name = symbol))
        fig.update_layout(xaxis_title = "Date", yaxis_title = "Stock value")
        figs.append((fig, "Stock evolution"))

        return figs

    def draw_plot(self, filename="output/graph.png"):
        """
        Draw plots
        """
        # Styling for plots

        plt.grid()
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.show()
        
        """if self.args.save_plot:
            plt.savefig(filename)
        else:
            plt.tight_layout()
            plt.show()""" # Plots were not being generated properly. Need to fix this.

    def print_and_plot_portfolio_weights(self, weights_dictionary: dict, strategy, plot_num: int) -> None:
        print("\n-------- Weights for %s --------" % strategy)
        symbols = list(sorted(weights_dictionary.keys()))
        symbol_weights = []
        for symbol in symbols:
            print("Symbol: %s, Weight: %.4f" %
                  (symbol, weights_dictionary[symbol]))
            symbol_weights.append(weights_dictionary[symbol])

        # Plot
        width = 0.1
        x = np.arange(len(symbol_weights))
        plt.bar(x + (width * (plot_num - 1)) + 0.05,
                symbol_weights, label=strategy, width=width)
        plt.xticks(x, symbols, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Symbols", fontsize=14)
        plt.ylabel("Weight in Portfolio", fontsize=14)
        plt.title("Portfolio Weights for Different Strategies", fontsize=14)
