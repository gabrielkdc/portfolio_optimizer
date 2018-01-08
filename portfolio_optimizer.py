import os
import numpy as np
import matplotlib as mlp
import pandas as pd
import scipy.optimize as spo
import datetime
from decimal import Decimal


def symbol_to_path(symbol, base_dir="stocks"):
	return os.path.join(base_dir, str(symbol))


def get_data(dates, symbols):
	df = pd.DataFrame(index = dates)

	for symbol in symbols:
		df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date",  usecols=['Date', 'Adj Close'], na_values="NaN")
		df_temp = df_temp.rename(columns={'Adj Close':symbol})
		df = df.join(df_temp, how="inner")

	return df

def get_daily_returns_panda(df):
	daily_returns = ((df / df.shift(1)) -1)*100
	daily_returns.ix[0, :] = 0
	return daily_returns

def minimize_function(C, df):	
	daily_returns = df.copy()	
	daily_returns =  daily_returns*C
	port_val = daily_returns.sum(axis = 1)
	
	avg_daily_returns = port_val.mean(axis = 0)
	std_daily_returns = port_val.std(axis = 0)
	sharpe_ratio = (252**(1/2.0)) * (avg_daily_returns/std_daily_returns)
	return sharpe_ratio*-1

def con(x):
	return x.sum()-1

def optimize(data, error_func):
	same_avg = 1.0/(len(data.columns)-1)
	initial_guess = np.full((len(data.columns)),same_avg )
	cons = ({'type':'eq', 'fun': con})
	bounds = ((0,1),)*len(data.columns)
	result = spo.minimize(error_func, initial_guess, args=(data,), method='SLSQP', options={'disp':True}, constraints=cons, bounds=bounds )
	return result

def run():
	dates = pd.date_range('2017-01-01', '2017-12-31')		
	symbols = os.listdir("stocks")	
	df = get_data(dates, symbols)
	initial_investment = 15000
	normalized = df.copy()	
	normalized[1:] = ((normalized[1:]/normalized[:-1].values) -1)
	normalized.ix[0] = 0
	result =optimize(normalized, minimize_function)	
	print(symbols)
	print(np.around(result.x*initial_investment, decimals=4))




run()

	
