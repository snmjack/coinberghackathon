# Coinberg Hackathon - 
CryptoCurrency Time series problem statement

# Problem statament: 
Forecasting Crypto currency values using Time Series for next 30 days

# Data Given :- 
Bitcoin, Cardano, Ethereum, Ripple, Tron (source Coinberg API)

# Setup 
data folder : contains all price files and other data (tweets for each currency, gold , eur prices)

install : ta-lib, textblob

# Approvh 
Use of regression models for timeseries forecasting. 

Based on historical information, prices for each day of the next months are predicted .

# Features used :

Price Data: 'Close', 'High', 'Low', 'Market Cap', 'Open','Volume',

Previous days returns : 'PrevRet_1day', 'PrevRet_3day', 'PrevRet_5day','PrevRet_15day',

Moving Averages :   'SMA_30', 'SMA_40', 'SMA_50', 'EMA_30', 'EMA_40','EMA_50',

Previous days EUR and gold returns: 'PrevRet_1_day_EUR', 'PrevRet_3_day_EUR', 'PrevRet_5_day_EUR', 'PrevRet_1_day_Gold', 'PrevRet_3_day_Gold', 'PrevRet_5_day_Gold',

TweetSentiments :         'tweet_sentiment_score'  (calculated using tweet polarity * subjectivity * (# of likes + # of retweets)


# Models Used:-
-Linear Regression

-Lasso Regression

-Polynomial Regression

-RandomForest regressor

-ARIMA

-SARIMA

# Final Model used:-
- RandomForest Regressor

# Results
 30day_predictions_May23Prices.txt contains 30 day projections based on data avaialable till may 23, 2018
 
