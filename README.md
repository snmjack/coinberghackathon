# Coinberg Hackathon - Time series problem statement

# Problem statament: Forecasting Crypto currency values using Time Series for next 30 days

# Data used :- Bitcoin, Cardano, Ethereum, Ripple, Tron

# Approach:-
-Got the data from Coinberg API
-Checked the data, found some null values in files, so treated and cleaned the null values

# Data Engineering:-
-Derived a column named returns which calculates the returns in percentage based on previous days and current days closing price
-Included USD and EUR currency data for each date in the dataset since internation currence also affects the price of the cryto currencies
-Downloaded all the tweets of official crypto currence handles with favourites and retweets

# Models tried:-
-Linear Regression
-Lasso Regression
-Polynomial Regression
-RandomForest regressor
-ARIMA
-SARIMA

# Final Model used:-
- RandomForest Regressor
