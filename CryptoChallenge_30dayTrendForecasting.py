
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import *   
from math import * 

from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from fbprophet import Prophet
import talib

def stat_model (pipeline, param_grid, X_train,X_test,y_train,  cv=3):    
  #get returns as y_pred
  y_train = calc_returns(y_train['NextDayPrice'], X_train["Close"]) 
  close_prices_test= X_test["Close"]  

  clf = GridSearchCV(pipeline ,param_grid=param_grid,  cv=cv)
  clf.fit(X_train,y_train) 
  model = clf.best_estimator_ 
  model.fit(X_train,y_train)
  y_pred= model.predict(X_test)

  #convert returns to price
  y_pred = calc_price (close_prices_test, y_pred)
  return model.__class__.__name__, model, y_pred
  
def analyze_results(y_test, y_pred):
  rmse= sqrt( mean_squared_error(y_test, y_pred ))
  mae= mean_absolute_error (y_test,y_pred)
  r2= r2_score(y_test,y_pred)
  avg_y = y_test.mean()[0]
  min_y = y_test.min()[0]
  max_y = y_test.max()[0]


  results =  pd.DataFrame()
  results["score"]= pd.Series([str(min_y)+'-'+str(max_y),avg_y, rmse,mae,r2])

  results.index = ['Price Range','Average Value','rmse','mae','r2']
  
  #plot y_test vs y_pred
  df =  pd.DataFrame()
  df["y_test"] = y_test["NextDayPrice"]
  df["y_pred"] = 0.0
  ctr = 0
  for v in y_pred:
      df.iloc[ctr,1] = y_pred[ctr] 
      ctr = ctr+1
  fig, ax1 = plt.subplots()
  ax1.plot(df.index, df['y_test'], 'g-', label= "y_test")
  ax1.plot(df.index, df['y_pred'], 'b-', label= "y_pred")
  ax1.legend()
  plt.show()
  return results

def split_data(data,testdays,predict_day=1):
  data['NextDayPrice'] = data["Close"].shift(-1* predict_day)

  features = data.iloc[:-1,:-1]
  labels = data.iloc[:-1,-1:] 
  if predict_day > testdays:
    testdays =predict_day+1

  # X_train, X_test, y_train, y_test = \
  #             train_test_split(features, labels , random_state=42)
  X_train= features[:-1 * testdays]
  y_train= labels[:-1 * testdays]

  X_test= features[-1 * testdays:]
  y_test= labels[-1 * testdays:]
  
  return X_train, X_test,y_train,y_test

def calc_returns(newprice,oldprice):
  return ((newprice-oldprice)/oldprice).as_matrix()

def calc_price(oldprice,est_return):
  return oldprice * (1+ est_return)

def run_nextday_return_pipeline (pipeline, param_grid ,prev_returns_to_be_added,  testdays, x_test =None):
  currency_list = ['Bitcoin','Ethereum','Ripple',
                   'Tron',
                   'Cardano']   
 
  nextday_returns_models = {}          
  for cur in currency_list:
      print (cur)
      filepath = './data/'+cur + '.csv'
      tweeterfilepath= './data/'+cur + '_tweets.csv'
      data = getData(filepath ,tweeterfilepath,prev_returns_to_be_added)

      X_train, X_test,y_train,y_test = split_data(data,testdays)
      if x_test is not None: 
        X_test = x_test
        
      (model_name, model,y_pred) =  stat_model (pipeline, param_grid, X_train, X_test, y_train)
      
      nextday_returns_models[cur] = model 
      print( analyze_results(y_test, y_pred) )
  return nextday_returns_models





def run_next_n_th_day_return_pipeline (pipeline, param_grid ,prev_returns_to_be_added,  testdays, x_test =None , day =1):
  currency_list = ['Bitcoin','Ethereum','Ripple',
                   'Tron',
                   'Cardano']   
 
  nextday_returns_models = {}          
  for cur in currency_list:
      print (cur)
      filepath = './data/'+cur + '.csv'
      tweeterfilepath= './data/'+cur + '_tweets.csv'
      data = getData(filepath ,tweeterfilepath,prev_returns_to_be_added)

      X_train, X_test,y_train,y_test = split_data(data,testdays, predict_day=day)
      if x_test is not None: 
        X_test = x_test
        
      (model_name, model,y_pred) =  stat_model (pipeline, param_grid, X_train, X_test, y_train)
      
      nextday_returns_models[cur] = model 
      print( analyze_results(y_test[:testdays - day], y_pred[:testdays-day]) )
  return nextday_returns_models

def run_nextwholemonth_return_pipeline (pipeline, param_grid ,prev_returns_to_be_added,  testdays, x_test =None):
  currency_list = ['Bitcoin','Ethereum','Ripple',
                   'Tron',
                   'Cardano']   
 
  nextwholemonth_returns_models = {}          
  for cur in currency_list:
      print (cur)
      filepath = './data/'+cur + '.csv'
      tweeterfilepath= './data/'+cur + '_tweets.csv'
      data = getData(filepath ,tweeterfilepath,prev_returns_to_be_added)
      nextwholemonth_returns_models[cur] = {}
      
      models ={}
      for i in range(1,30):
        print('building model for predicting :'+cur + ' returns on day : ' + str(i) + '')
        X_train, X_test,y_train,y_test = split_data(data,testdays, predict_day=i)
        if x_test is not None: 
          X_test = x_test        
        (model_name, model,y_pred) =  stat_model (pipeline, param_grid, X_train, X_test, y_train)
        models[i] =model
       
      nextwholemonth_returns_models[cur]["data"]  =  data
      nextwholemonth_returns_models[cur]["models"] =  models
        
  return nextwholemonth_returns_models

import re
from textblob import TextBlob

def get_tweet_sentiments(tweet_filepath):
  df = pd.read_csv(tweet_filepath, index_col=0)
  df["Date"] = pd.DatetimeIndex(df["created_at"]).date

  df["tweet_sentiment_actions"] =df["Likes"] + df["Retweet"] +1 # to handle 0likes/retweets
  df["tweet_sentiment_polarity"] =0.0
  df["tweet_sentiment_subjectivity"] =0.0

  #add sentiment columns
  for idx,row in df.iterrows():
    #clean tweet text by removing links, special characters
    tweet_txt =' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row["text"]).split())
    # create TextBlob object of passed tweet text
    tweet_sentiment = TextBlob(tweet_txt).sentiment  
    df["tweet_sentiment_polarity"]=  tweet_sentiment.polarity
    df["tweet_sentiment_subjectivity"] = tweet_sentiment.subjectivity
    
  df["tweet_sentiment_score"] = df["tweet_sentiment_polarity"] * df["tweet_sentiment_actions"] *df["tweet_sentiment_subjectivity"] 
  return df[["Date","tweet_sentiment_score"]].groupby("Date").sum()
 


import statsmodels.api as sm

def getData(filepath, tweeterfilepath,prev_returns_to_be_added):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime( df["Date"]) 
    
    df = df.set_index(df["Date"], drop =True)   
    
    df.drop(df.columns[0], axis=1 ,inplace=True)
    df.drop(["Date"], axis=1 ,inplace=True)
    
    print("Get previous days high, low, market cap, open, volume")
    df["PrevHigh"]= df["High"].shift(1)
    df["PrevLow"]= df["Low"].shift(1)
    df["PrevMarket Cap"]= df["Market Cap"].shift(1)
    df["PrevOpen"]= df["Open"].shift(1)
    df["PrevVolume"]= df["Volume"].shift(1)    
    df.drop(["High","Low","Market Cap", "Open","Volume"], axis=1 ,inplace=True)
    df = df.sort_index ( ascending = True) 
    
    print('Add previous returns as a features to predict next returns')
    for day in prev_returns_to_be_added:
      df["OldPrice"] =   df["Close"].shift(day)
      df["PrevRet_"+str(day) + "day"] = calc_returns (df["Close"] , df["OldPrice"] ) 
      #df["PrevRet_"+str(day) +"day"].fillna(method='bfill', inplace=True)    
      df.drop(["OldPrice"], axis=1 ,inplace=True  ) 
    df = df.dropna()
    
    
    
    print ('Add Simple Moving Average ( 30, 40, 50 Days)')
    df['SMA_30']= talib.SMA(df['Close'].values, timeperiod = 30)
    df['SMA_40']= talib.SMA(df['Close'].values, timeperiod = 40)
    df['SMA_50']= talib.SMA(df['Close'].values, timeperiod = 50)


    print ('Add Exponential Moving Average ( 30, 40, 50 Days):more weightage to latest data ')
    df['EMA_30']= talib.EMA(df['Close'].values, timeperiod = 30)
    df['EMA_40']= talib.EMA(df['Close'].values, timeperiod = 40)
    df['EMA_50']= talib.EMA(df['Close'].values, timeperiod = 50)

    print('add EUR and gold returns')
    eur= pd.read_csv ("data/EUR.csv")
    eur["Date"] = pd.to_datetime( eur["Date"]) 
    eur = eur.set_index(eur["Date"], drop =True)   
    eur.drop(["Date"], axis=1 ,inplace=True  ) 
    eur["PrevRet_1_day_EUR"] = calc_returns(eur["Close"], eur["Close"].shift(1))
    eur["PrevRet_3_day_EUR"] = calc_returns(eur["Close"], eur["Close"].shift(3))
    eur["PrevRet_5_day_EUR"] = calc_returns(eur["Close"], eur["Close"].shift(5))


    gold= pd.read_csv ("data/gold.csv", header=0)
    gold["Date"] = pd.to_datetime( gold["Date"]) 
    gold = gold.set_index(gold["Date"], drop =True)   
    gold.drop(["Date"], axis=1 ,inplace=True  )  
    gold["Close"] =pd.to_numeric (gold["Price"])
    gold["PrevRet_1_day_Gold"] = calc_returns(gold["Close"], gold["Close"].shift(1))
    gold["PrevRet_3_day_Gold"] = calc_returns(gold["Close"], gold["Close"].shift(3))
    gold["PrevRet_5_day_Gold"] = calc_returns(gold["Close"], gold["Close"].shift(5))

    df["PrevRet_1_day_EUR"]=eur["PrevRet_1_day_EUR"]
    df["PrevRet_3_day_EUR"]=eur["PrevRet_3_day_EUR"]
    df["PrevRet_5_day_EUR"]=eur["PrevRet_5_day_EUR"]

    df["PrevRet_1_day_Gold"]=gold["PrevRet_1_day_Gold"]
    df["PrevRet_3_day_Gold"]=gold["PrevRet_3_day_Gold"]
    df["PrevRet_5_day_Gold"]=gold["PrevRet_5_day_Gold"]

    print ("Fill all NAs with closest numbers")
    df.fillna(method="bfill",inplace=True )
    df.fillna(method="ffill", inplace =True)
    
    
    print('Add tweeter data')     
    sentiments_df = get_tweet_sentiments(tweeterfilepath)
    df["tweet_sentiment_score"]=sentiments_df["tweet_sentiment_score"]
    df["tweet_sentiment_score"]=df["tweet_sentiment_score"].shift(1)
    df["tweet_sentiment_score"].fillna(0, inplace=True)
    
    return df




pipeline = Pipeline([
        ('StandardScaler',StandardScaler()),   
        ('RandomForestRegressor', RandomForestRegressor(oob_score=True, random_state=9))
        ])
    
rfr_param_grid ={  
                  "RandomForestRegressor__max_features": ['sqrt', 2, "log2"],
                  "RandomForestRegressor__n_estimators": [120],
                  "RandomForestRegressor__max_depth": [40],
                  "RandomForestRegressor__max_leaf_nodes": [ 10]}
 # last 20 days of data is test data 
nextday_return_models = run_nextday_return_pipeline  (pipeline, rfr_param_grid, [1 ,3, 5,15], testdays = 60)
    



pipeline = Pipeline([
        ('StandardScaler',StandardScaler()),   
        ('RandomForestRegressor', RandomForestRegressor(oob_score=True, random_state=9))
        ])
    
rfr_param_grid ={  
                  "RandomForestRegressor__max_features": ['sqrt', 2, "log2"],
                  "RandomForestRegressor__n_estimators": [120],
                  "RandomForestRegressor__max_depth": [40],
                  "RandomForestRegressor__max_leaf_nodes": [ 10]}
 # last 20 days of data is test data 
nextday_return_models = run_next_n_th_day_return_pipeline  (pipeline, rfr_param_grid, [1 ,3, 5,15], testdays = 60 , day =10)
    



pipeline = Pipeline([
        ('StandardScaler',StandardScaler()),   
        ('RandomForestRegressor', RandomForestRegressor(oob_score=True, random_state=9))
        ])
    
rfr_param_grid ={  
                  "RandomForestRegressor__max_features": ['sqrt', 2, "log2"],
                  "RandomForestRegressor__n_estimators": [120],
                  "RandomForestRegressor__max_depth": [40],
                  "RandomForestRegressor__max_leaf_nodes": [ 10]}
 # last 30 days of data is test data 
nextwholemonth_return_models = run_nextwholemonth_return_pipeline  (pipeline, rfr_param_grid, [1 ,3, 5,15], testdays = 60)


cur_predictions_30days = {}     

for curr, curr_items in nextwholemonth_return_models.items():
  data = curr_items ["data"]
  models = curr_items["models"]   
  if(data.columns.contains("NextDayPrice")):
    data = data.drop(["NextDayPrice"], axis=1)
  
  wholemonth_model_predictions = pd.DataFrame()   

  
  for day, model in models.items():    
    y_pred_day= model.predict(data)
    wholemonth_model_predictions["date"] = data.index
    wholemonth_model_predictions["close"] = data["Close"].as_matrix()
    wholemonth_model_predictions[str(day)+"_day_return"] =y_pred_day 
    wholemonth_model_predictions[str(day)+"_day_price"] =\
              (1+wholemonth_model_predictions[str(day)+"_day_return"] )\
            * ( wholemonth_model_predictions["close"]  )
    wholemonth_model_predictions.drop([str(day)+"_day_return"], axis =1, inplace=True)
    wholemonth_model_predictions.set_index(wholemonth_model_predictions["date"], drop =True, inplace=True)
  cur_predictions_30days[curr] = wholemonth_model_predictions
  
  

for cur, pred in cur_predictions_30days.items():
   print ("Currency: " + cur)
   print(pred[-1:].transpose())
