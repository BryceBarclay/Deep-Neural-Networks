# Deep-Neural-Networks
This repository consists of code for assignments and projects for the class APM 598: Deep Neural Networks at ASU. 

## Getting Stock Data
To obtain stock data go https://finance.yahoo.com. Here:

(1) search a stock

(2) click on historical data

(3) edit the criteria and press apply

(4) click download data.

The data will download as a CSV file.

Alternatively, one can use the yfinance and the pandas_datareader.data libraries to download the data in a python script; however, this method has its difficulties. 

Edit: unfortunately yfinance along with other providers will only give away the daily data for free, and we would like to get smaller timeframe for our data (like 1 minute), since daily bars may hide significant fluctuations of price and we may need information on intraday extremums. The prices on smaller timeframes for US markets are varying, but we can just get some other data, like Russian stock market prices for free.

## Preparing the data
1. Characterize the data with respect to each question below to make sure we have a good variability for the classification problems.
2. Set up the train set to make sure that the proportions are good.

## Posing the question
Even though eventual goal of any sort market analysis is to get profit, there are different ways we can train neural network to try trading for profit later. We anticipate that the quality of our predictions will vary depending on what exactly we try to predict. We came up with different characteristic of the market and want to try to predict them. Below is the list of questions we want to ask our neural networks to find the answer for:
1. What will be the price *n* bars after the last one observed? (regression)
2. Will the price be above or below the last observed price *n* bars after it? (classification)
3. Will the price be *x* points higher/lower in *n* bars? (classification)
4. Will the price be *x* points higher/lower in *n* bars without going *y* points in the opposite direction? (classification)
5. Will the price in *n* following bars be consistently growing over each set of *n/5* bars? (classification)

## NN Architectures
For the project we thought it would be interesting to compare different kinds of architecures on this task, and we decided to go with the following types:
1. A regular fully-connected NN
2. A variation of RNN (and maybe separately LSTM)
3. A variation of CNN
4. Autoencoder (if we have time)

## Organization of Code
The code is organized so that modifications can be made for the Keras version of the project as follows:

1. to add/modify data or change problem parameters modify data_prep.py script
2. to add/modify models (CNN, GRU, LSTM, MLP) modify keras_models.py in utils
3. to modify hyperparameters modify main_keras.py script.

The results of a given run (the hyperparameters, train and test loss) will be saved into a file in the results folder with a time stamp of the run. 
