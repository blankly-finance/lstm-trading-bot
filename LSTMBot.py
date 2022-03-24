import blankly
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.optim as optim
from torch.autograd import Variable

def episode_gen(data, seq_length,output_size):
    x = []
    y = []
    #Loop through data, adding input data to x array and output data to y array
    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length - output_size)]
        _y = data[i+seq_length - output_size:i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def init_NN(symbol, state: blankly.StrategyState):
    interface = state.interface
    resolution = state.resolution
    variables = state.variables

    #Get price data
    variables['history'] = interface.history(symbol, 300, resolution, return_as='list')['close']
    '''We use Blankly's built-in indicator functions to calculate
    indicators we can use along with price data to predict future prices
    '''
    rsi = blankly.indicators.rsi(state.variables['history'])
    macd = blankly.indicators.macd(state.variables['history'])

    '''We'll break the historical Ethereum data into 8 day episodes
    and attempt to predict the final three days using the first 5.
    '''
    seq_length = 8
    output_size = 3

    '''Feature engineering -- here, we calculate the price change from the day before
    as a ratio. This is useful because it means we have less issues with scaling with the model,
    as the data will all already be in roughly the same range. We ignore the first 25 elements
    because we want every observation to have corresponding RSI + MACD data, and MACD requires
    26 periods.
    '''
    x = [variables['history'][i] / variables['history'][i-1] for i in range(25,len(variables['history']))]
    x, y = episode_gen(x, seq_length, output_size)
    y = Variable(torch.Tensor(np.array(y))).unsqueeze(0)
    #RSI data gathering
    x_rsi = rsi[11:]
    x_rsi,_ = episode_gen(x_rsi,seq_length, output_size)

    #MACD data gathering
    macd_vals,_ = episode_gen(macd[0], seq_length, output_size)
    macd_signals,_ = episode_gen(macd[1],seq_length, output_size)

    '''In this section, we put all the features we just extracted into one NumPy array
    which we then convert to a PyTorch tensor that we can run our model on.
    '''
    x_agg = np.zeros((len(macd_signals),seq_length-output_size, 4))
    for i in range(len(macd_signals)):
      for j in range(seq_length - output_size):
        x_agg[i][j][0] = x[i][j]
        x_agg[i][j][1] = x_rsi[i][j]
        x_agg[i][j][2] = macd_vals[i][j]
        x_agg[i][j][3] = macd_signals[i][j]
    x_tot = Variable(torch.Tensor(x_agg))
    '''Here's our training loop! We have a fairly small dataset, so we can run through 10,000 epochs pretty quickly
    Here's also the first place we see our model architecture. We use an LSTM that takes in 4 features and has 
    a hidden state with dimension 20 and feed the hidden state into a linear neural network layer and sigmoid 
    activation that output 3 numbers, the predicted prices for the next three days. We use mean-squared-error loss
    and an Adam optimizer
    '''
    num_epochs = 4000
    learning_rate = 0.00005


    state.lstm = LSTM(4,20, batch_first = True)
    state.lin = nn.Linear(20,3)
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    #Optimizer: make sure to include parameters of both linear layer and LSTM
    optimizer = optim.Adam([
                {'params': state.lstm.parameters()},
                {'params': state.lin.parameters()}
            ], lr=learning_rate)
    state.lstm.load_state_dict(torch.load('lstm_pk.pth'))
    state.lin.load_state_dict(torch.load('lin_pk.pth'))
    # Train the model

    for epoch in range(num_epochs):
        #run model
        outputs, (h_n, c_n) = state.lstm(x_tot)
        out = state.lin(h_n)
        out = 0.5 + F.sigmoid(out)
        optimizer.zero_grad()
        
        loss = criterion(out, y) #calculate loss function
        
        loss.backward() #backprop
        
        optimizer.step() #gradient descent

        #Output loss functions every 500 epochs so we can make sure the model is training
        if epoch % 500 == 0:
          print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    '''We use this in the trading algorithm for more stability.
    Essentially, instead of relying on a single output of the model 
    to tell us whether to buy or sell, we average the readings from three different calculations
    (3 days before, 2 days before, day before)
    '''    
    state.lastthree = [[0,0],[0,0],[0,0]] 
    #torch.save(state.lstm.state_dict(), 'lstm_pm.pth')
    #torch.save(state.lin.state_dict(),'lin_pm.pth')
    '''If we want, we can save the model parameters for future use.
    '''

def price_lstm(price,symbol,state: blankly.StrategyState):
    state.variables['history'].append(price) #Add latest price to current list of data

    '''Here, we pull the data from the last few days, prepare it,
    and run the necessary indicator functions to feed into our model
    '''
    into = [state.variables['history'][i]/state.variables['history'][i-1] for i in range(-5,0)]
    rsi = blankly.indicators.rsi(state.variables['history'])
    rsi_in = np.array(rsi[-5:])
    macd = blankly.indicators.macd(state.variables['history'])
    macd_vals = np.array(macd[0][-5:])
    macd_signals = np.array(macd[1][-5:])

    '''We put the data into the torch Tensor that we'll run the model on
    '''
    pred_in = np.zeros((1,len(into),4))
    for i in range(len(into)):
      pred_in[0][i][0] = into[i]
      pred_in[0][i][1] = rsi_in[i]
      pred_in[0][i][2] = macd_vals[i]
      pred_in[0][i][3] = macd_signals[i]
    pred_in = torch.Tensor(pred_in)
  
    #print(price) #Print the price so we can see what's going on

    '''Run the data through the trained model. 
    The field out stores the prediction values we want
    '''
    out,(h,c) = state.lstm(pred_in)
    out = state.lin(h)
    out = 0.5 + F.sigmoid(out)
    
    '''This definitely could be shortened with a loop,
    but basically, we add the percentage increase to the other values in the
    3-day-average array. We also increment a counter showing how many values have been
    added before averaging. This handles the edge case of the first few values (where
    we wouldn't divide by 3)
    '''
    state.lastthree[0][0]+=out[0][0][0]
    state.lastthree[0][1]+=1
    state.lastthree[1][0]+=out[0][0][1]
    state.lastthree[1][1]+=1
    state.lastthree[2][0]+=out[0][0][2]
    state.lastthree[2][1]+=1

    '''The avg price increase is calculated by dividing the sum of next day predictions
    by the number of predictions for the next day.
    '''
    priceavg = state.lastthree[0][0]/state.lastthree[0][1]

    curr_value = blankly.trunc(state.interface.account[state.base_asset].available, 2) #Amount of Ethereum available
    if priceavg > 1:
        # If we think price will increase, we buy
        buy = blankly.trunc(state.interface.cash   * 1.6* (priceavg.item() - 1)/price, 2) #Buy an amount proportional to priceavg - 1
        if buy > 0:
          state.interface.market_order(symbol, side='buy', size=buy)
    elif curr_value > 0:
        #If we think price will decrease, we sell
         cv =  blankly.trunc(curr_value  * 1.6 * (1 - priceavg.item()),2) #Sell an amount proportional to 1 - priceavg
         if cv > 0:
          state.interface.market_order(symbol, side='sell', size=cv)

    #print("prediction for price --",priceavg) #Print so we can see what's happening
    state.lastthree = [state.lastthree[1], state.lastthree[2], [0,0]] #Shift the values in our 3-day-average array

if __name__ == '__main__':
    exchange = blankly.FTX() #Connect to FTX API
    strategy = blankly.Strategy(exchange) #Initialize a Blankly strategy
    strategy.add_price_event(price_lstm, symbol='ETH-USD', resolution='1d', init=init_NN) #Add our price event and initialization
    results = strategy.backtest(to='1y', initial_values={'USD': 10000}) #Backtest one year starting with $10,000
    with open('backtest_results.txt', 'w+') as br:
        br.write(str(results))
