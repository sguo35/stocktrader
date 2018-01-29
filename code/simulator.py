from keras.models import load_model
model = load_model('./model.h5')
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler() 
import time

  
def simulate(timeseries):
    num_trials = 30
    # simulate num_trials of 2 weeks returns
    i = 0
    trials = []
    while i < num_trials:
        start = time.time()
        num_timesteps = 850
        j = 0

        stock_amt = 1000
        money = 10000
        returns = 1
        actions_x = []
        actions_y = []
        while j < num_timesteps:
            # simulate and add noise
            original_data = [timeseries[0+j:64+j]]
            
            reshaped = [min_max_scaler.transform(timeseries[0+j:64+j])]
            result = model.predict(np.array(reshaped))
            # add noise
            print(str(result[0]) + " Timestep " + str(j), end='\r')
            result = np.add(np.random.rand(1, 3), result[0])
            maxIndex = np.argmax(result)
            actions_x.append(reshaped)
            actions_y.append(maxIndex)
            if maxIndex == 0:
                # buy
                stock_amt += 1000 / timeseries[64+j-1][3]
                money -= 1020
            if maxIndex == 1:
                # sell
                stock_amt -= 1000 / timeseries[64+j-1][3]
                money += 980
            # next timestep
            j += 1
            if j == num_timesteps - 1:
                # calculate returns
                total = (stock_amt * timeseries[64+j-1][3]) + money
                returns = total / ((stock_amt * timeseries[63][3]) + 10000)
        trials.append([actions_x, actions_y, returns])
        print("Returns: " + str(returns))
        print("Trial # " + str(i) + " / " + str(num_trials))
        end = time.time()
        print(str(end - start) + " seconds / trial")
        i += 1
    indices = np.argpartition(trials, int(num_trials * 0.2))
    return trials[indices]


import os
files = os.listdir('../data/nyse_stocks/1')
import pandas as pd
import numpy as np

for file in files:
    print('Opening ' + file)
    data = pd.read_csv('../data/nyse_stocks/1/' + file, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])
    new_data = []
    data.drop(data.index[0])
    for row in data.as_matrix():
        new_row = [row[2], row[3], row[4], row[5], row[6]]
        new_data.append(new_row)
    if len(new_data) > 100:
        new_data = np.asfarray(new_data[1:], float)
        #print(min_max_scaler)
        min_max_scaler.fit(new_data)
        top_simulations = simulate(new_data)
        model.fit(top_simulations[:,0], top_simulations[:,1], epochs=1, batch_size=128, verbose=1)

