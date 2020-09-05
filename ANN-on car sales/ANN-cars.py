##import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##import dataset
car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')
print(
	car_df.head(5)
	)

##visualize dataset
plt.show(
	sns.pairplot(car_df)
	)

#Preprocesing data
clean_df = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
print(
	clean_df.head(5)
	)
output_df = car_df['Car Purchase Amount']
print(
	output_df
	)
print(
	clean_df.shape
	)
print(
	output_df.shape
	)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
clean_scaled = scaler.fit_transform(clean_df)	
print(
	clean_scaled
	)
print(
	scaler.data_max_
	)
print(
	scaler.data_min_
	)
output_df = output_df.values.reshape(-1,1)
output_scaled = scaler.fit_transform(output_df)
print(
	output_scaled
	)

##training the model
from sklearn.model_selection import train_test_split
clean_train, clean_test, output_train, output_test = train_test_split(clean_scaled, output_scaled, test_size = 0.3)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(25, input_dim = 5, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

epochs_hist = model.fit(clean_train, output_train, epochs = 100, batch_size = 25, verbose = 1, validation_split = 0.2)

##Evaluation of the model
epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Progress')
plt.ylabel('Training and Validation loss')
plt.xlabel('Epoch')
plt.legend(['Training loss','Validation loss'])
plt.show()

##Prediction
value_to_pred = np.array([[1, 50, 50000, 10000, 600000]])
pred_value =  model.predict(value_to_pred)
print('Expected Purchase Amount', pred_value)
