# ---------------------------------------------------------------------
# RNN4AP project
# Copyright (C) 2021-2022 ISAE
# 
# Purpose:
# Evaluation of Recurrent Neural Networks for future Autopilot Systems
# 
# Contact:
# jean-baptiste.chaudron@isae-supaero.fr
# ---------------------------------------------------------------------

# General import from Torch, Sklearn, Pandas etc...
from collections import deque
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Specific import for local defined RNN model
from cl_gru_model import cl_gru_model
from cl_lstm_model import cl_lstm_model

# Training configuration
num_epochs = 10 # you may put a number > 10 because of the logs otherwise comment the writing part in the csv file
batch_size = 1
seq_len = 5 #timesteps

# Rnn configuration
input_dim = 9
hidden_dim = 90
layer_dim = 3 # number of hidden layers with a size of hidden_dim then you should add the output Linear Layer
output_dim = 3

# ----------------------------------------------------------------------------------
# slinding window function
def sliding_windows(X_numpy, Y_numpy, seq_length):
    x = []
    y = []

    for i in range(len(X_numpy)-seq_length-1):
        _x = X_numpy[i:(i+seq_length)]
        _y = Y_numpy[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

# ----------------------------------------------------------------------------------
# low pass filter function
def low_pass_filter(X_numpy, Y_numpy, order):
    x = []
    y = []

    for i in range(len(X_numpy)):
        if i == 0:
            _x = X_numpy[i]
            _y = Y_numpy[i]
            x.append(_x)
            y.append(_y)
        else:
            _x = X_numpy[i] * order + (1-order) * x[i-1]
            _y = Y_numpy[i] * order + (1-order) * y[i-1]
            x.append(_x)
            y.append(_y)

    return np.array(x),np.array(y)

# GPU execution required
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
is_cuda = torch.cuda.is_available()

# Prepare dataset/loader (Note that this python scripts has to be called in the install folder not in the src...)
dataframe = pd.read_csv('../datasets/imu/D1.csv',dtype = np.float32)

#filters for csv data
filtX = dataframe.columns.str.contains('ax')  | dataframe.columns.str.contains('ay') | dataframe.columns.str.contains('az') | dataframe.columns.str.contains('gx')  | dataframe.columns.str.contains('gy') | dataframe.columns.str.contains('gz') | dataframe.columns.str.contains('mx')  | dataframe.columns.str.contains('my') | dataframe.columns.str.contains('mz')
filtY = dataframe.columns.str.contains('phi')  | dataframe.columns.str.contains('theta') | dataframe.columns.str.contains('psi')

#extract relevant data for training and testing
X_numpy = dataframe.loc[:,filtX].values
Y_numpy = dataframe.loc[:,filtY].values
print("X_numpy type is ", type(X_numpy))
print("X_numpy shape is ", X_numpy.shape)
print("Y_numpy shape is ", Y_numpy.shape)
#print(X_numpy)

X_numpy_filtered, Y_numpy_filtered = low_pass_filter(X_numpy, Y_numpy, 0.2)

file = open("D1_filtered.csv", mode="w")
file.write("ax,ay,az,gx,gy,gz,mx,my,mz,ax_lp,ay_lp,az_lp,gx_lp,gy_lp,gz_lp,mx_lp,my_lp,mz_lp\n")
for i in range(len(X_numpy)):
	file.write(str(X_numpy[i][0]) + "," + str(X_numpy[i][1]) + "," + str(X_numpy[i][2]) + "," ) #ax, ay, az
	file.write(str(X_numpy[i][3]) + "," + str(X_numpy[i][4]) + "," + str(X_numpy[i][5]) + "," ) #gx, gy, gz
	file.write(str(X_numpy[i][6]) + "," + str(X_numpy[i][7]) + "," + str(X_numpy[i][8]) + "," ) #gx, gy, gz
	file.write(str(X_numpy_filtered[i][0]) + "," + str(X_numpy_filtered[i][1]) + "," + str(X_numpy_filtered[i][2]) + "," ) #ax, ay, az
	file.write(str(X_numpy_filtered[i][3]) + "," + str(X_numpy_filtered[i][4]) + "," + str(X_numpy_filtered[i][5]) + "," ) #gx, gy, gz
	file.write(str(X_numpy_filtered[i][6]) + "," + str(X_numpy_filtered[i][7]) + "," + str(X_numpy_filtered[i][8]) + "\n" ) #gx, gy, gz
file.close()

# scalerX = preprocessing.Normalizer().fit(X_numpy_filtered) #MinMaxScaler, Normalizer, here we are using low pass filtered data
# scalerY = preprocessing.Normalizer().fit(Y_numpy) #MinMaxScaler, Normalizer

#scalerX = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_numpy_filtered) 
scalerX = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_numpy) 
scalerY = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(Y_numpy) 

# scale data for inputs
X_numpy_scaled = scalerX.transform(X_numpy)
Y_numpy_scaled = Y_numpy # scalerY.transform(Y_numpy)
print("X_numpy_scaled shape is ", X_numpy_scaled.shape)

X_numpy_seq, Y_numpy_seq = sliding_windows(X_numpy_scaled, Y_numpy_scaled, seq_len) #here we can used scaled data or not
print("X_numpy_seq type is ", type(X_numpy_seq))
print("X_numpy_seq shape is ", X_numpy_seq.shape)
print("Y_numpy_seq shape is ", Y_numpy_seq.shape)

# LSTM for IMU https://github.com/nsparag/LSTM-INC/tree/main/Results
# train test split. Size of train data is 70% and size of test data is 30%. 
X_train, X_test, Y_train, Y_test = train_test_split(X_numpy_seq,Y_numpy_seq,test_size = 0.3,random_state=None,shuffle=False)
print("X_train type is ", type(X_train))
print("X_train shape is ", X_train.shape)
print("X_test shape is ", X_test.shape)
print("Y_train shape is ", Y_train.shape)
print("Y_test shape is ", Y_test.shape)

print("X_train type is ", type(X_train))

file = open("D1_training.csv", mode="w")
file.write("ax,ay,az,gx,gy,gz,mx,my,mz,ax_lp,ay_lp,az_lp,gx_lp,gy_lp,gz_lp,mx_lp,my_lp,mz_lp\n")
for i in range(len(X_train)):
	file.write(str(X_train[i][4][0]) + "," + str(X_train[i][4][1]) + "," + str(X_train[i][4][2]) + "," ) #ax, ay, az from pytorch training
	file.write(str(X_train[i][4][3]) + "," + str(X_train[i][4][4]) + "," + str(X_train[i][4][5]) + "," ) #gx, gy, gz
	file.write(str(X_train[i][4][6]) + "," + str(X_train[i][4][7]) + "," + str(X_train[i][4][8]) + "," ) #gx, gy, gz
	file.write(str(X_numpy_filtered[i][0]) + "," + str(X_numpy_filtered[i][1]) + "," + str(X_numpy_filtered[i][2]) + "," ) #ax, ay, az
	file.write(str(X_numpy_filtered[i][3]) + "," + str(X_numpy_filtered[i][4]) + "," + str(X_numpy_filtered[i][5]) + "," ) #gx, gy, gz
	file.write(str(X_numpy_filtered[i][6]) + "," + str(X_numpy_filtered[i][7]) + "," + str(X_numpy_filtered[i][8]) + "\n" ) #gx, gy, gz
file.close()

# create feature and targets tensor for train set. 
X_train_torch = torch.from_numpy(X_train)
Y_train_torch = torch.from_numpy(Y_train) 

# create feature and targets tensor for train set. 
X_test_torch = torch.from_numpy(X_test)
Y_test_torch = torch.from_numpy(Y_test) 

# Pytorch train and test sets
train_dataset = TensorDataset(X_train_torch,Y_train_torch)
test_dataset = TensorDataset(X_test_torch,Y_test_torch)

# data loaders
train_loader_shuffled = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)#, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)#, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)#, num_workers=4)

#lstm mmodel instance
lstm_model_instance = cl_lstm_model(input_dim, hidden_dim, layer_dim, output_dim)

#lstm mmodel instance
gru_model_instance = cl_gru_model(input_dim, hidden_dim, layer_dim, output_dim)

# enabling CUDA for model
if is_cuda:
    lstm_model_instance.cuda()
    gru_model_instance.cuda()

# MSE
error_lstm = nn.MSELoss()
error_gru = nn.MSELoss()

learning_rate = 0.005
optimizer_lstm = torch.optim.SGD(lstm_model_instance.parameters(), lr=learning_rate, momentum=0.9) 
optimizer_gru  = torch.optim.SGD(gru_model_instance.parameters(), lr=learning_rate, momentum=0.9) 

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  betas=(0.9, 0.999)) 

ref_phi = []
ref_theta = []
ref_psi = []

lstm_phi_01 = []
lstm_theta_01 = []
lstm_psi_01 = []

lstm_phi_02 = []
lstm_theta_02 = []
lstm_psi_02 = []

lstm_phi_03 = []
lstm_theta_03 = []
lstm_psi_03 = []

lstm_phi_04 = []
lstm_theta_04 = []
lstm_psi_04 = []

lstm_phi_05 = []
lstm_theta_05 = []
lstm_psi_05 = []

lstm_phi_06 = []
lstm_theta_06 = []
lstm_psi_06 = []

lstm_phi_07 = []
lstm_theta_07 = []
lstm_psi_07 = []

lstm_phi_08 = []
lstm_theta_08 = []
lstm_psi_08 = []

lstm_phi_09 = []
lstm_theta_09 = []
lstm_psi_09 = []

lstm_phi_10 = []
lstm_theta_10 = []
lstm_psi_10 = []

gru_phi_01 = []
gru_theta_01 = []
gru_psi_01 = []

gru_phi_02 = []
gru_theta_02 = []
gru_psi_02 = []

gru_phi_03 = []
gru_theta_03 = []
gru_psi_03 = []

gru_phi_04 = []
gru_theta_04 = []
gru_psi_04 = []

gru_phi_05 = []
gru_theta_05 = []
gru_psi_05 = []

gru_phi_06 = []
gru_theta_06 = []
gru_psi_06 = []

gru_phi_07 = []
gru_theta_07 = []
gru_psi_07 = []

gru_phi_08 = []
gru_theta_08 = []
gru_psi_08 = []

gru_phi_09 = []
gru_theta_09 = []
gru_psi_09 = []

gru_phi_10 = []
gru_theta_10 = []
gru_psi_10 = []

outputs_test_cpu_01 = []
outputs_test_cpu_02 = []
outputs_test_cpu_03 = []
outputs_test_cpu_04 = []
outputs_test_cpu_05 = []
outputs_test_cpu_06 = []
outputs_test_cpu_07 = []
outputs_test_cpu_08 = []
outputs_test_cpu_09 = []
outputs_test_cpu_10 = []

loss_list = []
iteration_list = []
accuracy_list = []
epoch_counter = 1
#results_dataframe = pd.DataFrame()
label_notwritten = True

loss_training_total = 0

for epoch in range(num_epochs):
	total_train_lstm = 0
	total_train_gru = 0
	iter_train = 0
	for i, (X, Y) in enumerate(train_loader_shuffled):

		if is_cuda:
			X = X.cuda()
			Y = Y.cuda()

		# Clear gradients w.r.t. parameters
		optimizer_lstm.zero_grad()
		optimizer_gru.zero_grad()

		# Forward pass to get output/logits
		outputs_lstm = lstm_model_instance(X)
		outputs_gru = gru_model_instance(X)

		# Calculate MSE Loss for LSTM
		loss_lstm = error_lstm(outputs_lstm, Y)
		total_train_lstm += loss_lstm

		# Calculate MSE Loss for GRU
		loss_gru = error_gru(outputs_gru, Y)
		total_train_gru += loss_gru

		iter_train += 1

		# Getting gradients
		loss_lstm.backward()
		loss_gru.backward()

		# Updating parameters
		optimizer_lstm.step()
		optimizer_gru.step()

	loss_train_lstm = total_train_lstm/iter_train
	loss_train_gru = total_train_gru/iter_train

	if epoch_counter % 1 == 0:
		# Calculate Accuracy         
		correct = 0
		total_test_lstm = 0
		total_test_gru = 0
		outputs_test_vect = []
		y_test_vect = []
		iter_test = 0
		t_vect = []

		for X_test, Y_test in test_loader:
			
			if is_cuda:
				X_test = X_test.cuda()
				Y_test = Y_test.cuda()

			# Forward pass only to get logits/output
			outputs_test_lstm = lstm_model_instance(X_test)
			outputs_test_gru = gru_model_instance(X_test)

			if (label_notwritten):
				Y_test_cpu = Y_test.cpu().numpy()
				ref_phi.append(Y_test_cpu[0][0])
				ref_theta.append(Y_test_cpu[0][1])
				ref_psi.append(Y_test_cpu[0][2])

			if (epoch_counter == 1):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_01.append(outputs_test_cpu[0][0])
				lstm_theta_01.append(outputs_test_cpu[0][1])
				lstm_psi_01.append(outputs_test_cpu[0][2])
				outputs_test_cpu_01.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_01.append(outputs_test_cpu[0][0])
				gru_theta_01.append(outputs_test_cpu[0][1])
				gru_psi_01.append(outputs_test_cpu[0][2])

			if (epoch_counter == 2):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_02.append(outputs_test_cpu[0][0])
				lstm_theta_02.append(outputs_test_cpu[0][1])
				lstm_psi_02.append(outputs_test_cpu[0][2])
				outputs_test_cpu_02.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				gru_phi_02.append(outputs_test_cpu[0][0])
				gru_theta_02.append(outputs_test_cpu[0][1])
				gru_psi_02.append(outputs_test_cpu[0][2])

			if (epoch_counter == 3):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_03.append(outputs_test_cpu[0][0])
				lstm_theta_03.append(outputs_test_cpu[0][1])
				lstm_psi_03.append(outputs_test_cpu[0][2])
				outputs_test_cpu_03.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_03.append(outputs_test_cpu[0][0])
				gru_theta_03.append(outputs_test_cpu[0][1])
				gru_psi_03.append(outputs_test_cpu[0][2])

			if (epoch_counter == 4):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_04.append(outputs_test_cpu[0][0])
				lstm_theta_04.append(outputs_test_cpu[0][1])
				lstm_psi_04.append(outputs_test_cpu[0][2])
				outputs_test_cpu_04.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_04.append(outputs_test_cpu[0][0])
				gru_theta_04.append(outputs_test_cpu[0][1])
				gru_psi_04.append(outputs_test_cpu[0][2])
				

			if (epoch_counter == 5):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_05.append(outputs_test_cpu[0][0])
				lstm_theta_05.append(outputs_test_cpu[0][1])
				lstm_psi_05.append(outputs_test_cpu[0][2])
				outputs_test_cpu_05.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_05.append(outputs_test_cpu[0][0])
				gru_theta_05.append(outputs_test_cpu[0][1])
				gru_psi_05.append(outputs_test_cpu[0][2])

			if (epoch_counter == 6):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_06.append(outputs_test_cpu[0][0])
				lstm_theta_06.append(outputs_test_cpu[0][1])
				lstm_psi_06.append(outputs_test_cpu[0][2])
				outputs_test_cpu_06.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_06.append(outputs_test_cpu[0][0])
				gru_theta_06.append(outputs_test_cpu[0][1])
				gru_psi_06.append(outputs_test_cpu[0][2])

			if (epoch_counter == 7):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_07.append(outputs_test_cpu[0][0])
				lstm_theta_07.append(outputs_test_cpu[0][1])
				lstm_psi_07.append(outputs_test_cpu[0][2])
				outputs_test_cpu_07.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_07.append(outputs_test_cpu[0][0])
				gru_theta_07.append(outputs_test_cpu[0][1])
				gru_psi_07.append(outputs_test_cpu[0][2])

			if (epoch_counter == 8):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_08.append(outputs_test_cpu[0][0])
				lstm_theta_08.append(outputs_test_cpu[0][1])
				lstm_psi_08.append(outputs_test_cpu[0][2])
				outputs_test_cpu_08.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_08.append(outputs_test_cpu[0][0])
				gru_theta_08.append(outputs_test_cpu[0][1])
				gru_psi_08.append(outputs_test_cpu[0][2])

			if (epoch_counter == 9):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_09.append(outputs_test_cpu[0][0])
				lstm_theta_09.append(outputs_test_cpu[0][1])
				lstm_psi_09.append(outputs_test_cpu[0][2])
				outputs_test_cpu_09.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_09.append(outputs_test_cpu[0][0])
				gru_theta_09.append(outputs_test_cpu[0][1])
				gru_psi_09.append(outputs_test_cpu[0][2])

			if (epoch_counter == 10):
				outputs_test_cpu = outputs_test_lstm.data.cpu().numpy()
				lstm_phi_10.append(outputs_test_cpu[0][0])
				lstm_theta_10.append(outputs_test_cpu[0][1])
				lstm_psi_10.append(outputs_test_cpu[0][2])
				outputs_test_cpu_10.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru.data.cpu().numpy()
				gru_phi_10.append(outputs_test_cpu[0][0])
				gru_theta_10.append(outputs_test_cpu[0][1])
				gru_psi_10.append(outputs_test_cpu[0][2])

			# Get predictions from the maximum value
			loss_test_lstm = error_lstm(outputs_test_lstm.data, Y_test)
			loss_test_gru = error_gru(outputs_test_gru.data, Y_test)

			# Total number of labels
			total_test_lstm += loss_test_lstm
			total_test_gru += loss_test_gru
			t_vect.append(iter_test)
			iter_test += 1

			# Total correct predictions

		if (label_notwritten):
			label_notwritten = False

		loss_test_lstm = total_test_lstm/iter_test
		loss_test_gru = total_test_gru/iter_test

		#loss_list.append(loss.data.item())
		#iteration_list.append(epoch_counter)
		#accuracy_list.append(loss_test)
		
		# Print Loss
		print('-----')
		print('Epoch: {}. LSTM Training Loss: {}. Testing Loss: {}'.format(epoch_counter, loss_train_lstm, loss_test_lstm))
		print('Epoch: {}. GRU  Training Loss: {}. Testing Loss: {}'.format(epoch_counter, loss_test_gru, loss_test_gru))
	epoch_counter += 1

outputs_test_cpu_1_filtered, outputs_test_cpu_2_filtered = low_pass_filter(np.array(outputs_test_cpu_01), np.array(outputs_test_cpu_02), 0.2)
outputs_test_cpu_3_filtered, outputs_test_cpu_4_filtered = low_pass_filter(np.array(outputs_test_cpu_03), np.array(outputs_test_cpu_04), 0.2)


# write results in a csv file
file = open("D1_results_lstm.csv", mode="w")
file.write("ref_phi,ref_theta,ref_psi,")
file.write("lstm_phi_epoch_01,lstm_theta_epoch_01,lstm_psi_epoch_01,")
file.write("lstm_phi_epoch_02,lstm_theta_epoch_02,lstm_psi_epoch_02,")
file.write("lstm_phi_epoch_03,lstm_theta_epoch_03,lstm_psi_epoch_03,")
file.write("lstm_phi_epoch_04,lstm_theta_epoch_04,lstm_psi_epoch_04,")
file.write("lstm_phi_epoch_05,lstm_theta_epoch_05,lstm_psi_epoch_05,")
file.write("lstm_phi_epoch_06,lstm_theta_epoch_06,lstm_psi_epoch_06,")
file.write("lstm_phi_epoch_07,lstm_theta_epoch_07,lstm_psi_epoch_07,")
file.write("lstm_phi_epoch_08,lstm_theta_epoch_08,lstm_psi_epoch_08,")
file.write("lstm_phi_epoch_09,lstm_theta_epoch_09,lstm_psi_epoch_09,")
file.write("lstm_phi_epoch_10,lstm_theta_epoch_10,lstm_psi_epoch_10,")
file.write("gru_phi_epoch_01,gru_theta_epoch_01,gru_psi_epoch_01,")
file.write("gru_phi_epoch_02,gru_theta_epoch_02,gru_psi_epoch_02,")
file.write("gru_phi_epoch_03,gru_theta_epoch_03,gru_psi_epoch_03,")
file.write("gru_phi_epoch_04,gru_theta_epoch_04,gru_psi_epoch_04,")
file.write("gru_phi_epoch_05,gru_theta_epoch_05,gru_psi_epoch_05,")
file.write("gru_phi_epoch_06,gru_theta_epoch_06,gru_psi_epoch_06,")
file.write("gru_phi_epoch_07,gru_theta_epoch_07,gru_psi_epoch_07,")
file.write("gru_phi_epoch_08,gru_theta_epoch_08,gru_psi_epoch_08,")
file.write("gru_phi_epoch_09,gru_theta_epoch_09,gru_psi_epoch_09,")
file.write("gru_phi_epoch_10,gru_theta_epoch_10,gru_psi_epoch_10,")
file.write("lstm_phi_0_lp,lstm_theta_0_lp,lstm_psi_0_lp,lstm_phi_1_lp,lstm_theta_1_lp,lstm_psi_1_lp,lstm_phi_2_lp,lstm_theta_2_lp,lstm_psi_2_lp,lstm_phi_3_lp,lstm_theta_3_lp,lstm_psi_3_lp\n")
for i, (X_test, Y_test) in enumerate(test_loader):
	file.write(str(ref_phi[i]) + "," + str(ref_theta[i]) + "," + str(ref_psi[i]) + "," )
	file.write(str(lstm_phi_01[i]) + "," + str(lstm_theta_01[i]) + "," + str(lstm_psi_01[i]) + ",")
	file.write(str(lstm_phi_02[i]) + "," + str(lstm_theta_02[i]) + "," + str(lstm_psi_02[i]) + ",")
	file.write(str(lstm_phi_03[i]) + "," + str(lstm_theta_03[i]) + "," + str(lstm_psi_03[i]) + ",")
	file.write(str(lstm_phi_04[i]) + "," + str(lstm_theta_04[i]) + "," + str(lstm_psi_04[i]) + ",")
	file.write(str(lstm_phi_05[i]) + "," + str(lstm_theta_05[i]) + "," + str(lstm_psi_05[i]) + ",")
	file.write(str(lstm_phi_06[i]) + "," + str(lstm_theta_06[i]) + "," + str(lstm_psi_06[i]) + ",")
	file.write(str(lstm_phi_07[i]) + "," + str(lstm_theta_07[i]) + "," + str(lstm_psi_07[i]) + ",")
	file.write(str(lstm_phi_08[i]) + "," + str(lstm_theta_08[i]) + "," + str(lstm_psi_08[i]) + ",")
	file.write(str(lstm_phi_09[i]) + "," + str(lstm_theta_09[i]) + "," + str(lstm_psi_09[i]) + ",")
	file.write(str(lstm_phi_10[i]) + "," + str(lstm_theta_10[i]) + "," + str(lstm_psi_10[i]) + ",")
	file.write(str(gru_phi_01[i]) + "," + str(gru_theta_01[i]) + "," + str(gru_psi_01[i]) + ",")
	file.write(str(gru_phi_02[i]) + "," + str(gru_theta_02[i]) + "," + str(gru_psi_02[i]) + ",")
	file.write(str(gru_phi_03[i]) + "," + str(gru_theta_03[i]) + "," + str(gru_psi_03[i]) + ",")
	file.write(str(gru_phi_04[i]) + "," + str(gru_theta_04[i]) + "," + str(gru_psi_04[i]) + ",")
	file.write(str(gru_phi_05[i]) + "," + str(gru_theta_05[i]) + "," + str(gru_psi_05[i]) + ",")
	file.write(str(gru_phi_06[i]) + "," + str(gru_theta_06[i]) + "," + str(gru_psi_06[i]) + ",")
	file.write(str(gru_phi_07[i]) + "," + str(gru_theta_07[i]) + "," + str(gru_psi_07[i]) + ",")
	file.write(str(gru_phi_08[i]) + "," + str(gru_theta_08[i]) + "," + str(gru_psi_08[i]) + ",")
	file.write(str(gru_phi_09[i]) + "," + str(gru_theta_09[i]) + "," + str(gru_psi_09[i]) + ",")
	file.write(str(gru_phi_10[i]) + "," + str(gru_theta_10[i]) + "," + str(gru_psi_10[i]) + ",")
	file.write(str(outputs_test_cpu_1_filtered[i][0]) + "," + str(outputs_test_cpu_1_filtered[i][1]) + "," + str(outputs_test_cpu_1_filtered[i][2]) + "," ) 
	file.write(str(outputs_test_cpu_2_filtered[i][0]) + "," + str(outputs_test_cpu_2_filtered[i][1]) + "," + str(outputs_test_cpu_2_filtered[i][2]) + "," ) 
	file.write(str(outputs_test_cpu_3_filtered[i][0]) + "," + str(outputs_test_cpu_3_filtered[i][1]) + "," + str(outputs_test_cpu_3_filtered[i][2]) + "," ) 
	file.write(str(outputs_test_cpu_4_filtered[i][0]) + "," + str(outputs_test_cpu_4_filtered[i][1]) + "," + str(outputs_test_cpu_4_filtered[i][2]) + "\n" )
file.close()
