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
hidden_dim = 50
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
dataframe = pd.read_csv('../datasets/imu/D3.csv',dtype = np.float32)

#filters for csv data
filtX = dataframe.columns.str.contains('ax')  | dataframe.columns.str.contains('ay') | dataframe.columns.str.contains('az') | dataframe.columns.str.contains('gx')  | dataframe.columns.str.contains('gy') | dataframe.columns.str.contains('gz') | dataframe.columns.str.contains('mx')  | dataframe.columns.str.contains('my') | dataframe.columns.str.contains('mz')
filtY = dataframe.columns.str.contains('phi')  | dataframe.columns.str.contains('theta') | dataframe.columns.str.contains('psi')

filtY_phi = dataframe.columns.str.contains('phi') 
filtY_theta = dataframe.columns.str.contains('theta') 

#extract relevant data for training and testing
X_numpy = dataframe.loc[:,filtX].values
Y_numpy = dataframe.loc[:,filtY].values
print("X_numpy type is ", type(X_numpy))
print("X_numpy shape is ", X_numpy.shape)
print("Y_numpy shape is ", Y_numpy.shape)
#print(X_numpy)

X_numpy_filtered, Y_numpy_filtered = low_pass_filter(X_numpy, Y_numpy, 0.2)

file = open("D3_filtered_low_pass.csv", mode="w")
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
scalerX = preprocessing.MinMaxScaler(feature_range=(-0.9, 0.9)).fit(X_numpy) 
scalerY = preprocessing.MinMaxScaler(feature_range=(-0.9, 0.9)).fit(Y_numpy) 

# scale data for inputs
X_numpy_scaled = scalerX.transform(X_numpy)
Y_numpy_scaled = scalerY.transform(Y_numpy)
print("X_numpy_scaled shape is ", X_numpy_scaled.shape)

file = open("D3_scaled_pytorch.csv", mode="w")
file.write("ax,ay,az,gx,gy,gz,mx,my,mz,phi,theta,psi\n")
for i in range(len(X_numpy)):
	file.write(str(X_numpy_scaled[i][0]) + "," + str(X_numpy_scaled[i][1]) + "," + str(X_numpy_scaled[i][2]) + "," ) #ax, ay, az
	file.write(str(X_numpy_scaled[i][3]) + "," + str(X_numpy_scaled[i][4]) + "," + str(X_numpy_scaled[i][5]) + "," ) #gx, gy, gz
	file.write(str(X_numpy_scaled[i][6]) + "," + str(X_numpy_scaled[i][7]) + "," + str(X_numpy_scaled[i][8]) + "," ) #mx, my, mz
	file.write(str(Y_numpy_scaled[i][0]) + "," + str(Y_numpy_scaled[i][1]) + "," + str(Y_numpy_scaled[i][2]) + "\n" ) #phi, theta, psi
file.close()

X_numpy_seq, Y_numpy_seq = sliding_windows(X_numpy_scaled, Y_numpy_scaled, seq_len) #here we can used scaled data or not
print("X_numpy_seq type is ", type(X_numpy_seq))
print("X_numpy_seq shape is ", X_numpy_seq.shape)
print("Y_numpy_seq shape is ", Y_numpy_seq.shape)

# train test split. Size of train data is 80% and size of test data is 20%. 
X_train, X_test, Y_train, Y_test = train_test_split(X_numpy_seq,Y_numpy_seq,test_size = 0.3,random_state=None,shuffle=False)
print("X_train type is ", type(X_train))
print("X_train shape is ", X_train.shape)
print("X_test shape is ", X_test.shape)
print("Y_train shape is ", Y_train.shape)
print("Y_test shape is ", Y_test.shape)
print("X_train type is ", type(X_train))


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

#lstm_all mmodel instance
model_instance_lstm_all = cl_lstm_model(input_dim, hidden_dim, layer_dim, output_dim)
model_instance_lstm_phi = cl_lstm_model(input_dim, hidden_dim, layer_dim, 1)
model_instance_lstm_theta = cl_lstm_model(input_dim, hidden_dim, layer_dim, 1)
model_instance_lstm_psi = cl_lstm_model(input_dim, hidden_dim, layer_dim, 1)

#gru mmodel instance
model_instance_gru_all = cl_gru_model(input_dim, hidden_dim, layer_dim, output_dim)
model_instance_gru_phi = cl_lstm_model(input_dim, hidden_dim, layer_dim, 1)
model_instance_gru_theta = cl_lstm_model(input_dim, hidden_dim, layer_dim, 1)
model_instance_gru_psi = cl_lstm_model(input_dim, hidden_dim, layer_dim, 1)

# enabling CUDA for model
if is_cuda:
	model_instance_lstm_all.cuda()
	model_instance_lstm_phi.cuda()
	model_instance_lstm_theta.cuda()
	model_instance_lstm_psi.cuda()
	model_instance_gru_all.cuda()
	model_instance_gru_phi.cuda()
	model_instance_gru_theta.cuda()
	model_instance_gru_psi.cuda()

# MSE LSTM
error_lstm_all = nn.MSELoss()
error_lstm_phi = nn.MSELoss()
error_lstm_theta = nn.MSELoss()
error_lstm_psi = nn.MSELoss()
error_gru_all = nn.MSELoss()
error_gru_phi = nn.MSELoss()
error_gru_theta = nn.MSELoss()
error_gru_psi = nn.MSELoss()

# OPTIMIZERS
learning_rate = 0.0005
#optimizer_lstm_all = torch.optim.SGD(lstm_all_model_instance.parameters(), lr=learning_rate, momentum=0.9) 
#optimizer_gru  = torch.optim.SGD(gru_model_instance.parameters(), lr=learning_rate, momentum=0.9) 

optimizer_lstm_all = torch.optim.Adam(model_instance_lstm_all.parameters(), lr=learning_rate, betas=(0.9, 0.999)) 
optimizer_lstm_phi = torch.optim.Adam(model_instance_lstm_phi.parameters(), lr=learning_rate, betas=(0.9, 0.999)) 
optimizer_lstm_theta = torch.optim.Adam(model_instance_lstm_theta.parameters(), lr=learning_rate, betas=(0.9, 0.999))
optimizer_lstm_psi = torch.optim.Adam(model_instance_lstm_psi.parameters(), lr=learning_rate, betas=(0.9, 0.999))

optimizer_gru_all  = torch.optim.Adam(model_instance_gru_all.parameters(), lr=learning_rate, betas=(0.9, 0.999)) 
optimizer_gru_phi  = torch.optim.Adam(model_instance_gru_phi.parameters(), lr=learning_rate, betas=(0.9, 0.999)) 
optimizer_gru_theta  = torch.optim.Adam(model_instance_gru_theta.parameters(), lr=learning_rate, betas=(0.9, 0.999)) 
optimizer_gru_psi  = torch.optim.Adam(model_instance_gru_psi.parameters(), lr=learning_rate, betas=(0.9, 0.999)) 

ref_phi = []
ref_theta = []
ref_psi = []

lstm_all_phi_01 = []
lstm_all_theta_01 = []
lstm_all_psi_01 = []

lstm_all_phi_02 = []
lstm_all_theta_02 = []
lstm_all_psi_02 = []

lstm_all_phi_03 = []
lstm_all_theta_03 = []
lstm_all_psi_03 = []

lstm_all_phi_04 = []
lstm_all_theta_04 = []
lstm_all_psi_04 = []

lstm_all_phi_05 = []
lstm_all_theta_05 = []
lstm_all_psi_05 = []

lstm_all_phi_06 = []
lstm_all_theta_06 = []
lstm_all_psi_06 = []

lstm_all_phi_07 = []
lstm_all_theta_07 = []
lstm_all_psi_07 = []

lstm_all_phi_08 = []
lstm_all_theta_08 = []
lstm_all_psi_08 = []

lstm_all_phi_09 = []
lstm_all_theta_09 = []
lstm_all_psi_09 = []

lstm_all_phi_10 = []
lstm_all_theta_10 = []
lstm_all_psi_10 = []

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

gru_all_phi_01 = []
gru_all_theta_01 = []
gru_all_psi_01 = []

gru_all_phi_02 = []
gru_all_theta_02 = []
gru_all_psi_02 = []

gru_all_phi_03 = []
gru_all_theta_03 = []
gru_all_psi_03 = []

gru_all_phi_04 = []
gru_all_theta_04 = []
gru_all_psi_04 = []

gru_all_phi_05 = []
gru_all_theta_05 = []
gru_all_psi_05 = []

gru_all_phi_06 = []
gru_all_theta_06 = []
gru_all_psi_06 = []

gru_all_phi_07 = []
gru_all_theta_07 = []
gru_all_psi_07 = []

gru_all_phi_08 = []
gru_all_theta_08 = []
gru_all_psi_08 = []

gru_all_phi_09 = []
gru_all_theta_09 = []
gru_all_psi_09 = []

gru_all_phi_10 = []
gru_all_theta_10 = []
gru_all_psi_10 = []

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

loss_train_list_lstm_all = []
loss_train_list_lstm_phi = []
loss_train_list_lstm_theta = []
loss_train_list_lstm_psi = []
loss_train_list_gru_all = []
loss_train_list_gru_phi = []
loss_train_list_gru_theta = []
loss_train_list_gru_psi = []
loss_test_list_lstm_all = []
loss_test_list_lstm_phi = []
loss_test_list_lstm_theta = []
loss_test_list_lstm_psi = []
loss_test_list_gru_all = []
loss_test_list_gru_phi = []
loss_test_list_gru_theta = []
loss_test_list_gru_psi = []

loss_test_list = []
iteration_list = []
accuracy_list = []
epoch_counter = 1
#results_dataframe = pd.DataFrame()
label_notwritten = True

loss_training_total = 0

for epoch in range(num_epochs):
	total_train_lstm_all = 0
	total_train_lstm_phi = 0
	total_train_lstm_theta = 0
	total_train_lstm_psi = 0
	total_train_gru_all = 0
	total_train_gru_phi = 0
	total_train_gru_theta = 0
	total_train_gru_psi = 0
	iter_train = 0
	for i, (X, Y) in enumerate(train_loader_shuffled):

		if is_cuda:
			X = X.cuda()
			Y = Y.cuda()

		# Clear gradients w.r.t. parameters
		optimizer_lstm_all.zero_grad()
		optimizer_lstm_phi.zero_grad()
		optimizer_lstm_theta.zero_grad()
		optimizer_lstm_psi.zero_grad()
		optimizer_gru_all.zero_grad()
		optimizer_gru_phi.zero_grad()
		optimizer_gru_theta.zero_grad()
		optimizer_gru_psi.zero_grad()

		# Forward pass to get output/logits
		outputs_lstm_all = model_instance_lstm_all(X)
		outputs_lstm_phi = model_instance_lstm_phi(X)
		outputs_lstm_theta = model_instance_lstm_theta(X)
		outputs_lstm_psi = model_instance_lstm_psi(X)
		outputs_gru_all = model_instance_gru_all(X)
		outputs_gru_phi = model_instance_gru_phi(X)
		outputs_gru_theta = model_instance_gru_theta(X)
		outputs_gru_psi = model_instance_gru_psi(X)

		# print("Y[:,0] shape is ", torch.index_select(Y,1,torch.tensor([0]).cuda()).shape)
		# print("Y[:,0] type is ", type(Y[:,0]))

		# print("outputs_lstm_phi shape is ", outputs_lstm_phi.shape)
		# print("outputs_lstm_phi type is ", type(outputs_lstm_phi))

		# Calculate MSE Loss for LSTM
		loss_lstm_all = error_lstm_all(outputs_lstm_all,Y)
		loss_lstm_phi = error_lstm_phi(outputs_lstm_phi,torch.index_select(Y,1,torch.tensor([0]).cuda()))
		loss_lstm_theta = error_lstm_theta(outputs_lstm_theta,torch.index_select(Y,1,torch.tensor([1]).cuda()))
		loss_lstm_psi = error_lstm_psi(outputs_lstm_psi,torch.index_select(Y,1,torch.tensor([2]).cuda()))
		
		total_train_lstm_all += loss_lstm_all.data.item()
		total_train_lstm_phi += loss_lstm_phi.data.item()
		total_train_lstm_theta += loss_lstm_theta.data.item()
		total_train_lstm_psi += loss_lstm_psi.data.item()

		# Calculate MSE Loss for GRU
		loss_gru_all = error_lstm_all(outputs_gru_all,Y)
		loss_gru_phi = error_lstm_phi(outputs_gru_phi,torch.index_select(Y,1,torch.tensor([0]).cuda()))
		loss_gru_theta = error_lstm_theta(outputs_gru_theta,torch.index_select(Y,1,torch.tensor([1]).cuda()))
		loss_gru_psi = error_lstm_psi(outputs_gru_psi,torch.index_select(Y,1,torch.tensor([2]).cuda()))
		
		total_train_gru_all += loss_gru_all.data.item()
		total_train_gru_phi += loss_gru_phi.data.item()
		total_train_gru_theta += loss_gru_theta.data.item()
		total_train_gru_psi += loss_gru_psi.data.item()

		iter_train += 1

		# Getting gradients for LSTM entities
		loss_lstm_all.backward()
		loss_lstm_phi.backward()
		loss_lstm_theta.backward()
		loss_lstm_psi.backward()
		# Getting gradients for GRU entities
		loss_gru_all.backward()
		loss_gru_phi.backward()
		loss_gru_theta.backward()
		loss_gru_psi.backward()

		# Updating parameters
		optimizer_lstm_all.step()
		optimizer_lstm_phi.step()
		optimizer_lstm_theta.step()
		optimizer_lstm_psi.step()
		optimizer_gru_all.step()
		optimizer_gru_phi.step()
		optimizer_gru_theta.step()
		optimizer_gru_psi.step()
		if (iter_train%5000==0):
			print('iter train = ',iter_train)

	total_train_lstm_all = total_train_lstm_all/iter_train
	total_train_lstm_phi = total_train_lstm_phi/iter_train
	total_train_lstm_theta = total_train_lstm_theta/iter_train
	total_train_lstm_psi = total_train_lstm_psi/iter_train

	total_train_gru_all = total_train_gru_all/iter_train
	total_train_gru_phi = total_train_gru_phi/iter_train
	total_train_gru_theta = total_train_gru_theta/iter_train
	total_train_gru_psi = total_train_gru_psi/iter_train

	loss_train_list_lstm_all.append(total_train_lstm_all)
	loss_train_list_lstm_phi.append(total_train_lstm_phi)
	loss_train_list_lstm_theta.append(total_train_lstm_theta)
	loss_train_list_lstm_psi.append(total_train_lstm_psi)

	loss_train_list_gru_all.append(total_train_gru_all)
	loss_train_list_gru_phi.append(total_train_gru_phi)
	loss_train_list_gru_theta.append(total_train_gru_theta)
	loss_train_list_gru_psi.append(total_train_gru_psi)

	if epoch_counter % 1 == 0:
		# Calculate Accuracy         
		correct = 0
		total_test_lstm_all = 0
		total_test_lstm_phi = 0
		total_test_lstm_theta = 0
		total_test_lstm_psi = 0
		total_test_gru_all = 0
		total_test_gru_phi = 0
		total_test_gru_theta = 0
		total_test_gru_psi = 0
		outputs_test_vect = []
		y_test_vect = []
		iter_test = 0
		t_vect = []

		for X_test, Y_test in test_loader:
			
			if is_cuda:
				X_test = X_test.cuda()
				Y_test = Y_test.cuda()

			# Forward pass only to get logits/output
			outputs_test_lstm_all = model_instance_lstm_all(X_test)
			outputs_test_lstm_phi = model_instance_lstm_phi(X_test)
			outputs_test_lstm_theta = model_instance_lstm_theta(X_test)
			outputs_test_lstm_psi = model_instance_lstm_psi(X_test)
			outputs_test_gru_all = model_instance_gru_all(X_test)
			outputs_test_gru_phi = model_instance_gru_phi(X_test)
			outputs_test_gru_theta = model_instance_gru_theta(X_test)
			outputs_test_gru_psi = model_instance_gru_psi(X_test)
			

			if (label_notwritten):
				Y_test_cpu = Y_test.cpu().numpy()
				ref_phi.append(Y_test_cpu[0][0])
				ref_theta.append(Y_test_cpu[0][1])
				ref_psi.append(Y_test_cpu[0][2])

			if ((epoch_counter == 1)):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_01.append(outputs_test_cpu[0][0])
				lstm_all_theta_01.append(outputs_test_cpu[0][1])
				lstm_all_psi_01.append(outputs_test_cpu[0][2])
				outputs_test_cpu_01.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_01.append(outputs_test_cpu[0][0])
				gru_all_theta_01.append(outputs_test_cpu[0][1])
				gru_all_psi_01.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_01.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_01.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_01.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_01.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_01.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_01.append(outputs_test_cpu[0][0])

			if (epoch_counter == 2):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_02.append(outputs_test_cpu[0][0])
				lstm_all_theta_02.append(outputs_test_cpu[0][1])
				lstm_all_psi_02.append(outputs_test_cpu[0][2])
				outputs_test_cpu_02.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_02.append(outputs_test_cpu[0][0])
				gru_all_theta_02.append(outputs_test_cpu[0][1])
				gru_all_psi_02.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_02.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_02.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_02.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_02.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_02.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_02.append(outputs_test_cpu[0][0])

			if (epoch_counter == 3):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_03.append(outputs_test_cpu[0][0])
				lstm_all_theta_03.append(outputs_test_cpu[0][1])
				lstm_all_psi_03.append(outputs_test_cpu[0][2])
				outputs_test_cpu_03.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_03.append(outputs_test_cpu[0][0])
				gru_all_theta_03.append(outputs_test_cpu[0][1])
				gru_all_psi_03.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_03.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_03.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_03.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_03.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_03.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_03.append(outputs_test_cpu[0][0])

			if (epoch_counter == 4):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_04.append(outputs_test_cpu[0][0])
				lstm_all_theta_04.append(outputs_test_cpu[0][1])
				lstm_all_psi_04.append(outputs_test_cpu[0][2])
				outputs_test_cpu_04.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_04.append(outputs_test_cpu[0][0])
				gru_all_theta_04.append(outputs_test_cpu[0][1])
				gru_all_psi_04.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_04.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_04.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_04.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_04.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_04.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_04.append(outputs_test_cpu[0][0])
				

			if (epoch_counter == 5):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_05.append(outputs_test_cpu[0][0])
				lstm_all_theta_05.append(outputs_test_cpu[0][1])
				lstm_all_psi_05.append(outputs_test_cpu[0][2])
				outputs_test_cpu_05.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_05.append(outputs_test_cpu[0][0])
				gru_all_theta_05.append(outputs_test_cpu[0][1])
				gru_all_psi_05.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_05.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_05.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_05.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_05.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_05.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_05.append(outputs_test_cpu[0][0])

			if (epoch_counter == 6):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_06.append(outputs_test_cpu[0][0])
				lstm_all_theta_06.append(outputs_test_cpu[0][1])
				lstm_all_psi_06.append(outputs_test_cpu[0][2])
				outputs_test_cpu_06.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_06.append(outputs_test_cpu[0][0])
				gru_all_theta_06.append(outputs_test_cpu[0][1])
				gru_all_psi_06.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_06.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_06.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_06.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_06.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_06.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_06.append(outputs_test_cpu[0][0])

			if (epoch_counter == 7):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_07.append(outputs_test_cpu[0][0])
				lstm_all_theta_07.append(outputs_test_cpu[0][1])
				lstm_all_psi_07.append(outputs_test_cpu[0][2])
				outputs_test_cpu_07.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_07.append(outputs_test_cpu[0][0])
				gru_all_theta_07.append(outputs_test_cpu[0][1])
				gru_all_psi_07.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_07.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_07.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_07.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_07.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_07.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_07.append(outputs_test_cpu[0][0])

			if (epoch_counter == 8):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_08.append(outputs_test_cpu[0][0])
				lstm_all_theta_08.append(outputs_test_cpu[0][1])
				lstm_all_psi_08.append(outputs_test_cpu[0][2])
				outputs_test_cpu_08.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_08.append(outputs_test_cpu[0][0])
				gru_all_theta_08.append(outputs_test_cpu[0][1])
				gru_all_psi_08.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_08.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_08.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_08.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_08.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_08.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_08.append(outputs_test_cpu[0][0])

			if (epoch_counter == 9):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_09.append(outputs_test_cpu[0][0])
				lstm_all_theta_09.append(outputs_test_cpu[0][1])
				lstm_all_psi_09.append(outputs_test_cpu[0][2])
				outputs_test_cpu_09.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_09.append(outputs_test_cpu[0][0])
				gru_all_theta_09.append(outputs_test_cpu[0][1])
				gru_all_psi_09.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_09.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_09.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_09.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_09.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_09.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_09.append(outputs_test_cpu[0][0])

			if (epoch_counter == 10):
				# ALL
				outputs_test_cpu = outputs_test_lstm_all.data.cpu().numpy()
				lstm_all_phi_10.append(outputs_test_cpu[0][0])
				lstm_all_theta_10.append(outputs_test_cpu[0][1])
				lstm_all_psi_10.append(outputs_test_cpu[0][2])
				outputs_test_cpu_10.append(outputs_test_cpu[0])
				outputs_test_cpu = outputs_test_gru_all.data.cpu().numpy()
				gru_all_phi_10.append(outputs_test_cpu[0][0])
				gru_all_theta_10.append(outputs_test_cpu[0][1])
				gru_all_psi_10.append(outputs_test_cpu[0][2])
				# DEDICATED
				outputs_test_cpu = outputs_test_lstm_phi.data.cpu().numpy()
				lstm_phi_10.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_theta.data.cpu().numpy()
				lstm_theta_10.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_lstm_psi.data.cpu().numpy()
				lstm_psi_10.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_phi.data.cpu().numpy()
				gru_phi_10.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_theta.data.cpu().numpy()
				gru_theta_10.append(outputs_test_cpu[0][0])
				outputs_test_cpu = outputs_test_gru_psi.data.cpu().numpy()
				gru_psi_10.append(outputs_test_cpu[0][0])


			# Calculate MSE Loss for LSTM
			loss_test_lstm_all = error_lstm_all(outputs_lstm_all.data,Y_test)
			loss_test_lstm_phi = error_lstm_phi(outputs_lstm_phi.data,torch.index_select(Y_test,1,torch.tensor([0]).cuda()))
			loss_test_lstm_theta = error_lstm_theta(outputs_lstm_theta.data,torch.index_select(Y_test,1,torch.tensor([1]).cuda()))
			loss_test_lstm_psi = error_lstm_psi(outputs_lstm_psi.data,torch.index_select(Y_test,1,torch.tensor([2]).cuda()))
			
			total_test_lstm_all += loss_test_lstm_all.data.item()
			total_test_lstm_phi += loss_test_lstm_phi.data.item()
			total_test_lstm_theta += loss_test_lstm_theta.data.item()
			total_test_lstm_psi += loss_test_lstm_psi.data.item()

			# Calculate MSE Loss for GRU
			loss_test_gru_all = error_lstm_all(outputs_gru_all.data,Y_test)
			loss_test_gru_phi = error_lstm_phi(outputs_gru_phi.data,torch.index_select(Y_test,1,torch.tensor([0]).cuda()))
			loss_test_gru_theta = error_lstm_theta(outputs_gru_theta.data,torch.index_select(Y_test,1,torch.tensor([1]).cuda()))
			loss_test_gru_psi = error_lstm_psi(outputs_gru_psi.data,torch.index_select(Y_test,1,torch.tensor([2]).cuda()))
			
			total_test_gru_all += loss_test_gru_all.data.item()
			total_test_gru_phi += loss_test_gru_phi.data.item()
			total_test_gru_theta += loss_test_gru_theta.data.item()
			total_test_gru_psi += loss_test_gru_psi.data.item()

			

			t_vect.append(iter_test)
			iter_test += 1

			# Total correct predictions

		if (label_notwritten):
			label_notwritten = False

		total_test_lstm_all = total_test_lstm_all/iter_test
		total_test_lstm_phi = total_test_lstm_phi/iter_test
		total_test_lstm_theta = total_test_lstm_theta/iter_test
		total_test_lstm_psi = total_test_lstm_psi/iter_test

		total_test_gru_all = total_test_gru_all/iter_test
		total_test_gru_phi = total_test_gru_phi/iter_test
		total_test_gru_theta = total_test_gru_theta/iter_test
		total_test_gru_psi = total_test_gru_psi/iter_test

		loss_test_list_lstm_all.append(total_test_lstm_all)
		loss_test_list_lstm_phi.append(total_test_lstm_phi)
		loss_test_list_lstm_theta.append(total_test_lstm_theta)
		loss_test_list_lstm_psi.append(total_test_lstm_psi)

		loss_test_list_gru_all.append(total_test_gru_all)
		loss_test_list_gru_phi.append(total_test_gru_phi)
		loss_test_list_gru_theta.append(total_test_gru_theta)
		loss_test_list_gru_psi.append(total_test_gru_psi)

		#loss_list.append(loss.data.item())
		#iteration_list.append(epoch_counter)
		#accuracy_list.append(loss_test)
		
		# Print Loss
		print('-----')
		print('Epoch: {}. LSTM ALL Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_lstm_all, total_test_lstm_all))
		print('Epoch: {}. LSTM PHI Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_lstm_phi, total_test_lstm_phi))
		print('Epoch: {}. LSTM THETA Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_lstm_theta, total_test_lstm_theta))
		print('Epoch: {}. LSTM PSI Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_lstm_psi, total_test_lstm_psi))
		print('Epoch: {}. GRU  ALL Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_gru_all, total_test_gru_all))
		print('Epoch: {}. GRU  PHI Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_gru_phi, total_test_gru_phi))
		print('Epoch: {}. GRU  THETA Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_gru_theta, total_test_gru_theta))
		print('Epoch: {}. GRU  PSI Training Loss: {}. Testing Loss: {}'.format(epoch_counter, total_train_gru_psi, total_test_gru_psi))
	epoch_counter += 1

outputs_test_cpu_1_filtered, outputs_test_cpu_2_filtered = low_pass_filter(np.array(outputs_test_cpu_01), np.array(outputs_test_cpu_02), 0.2)
outputs_test_cpu_3_filtered, outputs_test_cpu_4_filtered = low_pass_filter(np.array(outputs_test_cpu_03), np.array(outputs_test_cpu_04), 0.2)


# write results in a csv file
file = open("D3_results_lstm_gru.csv", mode="w")
file.write("ref_phi,ref_theta,ref_psi,")
file.write("lstm_all_phi_all_epoch_01,lstm_all_theta_epoch_01,lstm_all_psi_epoch_01,")
file.write("lstm_all_phi_epoch_02,lstm_all_theta_epoch_02,lstm_all_psi_epoch_02,")
file.write("lstm_all_phi_epoch_03,lstm_all_theta_epoch_03,lstm_all_psi_epoch_03,")
file.write("lstm_all_phi_epoch_04,lstm_all_theta_epoch_04,lstm_all_psi_epoch_04,")
file.write("lstm_all_phi_epoch_05,lstm_all_theta_epoch_05,lstm_all_psi_epoch_05,")
file.write("lstm_all_phi_epoch_06,lstm_all_theta_epoch_06,lstm_all_psi_epoch_06,")
file.write("lstm_all_phi_epoch_07,lstm_all_theta_epoch_07,lstm_all_psi_epoch_07,")
file.write("lstm_all_phi_epoch_08,lstm_all_theta_epoch_08,lstm_all_psi_epoch_08,")
file.write("lstm_all_phi_epoch_09,lstm_all_theta_epoch_09,lstm_all_psi_epoch_09,")
file.write("lstm_all_phi_epoch_10,lstm_all_theta_epoch_10,lstm_all_psi_epoch_10,")
file.write("gru_all_phi_epoch_01,gru_all_theta_epoch_01,gru_all_psi_epoch_01,")
file.write("gru_all_phi_epoch_02,gru_all_theta_epoch_02,gru_all_psi_epoch_02,")
file.write("gru_all_phi_epoch_03,gru_all_theta_epoch_03,gru_all_psi_epoch_03,")
file.write("gru_all_phi_epoch_04,gru_all_theta_epoch_04,gru_all_psi_epoch_04,")
file.write("gru_all_phi_epoch_05,gru_all_theta_epoch_05,gru_all_psi_epoch_05,")
file.write("gru_all_phi_epoch_06,gru_all_theta_epoch_06,gru_all_psi_epoch_06,")
file.write("gru_all_phi_epoch_07,gru_all_theta_epoch_07,gru_all_psi_epoch_07,")
file.write("gru_all_phi_epoch_08,gru_all_theta_epoch_08,gru_all_psi_epoch_08,")
file.write("gru_all_phi_epoch_09,gru_all_theta_epoch_09,gru_all_psi_epoch_09,")
file.write("gru_all_phi_epoch_10,gru_all_theta_epoch_10,gru_all_psi_epoch_10,")
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
	file.write(str(lstm_all_phi_01[i]) + "," + str(lstm_all_theta_01[i]) + "," + str(lstm_all_psi_01[i]) + ",")
	file.write(str(lstm_all_phi_02[i]) + "," + str(lstm_all_theta_02[i]) + "," + str(lstm_all_psi_02[i]) + ",")
	file.write(str(lstm_all_phi_03[i]) + "," + str(lstm_all_theta_03[i]) + "," + str(lstm_all_psi_03[i]) + ",")
	file.write(str(lstm_all_phi_04[i]) + "," + str(lstm_all_theta_04[i]) + "," + str(lstm_all_psi_04[i]) + ",")
	file.write(str(lstm_all_phi_05[i]) + "," + str(lstm_all_theta_05[i]) + "," + str(lstm_all_psi_05[i]) + ",")
	file.write(str(lstm_all_phi_06[i]) + "," + str(lstm_all_theta_06[i]) + "," + str(lstm_all_psi_06[i]) + ",")
	file.write(str(lstm_all_phi_07[i]) + "," + str(lstm_all_theta_07[i]) + "," + str(lstm_all_psi_07[i]) + ",")
	file.write(str(lstm_all_phi_08[i]) + "," + str(lstm_all_theta_08[i]) + "," + str(lstm_all_psi_08[i]) + ",")
	file.write(str(lstm_all_phi_09[i]) + "," + str(lstm_all_theta_09[i]) + "," + str(lstm_all_psi_09[i]) + ",")
	file.write(str(lstm_all_phi_10[i]) + "," + str(lstm_all_theta_10[i]) + "," + str(lstm_psi_10[i]) + ",")
	file.write(str(gru_all_phi_01[i]) + "," + str(gru_all_theta_01[i]) + "," + str(gru_all_psi_01[i]) + ",")
	file.write(str(gru_all_phi_02[i]) + "," + str(gru_all_theta_02[i]) + "," + str(gru_all_psi_02[i]) + ",")
	file.write(str(gru_all_phi_03[i]) + "," + str(gru_all_theta_03[i]) + "," + str(gru_all_psi_03[i]) + ",")
	file.write(str(gru_all_phi_04[i]) + "," + str(gru_all_theta_04[i]) + "," + str(gru_all_psi_04[i]) + ",")
	file.write(str(gru_all_phi_05[i]) + "," + str(gru_all_theta_05[i]) + "," + str(gru_all_psi_05[i]) + ",")
	file.write(str(gru_all_phi_06[i]) + "," + str(gru_all_theta_06[i]) + "," + str(gru_all_psi_06[i]) + ",")
	file.write(str(gru_all_phi_07[i]) + "," + str(gru_all_theta_07[i]) + "," + str(gru_all_psi_07[i]) + ",")
	file.write(str(gru_all_phi_08[i]) + "," + str(gru_all_theta_08[i]) + "," + str(gru_all_psi_08[i]) + ",")
	file.write(str(gru_all_phi_09[i]) + "," + str(gru_all_theta_09[i]) + "," + str(gru_all_psi_09[i]) + ",")
	file.write(str(gru_all_phi_10[i]) + "," + str(gru_all_theta_10[i]) + "," + str(gru_all_psi_10[i]) + ",")	
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

file = open("D3_loss_report.csv", mode="w")
file.write("epoch_id,")
file.write("lstm_all_train,lstm_phi_train,lstm_theta_train,lstm_psi_train,")
file.write("lstm_all_test,lstm_phi_test,lstm_theta_test,lstm_psi_test,")
file.write("gru_all_train,gru_phi_train,gru_theta_train,gru_psi_train,")
file.write("gru_all_test,gru_phi_test,gru_theta_test,gru_psi_test\n")
for i in range(num_epochs):
	file.write(str(i) + "," )
	file.write(str(loss_train_list_lstm_all[i]) + "," + str(loss_train_list_lstm_phi[i]) + "," + str(loss_train_list_lstm_theta[i]) + "," + str(loss_train_list_lstm_psi[i]) + ",")
	file.write(str(loss_test_list_lstm_all[i]) + "," + str(loss_test_list_lstm_phi[i]) + "," + str(loss_test_list_lstm_theta[i]) + "," + str(loss_test_list_lstm_psi[i]) + ",")
	file.write(str(loss_train_list_gru_all[i]) + "," + str(loss_train_list_gru_phi[i]) + "," + str(loss_train_list_gru_theta[i]) + "," + str(loss_train_list_gru_psi[i]) + ",")
	file.write(str(loss_test_list_gru_all[i]) + "," + str(loss_test_list_gru_phi[i]) + "," + str(loss_test_list_gru_theta[i]) + "," + str(loss_test_list_gru_psi[i]) + "\n")
file.close()

# get some values on errors etc...
error_lstm_all_phi_01 = np.array(lstm_all_phi_01) - np.array(ref_phi)
error_lstm_all_phi_02 = np.array(lstm_all_phi_02) - np.array(ref_phi)
error_lstm_all_phi_03 = np.array(lstm_all_phi_03) - np.array(ref_phi)
error_lstm_all_phi_04 = np.array(lstm_all_phi_04) - np.array(ref_phi)
error_lstm_all_phi_05 = np.array(lstm_all_phi_05) - np.array(ref_phi)
error_lstm_all_phi_06 = np.array(lstm_all_phi_06) - np.array(ref_phi)
error_lstm_all_phi_07 = np.array(lstm_all_phi_07) - np.array(ref_phi)
error_lstm_all_phi_08 = np.array(lstm_all_phi_08) - np.array(ref_phi)
error_lstm_all_phi_09 = np.array(lstm_all_phi_09) - np.array(ref_phi)
error_lstm_all_phi_10 = np.array(lstm_all_phi_10) - np.array(ref_phi)

error_lstm_all_theta_01 = np.array(lstm_all_theta_01) - np.array(ref_theta)
error_lstm_all_theta_02 = np.array(lstm_all_theta_02) - np.array(ref_theta)
error_lstm_all_theta_03 = np.array(lstm_all_theta_03) - np.array(ref_theta)
error_lstm_all_theta_04 = np.array(lstm_all_theta_04) - np.array(ref_theta)
error_lstm_all_theta_05 = np.array(lstm_all_theta_05) - np.array(ref_theta)
error_lstm_all_theta_06 = np.array(lstm_all_theta_06) - np.array(ref_theta)
error_lstm_all_theta_07 = np.array(lstm_all_theta_07) - np.array(ref_theta)
error_lstm_all_theta_08 = np.array(lstm_all_theta_08) - np.array(ref_theta)
error_lstm_all_theta_09 = np.array(lstm_all_theta_09) - np.array(ref_theta)
error_lstm_all_theta_10 = np.array(lstm_all_theta_10) - np.array(ref_theta)

error_lstm_all_psi_01 = np.array(lstm_all_psi_01) - np.array(ref_psi)
error_lstm_all_psi_02 = np.array(lstm_all_psi_02) - np.array(ref_psi)
error_lstm_all_psi_03 = np.array(lstm_all_psi_03) - np.array(ref_psi)
error_lstm_all_psi_04 = np.array(lstm_all_psi_04) - np.array(ref_psi)
error_lstm_all_psi_05 = np.array(lstm_all_psi_05) - np.array(ref_psi)
error_lstm_all_psi_06 = np.array(lstm_all_psi_06) - np.array(ref_psi)
error_lstm_all_psi_07 = np.array(lstm_all_psi_07) - np.array(ref_psi)
error_lstm_all_psi_08 = np.array(lstm_all_psi_08) - np.array(ref_psi)
error_lstm_all_psi_09 = np.array(lstm_all_psi_09) - np.array(ref_psi)
error_lstm_all_psi_10 = np.array(lstm_all_psi_10) - np.array(ref_psi)

error_gru_all_phi_01 = np.array(gru_all_phi_01) - np.array(ref_phi)
error_gru_all_phi_02 = np.array(gru_all_phi_02) - np.array(ref_phi)
error_gru_all_phi_03 = np.array(gru_all_phi_03) - np.array(ref_phi)
error_gru_all_phi_04 = np.array(gru_all_phi_04) - np.array(ref_phi)
error_gru_all_phi_05 = np.array(gru_all_phi_05) - np.array(ref_phi)
error_gru_all_phi_06 = np.array(gru_all_phi_06) - np.array(ref_phi)
error_gru_all_phi_07 = np.array(gru_all_phi_07) - np.array(ref_phi)
error_gru_all_phi_08 = np.array(gru_all_phi_08) - np.array(ref_phi)
error_gru_all_phi_09 = np.array(gru_all_phi_09) - np.array(ref_phi)
error_gru_all_phi_10 = np.array(gru_all_phi_10) - np.array(ref_phi)

error_gru_all_theta_01 = np.array(gru_all_theta_01) - np.array(ref_theta)
error_gru_all_theta_02 = np.array(gru_all_theta_02) - np.array(ref_theta)
error_gru_all_theta_03 = np.array(gru_all_theta_03) - np.array(ref_theta)
error_gru_all_theta_04 = np.array(gru_all_theta_04) - np.array(ref_theta)
error_gru_all_theta_05 = np.array(gru_all_theta_05) - np.array(ref_theta)
error_gru_all_theta_06 = np.array(gru_all_theta_06) - np.array(ref_theta)
error_gru_all_theta_07 = np.array(gru_all_theta_07) - np.array(ref_theta)
error_gru_all_theta_08 = np.array(gru_all_theta_08) - np.array(ref_theta)
error_gru_all_theta_09 = np.array(gru_all_theta_09) - np.array(ref_theta)
error_gru_all_theta_10 = np.array(gru_all_theta_10) - np.array(ref_theta)

error_gru_all_psi_01 = np.array(gru_all_psi_01) - np.array(ref_psi)
error_gru_all_psi_02 = np.array(gru_all_psi_02) - np.array(ref_psi)
error_gru_all_psi_03 = np.array(gru_all_psi_03) - np.array(ref_psi)
error_gru_all_psi_04 = np.array(gru_all_psi_04) - np.array(ref_psi)
error_gru_all_psi_05 = np.array(gru_all_psi_05) - np.array(ref_psi)
error_gru_all_psi_06 = np.array(gru_all_psi_06) - np.array(ref_psi)
error_gru_all_psi_07 = np.array(gru_all_psi_07) - np.array(ref_psi)
error_gru_all_psi_08 = np.array(gru_all_psi_08) - np.array(ref_psi)
error_gru_all_psi_09 = np.array(gru_all_psi_09) - np.array(ref_psi)
error_gru_all_psi_10 = np.array(gru_all_psi_10) - np.array(ref_psi)

error_lstm_phi_01 = np.array(lstm_phi_01) - np.array(ref_phi)
error_lstm_phi_02 = np.array(lstm_phi_02) - np.array(ref_phi)
error_lstm_phi_03 = np.array(lstm_phi_03) - np.array(ref_phi)
error_lstm_phi_04 = np.array(lstm_phi_04) - np.array(ref_phi)
error_lstm_phi_05 = np.array(lstm_phi_05) - np.array(ref_phi)
error_lstm_phi_06 = np.array(lstm_phi_06) - np.array(ref_phi)
error_lstm_phi_07 = np.array(lstm_phi_07) - np.array(ref_phi)
error_lstm_phi_08 = np.array(lstm_phi_08) - np.array(ref_phi)
error_lstm_phi_09 = np.array(lstm_phi_09) - np.array(ref_phi)
error_lstm_phi_10 = np.array(lstm_phi_10) - np.array(ref_phi)

error_lstm_theta_01 = np.array(lstm_theta_01) - np.array(ref_theta)
error_lstm_theta_02 = np.array(lstm_theta_02) - np.array(ref_theta)
error_lstm_theta_03 = np.array(lstm_theta_03) - np.array(ref_theta)
error_lstm_theta_04 = np.array(lstm_theta_04) - np.array(ref_theta)
error_lstm_theta_05 = np.array(lstm_theta_05) - np.array(ref_theta)
error_lstm_theta_06 = np.array(lstm_theta_06) - np.array(ref_theta)
error_lstm_theta_07 = np.array(lstm_theta_07) - np.array(ref_theta)
error_lstm_theta_08 = np.array(lstm_theta_08) - np.array(ref_theta)
error_lstm_theta_09 = np.array(lstm_theta_09) - np.array(ref_theta)
error_lstm_theta_10 = np.array(lstm_theta_10) - np.array(ref_theta)

error_lstm_psi_01 = np.array(lstm_psi_01) - np.array(ref_psi)
error_lstm_psi_02 = np.array(lstm_psi_02) - np.array(ref_psi)
error_lstm_psi_03 = np.array(lstm_psi_03) - np.array(ref_psi)
error_lstm_psi_04 = np.array(lstm_psi_04) - np.array(ref_psi)
error_lstm_psi_05 = np.array(lstm_psi_05) - np.array(ref_psi)
error_lstm_psi_06 = np.array(lstm_psi_06) - np.array(ref_psi)
error_lstm_psi_07 = np.array(lstm_psi_07) - np.array(ref_psi)
error_lstm_psi_08 = np.array(lstm_psi_08) - np.array(ref_psi)
error_lstm_psi_09 = np.array(lstm_psi_09) - np.array(ref_psi)
error_lstm_psi_10 = np.array(lstm_psi_10) - np.array(ref_psi)

error_gru_phi_01 = np.array(gru_phi_01) - np.array(ref_phi)
error_gru_phi_02 = np.array(gru_phi_02) - np.array(ref_phi)
error_gru_phi_03 = np.array(gru_phi_03) - np.array(ref_phi)
error_gru_phi_04 = np.array(gru_phi_04) - np.array(ref_phi)
error_gru_phi_05 = np.array(gru_phi_05) - np.array(ref_phi)
error_gru_phi_06 = np.array(gru_phi_06) - np.array(ref_phi)
error_gru_phi_07 = np.array(gru_phi_07) - np.array(ref_phi)
error_gru_phi_08 = np.array(gru_phi_08) - np.array(ref_phi)
error_gru_phi_09 = np.array(gru_phi_09) - np.array(ref_phi)
error_gru_phi_10 = np.array(gru_phi_10) - np.array(ref_phi)

error_gru_theta_01 = np.array(gru_theta_01) - np.array(ref_theta)
error_gru_theta_02 = np.array(gru_theta_02) - np.array(ref_theta)
error_gru_theta_03 = np.array(gru_theta_03) - np.array(ref_theta)
error_gru_theta_04 = np.array(gru_theta_04) - np.array(ref_theta)
error_gru_theta_05 = np.array(gru_theta_05) - np.array(ref_theta)
error_gru_theta_06 = np.array(gru_theta_06) - np.array(ref_theta)
error_gru_theta_07 = np.array(gru_theta_07) - np.array(ref_theta)
error_gru_theta_08 = np.array(gru_theta_08) - np.array(ref_theta)
error_gru_theta_09 = np.array(gru_theta_09) - np.array(ref_theta)
error_gru_theta_10 = np.array(gru_theta_10) - np.array(ref_theta)

error_gru_psi_01 = np.array(gru_psi_01) - np.array(ref_psi)
error_gru_psi_02 = np.array(gru_psi_02) - np.array(ref_psi)
error_gru_psi_03 = np.array(gru_psi_03) - np.array(ref_psi)
error_gru_psi_04 = np.array(gru_psi_04) - np.array(ref_psi)
error_gru_psi_05 = np.array(gru_psi_05) - np.array(ref_psi)
error_gru_psi_06 = np.array(gru_psi_06) - np.array(ref_psi)
error_gru_psi_07 = np.array(gru_psi_07) - np.array(ref_psi)
error_gru_psi_08 = np.array(gru_psi_08) - np.array(ref_psi)
error_gru_psi_09 = np.array(gru_psi_09) - np.array(ref_psi)
error_gru_psi_10 = np.array(gru_psi_10) - np.array(ref_psi)

# file = open("D1_errors_report.csv", mode="w")
# file.write("error_id,min,max,lstm_phi_1_lp,lstm_theta_1_lp,lstm_psi_1_lp,lstm_phi_2_lp,lstm_theta_2_lp,lstm_psi_2_lp,lstm_phi_3_lp,lstm_theta_3_lp,lstm_psi_3_lp\n")

print('-----------------------------')
print('Error reports LSTM PHI ALL')
print('error_lstm_all_phi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_01), np.max(error_lstm_all_phi_01), np.mean(error_lstm_all_phi_01), np.std(error_lstm_all_phi_01)))
print('error_lstm_all_phi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_02), np.max(error_lstm_all_phi_02), np.mean(error_lstm_all_phi_02), np.std(error_lstm_all_phi_02)))
print('error_lstm_all_phi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_03), np.max(error_lstm_all_phi_03), np.mean(error_lstm_all_phi_03), np.std(error_lstm_all_phi_03)))
print('error_lstm_all_phi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_04), np.max(error_lstm_all_phi_04), np.mean(error_lstm_all_phi_04), np.std(error_lstm_all_phi_04)))
print('error_lstm_all_phi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_05), np.max(error_lstm_all_phi_05), np.mean(error_lstm_all_phi_05), np.std(error_lstm_all_phi_05)))
print('error_lstm_all_phi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_06), np.max(error_lstm_all_phi_06), np.mean(error_lstm_all_phi_06), np.std(error_lstm_all_phi_06)))
print('error_lstm_all_phi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_07), np.max(error_lstm_all_phi_07), np.mean(error_lstm_all_phi_07), np.std(error_lstm_all_phi_07)))
print('error_lstm_all_phi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_08), np.max(error_lstm_all_phi_08), np.mean(error_lstm_all_phi_08), np.std(error_lstm_all_phi_08)))
print('error_lstm_all_phi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_09), np.max(error_lstm_all_phi_09), np.mean(error_lstm_all_phi_09), np.std(error_lstm_all_phi_09)))
print('error_lstm_all_phi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_phi_10), np.max(error_lstm_all_phi_10), np.mean(error_lstm_all_phi_10), np.std(error_lstm_all_phi_10)))
print('-----------------------------')
print('Error reports LSTM PHI DEDICATED')
print('error_lstm_phi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_01), np.max(error_lstm_phi_01), np.mean(error_lstm_phi_01), np.std(error_lstm_phi_01)))
print('error_lstm_phi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_02), np.max(error_lstm_phi_02), np.mean(error_lstm_phi_02), np.std(error_lstm_phi_02)))
print('error_lstm_phi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_03), np.max(error_lstm_phi_03), np.mean(error_lstm_phi_03), np.std(error_lstm_phi_03)))
print('error_lstm_phi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_04), np.max(error_lstm_phi_04), np.mean(error_lstm_phi_04), np.std(error_lstm_phi_04)))
print('error_lstm_phi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_05), np.max(error_lstm_phi_05), np.mean(error_lstm_phi_05), np.std(error_lstm_phi_05)))
print('error_lstm_phi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_06), np.max(error_lstm_phi_06), np.mean(error_lstm_phi_06), np.std(error_lstm_phi_06)))
print('error_lstm_phi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_07), np.max(error_lstm_phi_07), np.mean(error_lstm_phi_07), np.std(error_lstm_phi_07)))
print('error_lstm_phi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_08), np.max(error_lstm_phi_08), np.mean(error_lstm_phi_08), np.std(error_lstm_phi_08)))
print('error_lstm_phi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_09), np.max(error_lstm_phi_09), np.mean(error_lstm_phi_09), np.std(error_lstm_phi_09)))
print('error_lstm_phi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_phi_10), np.max(error_lstm_phi_10), np.mean(error_lstm_phi_10), np.std(error_lstm_all_phi_10)))
print('-----------------------------')
print('Error reports LSTM THETA ALL')
print('error_lstm_all_theta_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_01), np.max(error_lstm_all_theta_01), np.mean(error_lstm_all_theta_01), np.std(error_lstm_all_theta_01)))
print('error_lstm_all_theta_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_02), np.max(error_lstm_all_theta_02), np.mean(error_lstm_all_theta_02), np.std(error_lstm_all_theta_02)))
print('error_lstm_all_theta_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_03), np.max(error_lstm_all_theta_03), np.mean(error_lstm_all_theta_03), np.std(error_lstm_all_theta_03)))
print('error_lstm_all_theta_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_04), np.max(error_lstm_all_theta_04), np.mean(error_lstm_all_theta_04), np.std(error_lstm_all_theta_04)))
print('error_lstm_all_theta_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_05), np.max(error_lstm_all_theta_05), np.mean(error_lstm_all_theta_05), np.std(error_lstm_all_theta_05)))
print('error_lstm_all_theta_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_06), np.max(error_lstm_all_theta_06), np.mean(error_lstm_all_theta_06), np.std(error_lstm_all_theta_06)))
print('error_lstm_all_theta_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_07), np.max(error_lstm_all_theta_07), np.mean(error_lstm_all_theta_07), np.std(error_lstm_all_theta_07)))
print('error_lstm_all_theta_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_08), np.max(error_lstm_all_theta_08), np.mean(error_lstm_all_theta_08), np.std(error_lstm_all_theta_08)))
print('error_lstm_all_theta_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_09), np.max(error_lstm_all_theta_09), np.mean(error_lstm_all_theta_09), np.std(error_lstm_all_theta_09)))
print('error_lstm_all_theta_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_theta_10), np.max(error_lstm_all_theta_10), np.mean(error_lstm_all_theta_10), np.std(error_lstm_all_theta_10)))
print('-----------------------------')
print('Error reports LSTM THETA DEDICATED')
print('error_lstm_theta_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_01), np.max(error_lstm_theta_01), np.mean(error_lstm_theta_01), np.std(error_lstm_theta_01)))
print('error_lstm_theta_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_02), np.max(error_lstm_theta_02), np.mean(error_lstm_theta_02), np.std(error_lstm_theta_02)))
print('error_lstm_theta_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_03), np.max(error_lstm_theta_03), np.mean(error_lstm_theta_03), np.std(error_lstm_theta_03)))
print('error_lstm_theta_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_04), np.max(error_lstm_theta_04), np.mean(error_lstm_theta_04), np.std(error_lstm_theta_04)))
print('error_lstm_theta_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_05), np.max(error_lstm_theta_05), np.mean(error_lstm_theta_05), np.std(error_lstm_theta_05)))
print('error_lstm_theta_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_06), np.max(error_lstm_theta_06), np.mean(error_lstm_theta_06), np.std(error_lstm_theta_06)))
print('error_lstm_theta_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_07), np.max(error_lstm_theta_07), np.mean(error_lstm_theta_07), np.std(error_lstm_theta_07)))
print('error_lstm_theta_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_08), np.max(error_lstm_theta_08), np.mean(error_lstm_theta_08), np.std(error_lstm_theta_08)))
print('error_lstm_theta_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_09), np.max(error_lstm_theta_09), np.mean(error_lstm_theta_09), np.std(error_lstm_theta_09)))
print('error_lstm_theta_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_theta_10), np.max(error_lstm_theta_10), np.mean(error_lstm_theta_10), np.std(error_lstm_theta_10)))
print('-----------------------------')
print('Error reports LSTM PSI ALL')
print('error_lstm_all_psi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_01), np.max(error_lstm_all_psi_01), np.mean(error_lstm_all_psi_01), np.std(error_lstm_all_psi_01)))
print('error_lstm_all_psi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_02), np.max(error_lstm_all_psi_02), np.mean(error_lstm_all_psi_02), np.std(error_lstm_all_psi_02)))
print('error_lstm_all_psi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_03), np.max(error_lstm_all_psi_03), np.mean(error_lstm_all_psi_03), np.std(error_lstm_all_psi_03)))
print('error_lstm_all_psi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_04), np.max(error_lstm_all_psi_04), np.mean(error_lstm_all_psi_04), np.std(error_lstm_all_psi_04)))
print('error_lstm_all_psi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_05), np.max(error_lstm_all_psi_05), np.mean(error_lstm_all_psi_05), np.std(error_lstm_all_psi_05)))
print('error_lstm_all_psi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_06), np.max(error_lstm_all_psi_06), np.mean(error_lstm_all_psi_06), np.std(error_lstm_all_psi_06)))
print('error_lstm_all_psi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_07), np.max(error_lstm_all_psi_07), np.mean(error_lstm_all_psi_07), np.std(error_lstm_all_psi_07)))
print('error_lstm_all_psi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_08), np.max(error_lstm_all_psi_08), np.mean(error_lstm_all_psi_08), np.std(error_lstm_all_psi_08)))
print('error_lstm_all_psi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_09), np.max(error_lstm_all_psi_09), np.mean(error_lstm_all_psi_09), np.std(error_lstm_all_psi_09)))
print('error_lstm_all_psi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_all_psi_10), np.max(error_lstm_all_psi_10), np.mean(error_lstm_all_psi_10), np.std(error_lstm_all_psi_10)))
print('-----------------------------')
print('Error reports LSTM PSI DEDICATED')
print('error_lstm_psi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_01), np.max(error_lstm_psi_01), np.mean(error_lstm_psi_01), np.std(error_lstm_psi_01)))
print('error_lstm_psi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_02), np.max(error_lstm_psi_02), np.mean(error_lstm_psi_02), np.std(error_lstm_psi_02)))
print('error_lstm_psi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_03), np.max(error_lstm_psi_03), np.mean(error_lstm_psi_03), np.std(error_lstm_psi_03)))
print('error_lstm_psi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_04), np.max(error_lstm_psi_04), np.mean(error_lstm_psi_04), np.std(error_lstm_psi_04)))
print('error_lstm_psi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_05), np.max(error_lstm_psi_05), np.mean(error_lstm_psi_05), np.std(error_lstm_psi_05)))
print('error_lstm_psi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_06), np.max(error_lstm_psi_06), np.mean(error_lstm_psi_06), np.std(error_lstm_psi_06)))
print('error_lstm_psi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_07), np.max(error_lstm_psi_07), np.mean(error_lstm_psi_07), np.std(error_lstm_psi_07)))
print('error_lstm_psi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_08), np.max(error_lstm_psi_08), np.mean(error_lstm_psi_08), np.std(error_lstm_psi_08)))
print('error_lstm_psi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_09), np.max(error_lstm_psi_09), np.mean(error_lstm_psi_09), np.std(error_lstm_psi_09)))
print('error_lstm_psi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_lstm_psi_10), np.max(error_lstm_psi_10), np.mean(error_lstm_psi_10), np.std(error_lstm_theta_10)))

print('-----------------------------')
print('Error reports GRU PHI ALL')
print('error_gru_all_phi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_01), np.max(error_gru_all_phi_01), np.mean(error_gru_all_phi_01), np.std(error_gru_all_phi_01)))
print('error_gru_all_phi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_02), np.max(error_gru_all_phi_02), np.mean(error_gru_all_phi_02), np.std(error_gru_all_phi_02)))
print('error_gru_all_phi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_03), np.max(error_gru_all_phi_03), np.mean(error_gru_all_phi_03), np.std(error_gru_all_phi_03)))
print('error_gru_all_phi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_04), np.max(error_gru_all_phi_04), np.mean(error_gru_all_phi_04), np.std(error_gru_all_phi_04)))
print('error_gru_all_phi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_05), np.max(error_gru_all_phi_05), np.mean(error_gru_all_phi_05), np.std(error_gru_all_phi_05)))
print('error_gru_all_phi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_06), np.max(error_gru_all_phi_06), np.mean(error_gru_all_phi_06), np.std(error_gru_all_phi_06)))
print('error_gru_all_phi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_07), np.max(error_gru_all_phi_07), np.mean(error_gru_all_phi_07), np.std(error_gru_all_phi_07)))
print('error_gru_all_phi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_08), np.max(error_gru_all_phi_08), np.mean(error_gru_all_phi_08), np.std(error_gru_all_phi_08)))
print('error_gru_all_phi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_09), np.max(error_gru_all_phi_09), np.mean(error_gru_all_phi_09), np.std(error_gru_all_phi_09)))
print('error_gru_all_phi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_phi_10), np.max(error_gru_all_phi_10), np.mean(error_gru_all_phi_10), np.std(error_gru_all_phi_10)))
print('-----------------------------')
print('Error reports GRU PHI DEDICATED')
print('error_gru_phi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_01), np.max(error_gru_phi_01), np.mean(error_gru_phi_01), np.std(error_gru_phi_01)))
print('error_gru_phi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_02), np.max(error_gru_phi_02), np.mean(error_gru_phi_02), np.std(error_gru_phi_02)))
print('error_gru_phi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_03), np.max(error_gru_phi_03), np.mean(error_gru_phi_03), np.std(error_gru_phi_03)))
print('error_gru_phi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_04), np.max(error_gru_phi_04), np.mean(error_gru_phi_04), np.std(error_gru_phi_04)))
print('error_gru_phi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_05), np.max(error_gru_phi_05), np.mean(error_gru_phi_05), np.std(error_gru_phi_05)))
print('error_gru_phi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_06), np.max(error_gru_phi_06), np.mean(error_gru_phi_06), np.std(error_gru_phi_06)))
print('error_gru_phi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_07), np.max(error_gru_phi_07), np.mean(error_gru_phi_07), np.std(error_gru_phi_07)))
print('error_gru_phi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_08), np.max(error_gru_phi_08), np.mean(error_gru_phi_08), np.std(error_gru_phi_08)))
print('error_gru_phi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_09), np.max(error_gru_phi_09), np.mean(error_gru_phi_09), np.std(error_gru_phi_09)))
print('error_gru_phi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_phi_10), np.max(error_gru_phi_10), np.mean(error_gru_phi_10), np.std(error_gru_all_phi_10)))
print('-----------------------------')
print('Error reports GRU THETA ALL')
print('error_gru_all_theta_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_01), np.max(error_gru_all_theta_01), np.mean(error_gru_all_theta_01), np.std(error_gru_all_theta_01)))
print('error_gru_all_theta_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_02), np.max(error_gru_all_theta_02), np.mean(error_gru_all_theta_02), np.std(error_gru_all_theta_02)))
print('error_gru_all_theta_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_03), np.max(error_gru_all_theta_03), np.mean(error_gru_all_theta_03), np.std(error_gru_all_theta_03)))
print('error_gru_all_theta_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_04), np.max(error_gru_all_theta_04), np.mean(error_gru_all_theta_04), np.std(error_gru_all_theta_04)))
print('error_gru_all_theta_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_05), np.max(error_gru_all_theta_05), np.mean(error_gru_all_theta_05), np.std(error_gru_all_theta_05)))
print('error_gru_all_theta_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_06), np.max(error_gru_all_theta_06), np.mean(error_gru_all_theta_06), np.std(error_gru_all_theta_06)))
print('error_gru_all_theta_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_07), np.max(error_gru_all_theta_07), np.mean(error_gru_all_theta_07), np.std(error_gru_all_theta_07)))
print('error_gru_all_theta_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_08), np.max(error_gru_all_theta_08), np.mean(error_gru_all_theta_08), np.std(error_gru_all_theta_08)))
print('error_gru_all_theta_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_09), np.max(error_gru_all_theta_09), np.mean(error_gru_all_theta_09), np.std(error_gru_all_theta_09)))
print('error_gru_all_theta_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_theta_10), np.max(error_gru_all_theta_10), np.mean(error_gru_all_theta_10), np.std(error_gru_all_theta_10)))
print('-----------------------------')
print('Error reports GRU THETA DEDICATED')
print('error_gru_theta_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_01), np.max(error_gru_theta_01), np.mean(error_gru_theta_01), np.std(error_gru_theta_01)))
print('error_gru_theta_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_02), np.max(error_gru_theta_02), np.mean(error_gru_theta_02), np.std(error_gru_theta_02)))
print('error_gru_theta_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_03), np.max(error_gru_theta_03), np.mean(error_gru_theta_03), np.std(error_gru_theta_03)))
print('error_gru_theta_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_04), np.max(error_gru_theta_04), np.mean(error_gru_theta_04), np.std(error_gru_theta_04)))
print('error_gru_theta_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_05), np.max(error_gru_theta_05), np.mean(error_gru_theta_05), np.std(error_gru_theta_05)))
print('error_gru_theta_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_06), np.max(error_gru_theta_06), np.mean(error_gru_theta_06), np.std(error_gru_theta_06)))
print('error_gru_theta_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_07), np.max(error_gru_theta_07), np.mean(error_gru_theta_07), np.std(error_gru_theta_07)))
print('error_gru_theta_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_08), np.max(error_gru_theta_08), np.mean(error_gru_theta_08), np.std(error_gru_theta_08)))
print('error_gru_theta_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_09), np.max(error_gru_theta_09), np.mean(error_gru_theta_09), np.std(error_gru_theta_09)))
print('error_gru_theta_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_theta_10), np.max(error_gru_theta_10), np.mean(error_gru_theta_10), np.std(error_gru_theta_10)))
print('-----------------------------')
print('Error reports GRU PSI ALL')
print('error_gru_all_psi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_01), np.max(error_gru_all_psi_01), np.mean(error_gru_all_psi_01), np.std(error_gru_all_psi_01)))
print('error_gru_all_psi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_02), np.max(error_gru_all_psi_02), np.mean(error_gru_all_psi_02), np.std(error_gru_all_psi_02)))
print('error_gru_all_psi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_03), np.max(error_gru_all_psi_03), np.mean(error_gru_all_psi_03), np.std(error_gru_all_psi_03)))
print('error_gru_all_psi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_04), np.max(error_gru_all_psi_04), np.mean(error_gru_all_psi_04), np.std(error_gru_all_psi_04)))
print('error_gru_all_psi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_05), np.max(error_gru_all_psi_05), np.mean(error_gru_all_psi_05), np.std(error_gru_all_psi_05)))
print('error_gru_all_psi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_06), np.max(error_gru_all_psi_06), np.mean(error_gru_all_psi_06), np.std(error_gru_all_psi_06)))
print('error_gru_all_psi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_07), np.max(error_gru_all_psi_07), np.mean(error_gru_all_psi_07), np.std(error_gru_all_psi_07)))
print('error_gru_all_psi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_08), np.max(error_gru_all_psi_08), np.mean(error_gru_all_psi_08), np.std(error_gru_all_psi_08)))
print('error_gru_all_psi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_09), np.max(error_gru_all_psi_09), np.mean(error_gru_all_psi_09), np.std(error_gru_all_psi_09)))
print('error_gru_all_psi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_all_psi_10), np.max(error_gru_all_psi_10), np.mean(error_gru_all_psi_10), np.std(error_gru_all_psi_10)))
print('-----------------------------')
print('Error reports GRU PSI DEDICATED')
print('error_gru_psi_01: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_01), np.max(error_gru_psi_01), np.mean(error_gru_psi_01), np.std(error_gru_psi_01)))
print('error_gru_psi_02: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_02), np.max(error_gru_psi_02), np.mean(error_gru_psi_02), np.std(error_gru_psi_02)))
print('error_gru_psi_03: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_03), np.max(error_gru_psi_03), np.mean(error_gru_psi_03), np.std(error_gru_psi_03)))
print('error_gru_psi_04: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_04), np.max(error_gru_psi_04), np.mean(error_gru_psi_04), np.std(error_gru_psi_04)))
print('error_gru_psi_05: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_05), np.max(error_gru_psi_05), np.mean(error_gru_psi_05), np.std(error_gru_psi_05)))
print('error_gru_psi_06: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_06), np.max(error_gru_psi_06), np.mean(error_gru_psi_06), np.std(error_gru_psi_06)))
print('error_gru_psi_07: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_07), np.max(error_gru_psi_07), np.mean(error_gru_psi_07), np.std(error_gru_psi_07)))
print('error_gru_psi_08: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_08), np.max(error_gru_psi_08), np.mean(error_gru_psi_08), np.std(error_gru_psi_08)))
print('error_gru_psi_09: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_09), np.max(error_gru_psi_09), np.mean(error_gru_psi_09), np.std(error_gru_psi_09)))
print('error_gru_psi_10: Min {}. Max: {}. Mean: {}. Std: {}'.format(np.min(error_gru_psi_10), np.max(error_gru_psi_10), np.mean(error_gru_psi_10), np.std(error_gru_theta_10)))


