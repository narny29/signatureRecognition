import numpy as np
import tensorflow as tf
import keras
import pickle
import inspect,os
from os.path import isfile, join
from scipy.misc import imread, imresize, imsave
from math import log
import xgboost as xgb
from sklearn.metrics import accuracy_score



data = np.load("file_name_vs_hash.npy")
n_samples = data.shape[0]



#train with only valid images

curr_dir = os.getcwd()
inp_dir = curr_dir + "/resized/"
files = [f for f in os.listdir(inp_dir) if isfile(join(inp_dir,f))]
img_width, img_height = 400, 100


t_data = []
for row in data:
	if row[0][1] == "F":
		temp = "1"
	else:
		temp = "0"
	t_data.append([row[1], int(temp + row[0][4:7]), int(temp + row[0][9:-4]), int(row[0][7:9])])
#								 who signed					who's sign 			sign number


new_data = {}
for row in t_data:
	if row[-3] == row[-2]:
		if not row[-2] in new_data:
			new_data[row[-2]] = []
		new_data[row[-2]].append(row[:-3])

# print(len(new_data.keys()))

t_data = []
tot_similar = 0
tot_diff = 0
for name in new_data:
	for row1 in new_data[name]:
		for row2 in new_data[name]:
			t = []
			for i in range(len(row1[0])):
				t.append(abs(row1[0][i]-row2[0][i]))
			t_data.append(t + [1])
			tot_similar += 1


k = int(log(25)/log(len(new_data) - 1)) + 1				#25 is 5*5 for the total possible combinations of genuine siganture combinaton of 2
# print(k)

while tot_diff < 1.2*tot_similar:
	for name in new_data:
		leng = min(k, len(new_data[name]))
		indices1 = np.random.permutation(leng)
		for i in indices1:
			row1 = new_data[name][i]
			for other_name in new_data:
				if not name==other_name:
					leng = min(k, len(new_data[other_name]))
					indices2 = np.random.permutation(leng)
					for j in indices2:
						row2 = new_data[other_name][j]
						t = []
						for i in range(len(row1[0])):
							t.append(abs(row1[0][i]-row2[0][i]))
						t_data.append(t + [0])
						tot_diff += 1

# print(tot_diff, tot_similar)

#you must shuffle the t_data


t_data = np.asarray(t_data)
# print(t_data.shape)
n_samples = t_data.shape[0]
data_split = 0.9
dataX, dataY = t_data[:,:-1], t_data[:, -1]


indices = np.random.permutation(n_samples)
# print(indices)
split = int(n_samples * data_split)
training_idx, test_idx = indices[:split], indices[split:]
trainX, testX = dataX[training_idx,:], dataX[test_idx,:]
trainY, testY = dataY[training_idx], dataY[test_idx]



# model = keras.models.Sequential([
#       keras.layers.Dense(10, input_shape = (4, ), activation=tf.nn.tanh),
#       # keras.layers.Dropout(0.05),
#       keras.layers.Dense(8, activation=tf.nn.tanh),
#       keras.layers.Dense(6, activation=tf.nn.tanh),
#       keras.layers.Dense(4, activation=tf.nn.tanh),
#       keras.layers.Dense(2, activation=tf.nn.softmax)
#     ])


# optimizer = 'adam'
# model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# h = model.fit(trainX, trainY,  batch_size = 64 ,validation_data = (testX, testY),epochs=50)

# model.evaluate(testX, testY)[1]


model = xgb.XGBClassifier(base_score = 0.7,max_depth = 25)
model.fit(trainX, trainY)
pred = model.predict(testX)
# print(pred)
print(accuracy_score(pred, testY))

pickle.dump(model, open("models/model_similarity.dump", "wb"))