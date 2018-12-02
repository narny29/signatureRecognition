import numpy as np
import tensorflow as tf
import keras
import pickle




data = np.load("data_forgery.npy")
n_samples = data.shape[0]
data_split = 0.9
img_width, img_height = 400, 100
dataX, dataY = data[:,:img_width * img_height], data[:, -3:]
dataX = dataX.reshape((n_samples, img_width, img_height, 1))
print(dataX.shape, dataY.shape)


indices = np.random.permutation(n_samples)
# print(indices)
split = int(n_samples * data_split)
training_idx, test_idx = indices[:split], indices[split:]
trainX, testX = dataX[training_idx,:], dataX[test_idx,:]
trainX, testX = trainX/255, testX/255
labels = np.array([ 1 if dataY[i][0]==dataY[i][2] else 0 for i in range(n_samples)])
trainY, testY = labels[training_idx], labels[test_idx]

print(trainX.shape, trainY.shape)



model = keras.models.Sequential([
      keras.layers.Conv2D(filters = 8, kernel_size = 5, padding='valid', input_shape = (img_width, img_height, 1), activation="elu"),
      keras.layers.Conv2D(filters = 16, kernel_size = 5, padding='valid', activation="elu"),
      keras.layers.Conv2D(filters = 32, kernel_size = 5, padding='valid', activation="elu"),
      # keras.layers.Conv2D(filters = 64, kernel_size = 5, padding='valid', activation="elu"),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.25),
      keras.layers.Dense(10, activation=tf.nn.sigmoid),
      keras.layers.Dropout(0.25),
      keras.layers.Dense(2, activation=tf.nn.softmax)
    ])


optimizer = 'adam'
model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
h = model.fit(trainX, trainY,  batch_size = 32 ,validation_data = (testX, testY),epochs=5)

model.evaluate(testX, testY)[1]



model_json = model.to_json()
model_name = "model1"
with open("models/"+model_name+".json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("models/"+model_name+".h5")
