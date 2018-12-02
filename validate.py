import pickle
from keras.models import model_from_json


json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")



data= np.load("data.npy")
n_samples = data.shape[0]
data_split = 0.9
img_width, img_height = 400, 100
dataX, dataY = data[:,:img_width * img_height], data[:, -3:]
dataX = dataX.reshape((n_samples, img_width, img_height, 1))
print(dataX.shape, dataY.shape)


