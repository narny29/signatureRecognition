from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from scipy.misc import imread, imresize, imsave
from PIL import Image
import os
import pickle
from os.path import isfile, join
import numpy as np
import imagehash
from keras.models import model_from_json

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST' and len(list(request.files.keys())) and len(request.form["custId"])>0:
		i=1
		for key in list(request.files.keys()):
			filename = photos.save(request.files[key],  folder = request.form["custId"])
		i+=1
	return render_template('upload.html')


@app.route('/check', methods = ['GET', 'POST'])
def check():
	with open('models/model1.json', 'r') as f:
		model_forgery = model_from_json(f.read())
		model_forgery.load_weights('models/model1.h5')


	model_similarity = pickle.load(open("models/model_similarity.dump", "rb"))


	if request.method == 'POST' and len(list(request.files.keys())) and len(request.form["custId"])>0:
		folders = [f for f in os.listdir("static/img")]
		hash_func = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]
		if request.form["custId"] in folders:
			filename = photos.save(request.files["photo1"],  folder = request.form["custId"]+"/test")
			files = [f for f in os.listdir("static/img/"+request.form["custId"]) if isfile(join("static/img/"+request.form["custId"],f))]
			img = imread("static/img/" + filename, flatten = True)
			img = imresize(img, (400, 100), interp = 'bicubic')
			imsave("static/img/random/" + filename.split("/")[-1], img)
			im = Image.open("static/img/random/" + filename.split("/")[-1])
			hash_arr = []
			for hashfunc in hash_func:
				hash_arr.append(hashfunc(im))
			img = img.reshape(400 * 100).tolist()
			img = np.asarray(img).reshape(1, 400, 100, 1)
			forgery_proba = model_forgery.predict(img)
			similarity_proba = []
			for file in files:
				inp = []
				img1 = imread("static/img/" + request.form["custId"] + "/" + file, flatten = True)
				img1 = imresize(img1, (400, 100), interp = 'bicubic')
				imsave("static/img/random/" + file, img1)
				im = Image.open("static/img/random/" + file)
				i = 0
				for hashfunc in hash_func:
					inp.append(abs(hashfunc(im)-hash_arr[i]))
					i+=1
				inp = np.asarray(inp)
				similarity_proba.append(model_similarity.predict_proba([inp])[:,1])
			similar_proba = sum(similarity_proba)/len(similarity_proba)
			if similar_proba < 0.55:
				return "The signature given doesn't match the person or is prabably a forged one. It gives a " + str(similar_proba) + " match status"
			else:
				return "The signature given is of the same person. The mode gives a " + str(similar_proba) + " genuine status"
		else:
			return "Requested CustomerID not found"
	return render_template('check.html')



if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port,debug=True)