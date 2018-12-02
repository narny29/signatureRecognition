import inspect,os
from os.path import isfile, join
from scipy.misc import imread, imresize, imsave
import numpy as np



curr_dir = os.getcwd()
inp_dir = curr_dir + "/all/"
outp_dir = curr_dir + "/resized/"
files = [f for f in os.listdir(inp_dir) if isfile(join(inp_dir,f))]
img_width, img_height = 400, 100


# for f in files:
# 	img = imread(inp_dir+f, flatten = True)
# 	img = imresize(img, (img_height, img_width), interp = 'bicubic')
# 	imsave(outp_dir+f, img)



files = [f for f in os.listdir(outp_dir) if isfile(join(outp_dir,f))]

data = []
for f in files:
	im = imread(outp_dir+f)
	data.append(im.reshape(img_width*img_height).tolist() + [int(f[4:7])] + [int(f[9:-4])] + [int(f[7:9])])

np.save("data_forgery.npy", data)