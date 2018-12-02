import inspect,os
from os.path import isfile, join
from scipy.misc import imread, imresize, imsave
import numpy as np
from math import log



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
	if f[1] == "F":
		temp = "1"
	else:
		temp = "0"
	data.append(im.reshape(img_width*img_height).tolist() + [int(temp + f[4:7])] + [int(temp + f[9:-4])] + [int(f[7:9])])
															#who signed					who's sign 			sign number

new_data = {}
for row in data:
	if row[-3] == row[-2]:
		if not row[-2] in new_data:
			new_data[row[-2]] = []
		new_data[row[-2]].append(row[:-3])



data = []
tot_similar = 0
tot_diff = 0
for name in new_data:
	for row1 in new_data[name]:
		for row2 in new_data[name]:
			data.append(row1[:img_width * img_height] + row2[:img_width * img_height] + [1])
			tot_similar += 1

k = int(log(25)/log(len(new_data) - 1)) + 1				#25 is 5*5 for the total possible combinations of genuine siganture combinaton of 2
print(k)
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
						data.append(row1[:img_width * img_height] + row2[:img_width * img_height] + [0])
						tot_diff += 1


print(tot_diff, tot_similar)






np.save("data_similarity.npy", data)