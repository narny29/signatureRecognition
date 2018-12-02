from PIL import Image
import six
import imagehash
import sys, os
from os.path import isfile, join
import numpy as np

# types of hash
# imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash


curr_dir = os.getcwd()
inp_dir = curr_dir + "/resized/"
# outp_dir = curr_dir + "/genuine/"
hash_func = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]#[imagehash.average_hash]#

# h1 = imagehash.dhash(Image.open(outp_dir + i))
# h2 = imagehash.dhash(Image.open(outp_dir + j))
# print(i, j, abs(h1-h2))



data = []
j = 0
files = [f for f in os.listdir(inp_dir) if isfile(join(inp_dir,f))]

for i in files:
    t = []
    for hash_f in hash_func:
        t.append(hash_f(Image.open(inp_dir + i)))
    data.append([i, t])

np.save("file_name_vs_hash.npy", data)