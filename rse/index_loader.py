from skimage import io
from skimage.feature import SIFT
import numpy as np
from collections import Counter
import nmslib
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import cv2

index = nmslib.init(method='hnsw', space='l2')
index.loadIndex(r"D:\RealCoderZ\ReverseImageSearch\PythonSIFT\index_9.bin")

with open(r"D:\RealCoderZ\ReverseImageSearch\desc_metadata9.pkl", 'rb') as f:
    img_name, original_indices, matrix = pickle.load(f)

detector_extractor = SIFT()

print('----------------Index with metadata loaded---------------')

image = io.imread(r"D:\RealCoderZ\ReverseImageSearch\rse\static\images9\img_1293.jpg", as_gray = True) 
detector_extractor.detect_and_extract(image)
inp_img_desc = detector_extractor.descriptors

all_indx = []
for i in inp_img_desc:
    query = i    
    indices, distances = index.knnQuery(query, k=5)
    all_indx.append(indices)

original_most = []
for j in all_indx:
    og_most = []
    for i in j:
        og_most.append(original_indices[i])
    original_most.append(og_most)

unique_list = []
for i in original_most:
    u_list = []
    for j in i:
        if j not in u_list:
            u_list.append(j)
            
    unique_list.append(u_list)

nf = np.concatenate(unique_list)

def most_repeated_value(lst):
    count_dict = Counter(lst)
    most_repeated = sorted(count_dict.items(), key=lambda x: x[1], reverse = True)
    return most_repeated

mr = most_repeated_value(nf)
found = [img_name[i[0]] for i in mr[:5]]

print(found)

# a = cv2.imread(r"D:\RealCoderZ\ReverseImageSearch\rse\static\images9\img_1316.jpg")
# b = cv2.imread(r"D:\RealCoderZ\ReverseImageSearch\rse\static\images9\img_3219.jpg")

# Verti = np.concatenate((a, b), axis=1)

# cv2.imshow('VERTICAL', Verti)

# cv2.waitKey(0)
# io.imshow(r"D:\RealCoderZ\ReverseImageSearch\rse\static\images9\img_1316.jpg")
# plt.show()
# io.imshow(r"D:\RealCoderZ\ReverseImageSearch\rse\static\images9\img_3219.jpg")
# plt.show()
#cluster 9 - 1306, 3723.jpg, img_1316, img_1336, img_2356, img_2362, img_3219, 1614, 2388, 1454, 5061, 2403, 2415, 2788, 2605, 2442, 2644, 1293


# import os
# import pickle
# import shutil

# images = ['img_1306.jpg', 'img_1316.jpg', 'img_3723.jpg', 'img_1336.jpg', 'img_2356.jpg', 'img_2362.jpg', 'img_3219.jpg', 'img_1614.jpg', 'img_2388.jpg', 'img_1454.jpg', 'img_5061.jpg', 'img_2403.jpg', 'img_2415.jpg', 'img_2788.jpg', 'img_2605.jpg', 'img_2442.jpg', 'img_2644.jpg', 'img_1293.jpg']

# with open(r"D:\RealCoderZ\ReverseImageSearch\desc_metadata9.pkl", 'rb') as f:
#     img_name, orginal_indices, desc = pickle.load(f)

# os.makedirs("D:/RealCoderZ/ReverseImageSearch/test", exist_ok = True)

# for img in images:
#     shutil.copy(f"D:/RealCoderZ/ReverseImageSearch/dataAllimages/{img}", f"D:/RealCoderZ/ReverseImageSearch/test/{img}")