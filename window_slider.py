import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from hog_svm import train_and_predict_multiclass

def slide_windows(filename, anchor_box, stride=1):
	# ------------------------------------
	# params:
	# 	filename: of entire image
	# 	anchor_box: class representing sliding window dimensions
	# returns:
	# 	xxxxxxxxxxxxxxxx return list of all identified objects in entire image xxxxxxxxxxxxxxx
	# 	return list of all hog windows to be used by SVM
	# ------------------------------------
	hog_windows = []

	image = cv2.imread(filename) # entire image
	# preprocessing

	image_wid = image.shape[1]
	image_hei = image.shape[0]

	image_wid = image_wid - (image_wid % anchor_box[0])
	image_hei = image_hei - (image_hei % anchor_box[1])

	print("image_wid")
	print(image_wid)
	print("image_hei")
	print(image_hei)

	image = cv2.resize(image, (image_wid, image_hei))

	# mod the image dimension with window dimension then minus remainder from image dimension by resizing
	# this prevents out of bound indexes when snipping out windows of the image array
	
	# hori_window_steps = (image_wid // anchor_box[0]) * stride
	# hori_window_steps = (image_wid // anchor_box[0]) * 2
	hori_window_steps = (image_wid // anchor_box[0]) * 2
	# vert_window_steps = (image_hei // anchor_box[1]) * stride
	# vert_window_steps = (image_hei // anchor_box[1]) * 2
	vert_window_steps = (image_hei // anchor_box[1]) * 2

	poss_objects = cv2.imread(filename)
	poss_objects = cv2.cvtColor(poss_objects, cv2.COLOR_BGR2RGB)

	i = 1

	for hori_step in range(hori_window_steps):
		for vert_step in range(vert_window_steps):
			print("y: " + str(int(vert_step * anchor_box[1] * 0.5)) + ":" + str(int(vert_step * anchor_box[1] * 0.5 + anchor_box[1] * 0.5)))
			print("x: " + str(int(hori_step * anchor_box[0] * 0.5)) + ":" + str(int(hori_step * anchor_box[0] * 0.5 + anchor_box[0] * 0.5)))
			poss_object = poss_objects[int(vert_step * anchor_box[1] * 0.5): int(vert_step * anchor_box[1] * 0.5 + anchor_box[1]),\
									   int(hori_step * anchor_box[0] * 0.5): int(hori_step * anchor_box[0] * 0.5 + anchor_box[0])]
			# poss_object = poss_objects[int(vert_step * anchor_box[1]): int(vert_step * anchor_box[1] + anchor_box[1]),\
			# 						   int(hori_step * anchor_box[0]): int(hori_step * anchor_box[0] + anchor_box[0])]

			img = Image.fromarray(poss_object)

			print("hog_window.shape 1")
			print(img.size)

			# img.show()
			# for i in range(hori_step+vert_step):
			img.save(str(i)+'.jpg')
			i+=1

			# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # remove 2 channels
			# img = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_BGR2GRAY) # remove 2 channels
			img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY) # remove 2 channels
			hog_window = cv2.resize(img, (100, 200))
			print("hog_window.shape 2")
			print(hog_window.size)
			hog_window = np.float32(hog_window)/255
			hog_window = cv2.Sobel(hog_window, cv2.CV_32F, 0, 1, ksize=1) # temporary

			# img_show = Image.fromarray(hog_window)
			# img_show.show()

			# hog_window_show = Image.fromarray(hog_window) # temp
			hog_window = hog_window.flatten()

			# print(hog_window.shape)

			hog_windows.append(hog_window)

	return hog_windows

if __name__ == "__main__":
	filename = "/home/charlie/Desktop/yolo/yolo-v3/data/coco/cocoapi/coco/images/train2017/train2017/000000000715.jpg"
	anchor_box = []
	anchor_box.append(50) # w
	anchor_box.append(50) # h

	hog_windows = slide_windows(filename, anchor_box, 2)

	train_and_predict_multiclass(['orange', 'tvmonitor', 'tennis racket', 'person'], '200', '7', hog_windows)