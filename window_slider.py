import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from hog_svm import train_and_predict_multiclass

def num_to_class(case):
    switcher = {
        '1': 'prn',
        '2': 'bcy',
        '3': 'car',
        '4': 'mtrb',
        '5': 'arpl',
        '6': 'bus',
       	'7': 'trn',
       	'43': 'tns',
       	'55': 'org',
       	'53': 'apl',
       	'58': 'hdg',
       	'57': 'crt',
       	'62': 'chr',
       	'42': 'srf',
    }
    return switcher.get(case, "invalid")

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
	hori_window_steps = (image_wid // anchor_box[0])
	# vert_window_steps = (image_hei // anchor_box[1]) * stride
	# vert_window_steps = (image_hei // anchor_box[1]) * 2
	vert_window_steps = (image_hei // anchor_box[1])

	poss_objects = cv2.imread(filename)
	poss_objects = cv2.cvtColor(poss_objects, cv2.COLOR_BGR2RGB)

	i = 1

	for hori_step in range(hori_window_steps):
		for vert_step in range(vert_window_steps):
			print("y: " + str(int(vert_step * anchor_box[1] * 0.5)) + ":" + str(int(vert_step * anchor_box[1] * 0.5 + anchor_box[1] * 0.5)))
			print("x: " + str(int(hori_step * anchor_box[0] * 0.5)) + ":" + str(int(hori_step * anchor_box[0] * 0.5 + anchor_box[0] * 0.5)))
			# poss_object = poss_objects[int(vert_step * anchor_box[1] * 0.5): int(vert_step * anchor_box[1] * 0.5 + anchor_box[1]),\
									   # int(hori_step * anchor_box[0] * 0.5): int(hori_step * anchor_box[0] * 0.5 + anchor_box[0])]
			poss_object = poss_objects[int(vert_step * anchor_box[1]): int(vert_step * anchor_box[1] + anchor_box[1]),\
									   int(hori_step * anchor_box[0]): int(hori_step * anchor_box[0] + anchor_box[0])]

			img = Image.fromarray(poss_object)

			print("hog_window.shape 1")
			print(img.size)

			# img.show()
			# for i in range(hori_step+vert_step):
			img.save('to_compare/'+str(i)+'.jpg')
			i+=1

			# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # remove 2 channels
			# img = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_BGR2GRAY) # remove 2 channels
			img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY) # remove 2 channels
			hog_window = cv2.resize(img, (100, 200))
			print("hog_window.shape 2")
			print(hog_window.size)
			hog_window = np.float32(hog_window)/255
			hog_window = cv2.Sobel(hog_window, cv2.CV_32F, 0, 1, ksize=1) # temporary
# 
			# img_show = Image.fromarray(hog_window)
			# img_show.show()

			# hog_window_show = Image.fromarray(hog_window) # temp
			hog_window = hog_window.flatten()

			# print(hog_window.shape)

			hog_windows.append(hog_window)

	return hog_windows, hori_window_steps, vert_window_steps

def draw_bboxes(anchor_box, window_preds, orignal_img, hori_win_amount, vert_win_amount):
	bbox_coords = []
	x = 0
	y = 0
	bbox = []
	print(hori_win_amount)
	print(vert_win_amount)
	new_win_preds = np.array(window_preds).reshape(hori_win_amount, vert_win_amount)
	new_win_preds = list(new_win_preds)
	print(new_win_preds)

	for vert_win in range(vert_win_amount):
		for hori_win in range(hori_win_amount):
			# if new_win_preds[vert_win][hori_win] != 0:
			if new_win_preds[hori_win][vert_win] != 0:
				bbox.append(hori_win * anchor_box[0]) # x
				bbox.append(vert_win * anchor_box[1]) # y
				bbox.append(anchor_box[0]) # w
				bbox.append(anchor_box[1]) # h
				# bbox.append(new_win_preds[vert_win][hori_win])
				bbox.append(new_win_preds[hori_win][vert_win])
				bbox_coords.append(bbox)
				bbox = []

	pink = (255,105,180)
	original = cv2.imread(orignal_img)
	original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

	print(bbox_coords)

	for bbox in bbox_coords:
		cv2.line(original,(bbox[0],bbox[1]), (bbox[0]+bbox[2], bbox[1]), pink, 1)
		cv2.line(original,(bbox[0]+bbox[2], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), pink, 1)
		cv2.line(original,(bbox[0]+bbox[2], bbox[1]+bbox[3]), (bbox[0], bbox[1]+bbox[3]), pink, 1)
		cv2.line(original,(bbox[0], bbox[1]+bbox[3]), (bbox[0], bbox[1]), pink, 1)
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		clss = num_to_class(str(bbox[4]))
		cv2.putText(original, clss, (bbox[0],bbox[1]+25), font, 1, pink, 2, cv2.LINE_AA)

	plt.axis("off")
	plt.imshow(original)
	plt.show()
	# cv2.imshow("lekka", original)

if __name__ == "__main__":
	# filename = "/home/charlie/Desktop/yolo/yolo-v3/data/coco/cocoapi/coco/images/train2017/train2017/000000323639.jpg"
	# filename = "got_chairs.jpg"
	# anchor_box = []
	# anchor_box.append(50) # w
	# anchor_box.append(50) # h

	# hog_windows, hori_win_steps, vert_win_steps = slide_windows(filename, anchor_box, 2)

	# # preds = train_and_predict_multiclass(['orange', 'tennis racket', 'bicycle', 'person'], '200_with_backgrd', '7', hog_windows)
	# preds = train_and_predict_multiclass(['chair'], '1500', '0', hog_windows)

	# print(preds)

	# draw_bboxes(anchor_box, preds, filename, hori_win_steps, vert_win_steps)


	# filename = "got_carrots.jpg"
	# anchor_box = []
	# anchor_box.append(50) # w
	# anchor_box.append(50) # h

	# hog_windows, hori_win_steps, vert_win_steps = slide_windows(filename, anchor_box, 2)

	# # preds = train_and_predict_multiclass(['orange', 'tennis racket', 'bicycle', 'person'], '200_with_backgrd', '7', hog_windows)
	# preds = train_and_predict_multiclass(['carrot'], '1500', '0', hog_windows)

	# print(preds)

	# draw_bboxes(anchor_box, preds, filename, hori_win_steps, vert_win_steps)

	# filenames.append("got_surfboards.jpg")
	filenames = ["6.jpg", "26.jpg", "30.jpg", "got_carrots.jpg", "got_chairs.jpg", "got_surfboards.jpg", "got_surfboards1.jpg", "got_surfboards2.jpg"]
	# filenames.append("got_surfboards1.jpg")
	# filenames.append("got_chairs1.jpg")
	# filenames.append("got_chairs7.jpg")
	# filenames.append("got_chairs6.jpg")
	# filenames.append("got_chairs5.jpg")
	# filenames.append("got_surfboards2.jpg")
	# filenames.append("got_chairs2.jpg")
	# filenames.append("got_surfboards3.jpg")
	# filenames.append("got_chairs3.jpg")
	# filenames.append("got_surfboards4.jpg")
	# filenames.append("got_chairs4.jpg")
	anchor_box = []
	anchor_box.append(50) # w
	anchor_box.append(50) # h

	for filename in filenames:
		hog_windows, hori_win_steps, vert_win_steps = slide_windows(filename, anchor_box, 2)
		# preds = train_and_predict_multiclass(['orange', 'tennis racket', 'bicycle', 'person'], '200_with_backgrd', '7', hog_windows)
		preds = train_and_predict_multiclass(['chair', 'surfboard', 'carrot'], '300', '0', hog_windows)
		# print(preds)
		draw_bboxes(anchor_box, preds, filename, hori_win_steps, vert_win_steps)