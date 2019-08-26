import os
import numpy as np
import cv2
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from mlxtend.plotting import plot_decision_regions
from PIL import Image
from sklearn.metrics import classification_report,accuracy_score
import random

# def compose_dataset_file():

anchor_boxes = {
	'1:2': ['bottle', 'vase', 'pottted_plant', 'refrigerator'],
	'2:1': ['microwave', 'tvmonitor', 'sofa', 'bed'],
	'1:1': ['cup', 'oven', 'toilet', 'clock'],
}

def construct_dataset(lower, upper, train_class):
	test_no_dup = 0
	train_class = int(train_class)
	print(train_class)
	X_train = np.zeros((upper-lower, 20000))  # make it dynamic size, make it 3 channels if found to increase accuracy else this will be faster
	Y_train = np.zeros((upper-lower))
	X_train_count = 0

	# with open('./data/'+str(lower)+str(upper)+".txt") as file:
	# fname = "./data/images_annos_0_4999_subset.txt"
	fname = "./data/images_annos_0_14999.txt"

	# with open("./data/images_annos_0_9999.txt") as file:

	x_ar = [] 
	y_ar = []
	w_ar = []
	h_ar = []
	cls_ar = []

	dim_x = 0
	dim_y = 0

	with open(fname) as file:
		line = file.readline()
		while line and X_train_count < (upper-lower):
			data = line.split()
			if len(data) == 1:
				# print("processing filename in textfile")
				if len(x_ar) != 0:
					backgrd_amount = len(x_ar)
					print("backgrd_amount")
					print(backgrd_amount)
					
					for i in range(backgrd_amount):
						# print("in for finding background boxes")s
						b_x = random.randint(0, int(dim_x))
						b_y = random.randint(0, int(dim_y))
						while not(b_x in x_ar) and not(b_y in y_ar):
							# print("in wile finding background box's coords and dims")

							b_x = random.randint(0, int(dim_x))
							b_y = random.randint(0, int(dim_y))

							b_h_index = random.randint(0, len(h_ar) - 1)
							# print("b_h_index")
							b_h = h_ar[b_h_index]

							b_w_index = random.randint(0, len(w_ar) - 1)
							# print("b_w_index")
							b_w = w_ar[b_w_index]

						backgrds = cv2.imread(filename)
						backgrds = cv2.cvtColor(backgrds, cv2.COLOR_BGR2RGB)

						backgrd = backgrds[b_y:b_y+b_h, b_x:b_x+b_w]
						
						backgrd_img = Image.fromarray(backgrd)
						# backgrd_img.show()

						backgrd = cv2.cvtColor(backgrd, cv2.COLOR_BGR2GRAY)
						backgrd = cv2.resize(backgrd, (100, 200))
						backgrd = np.float32(backgrd)/255
						# x_objects = cv2.cvtColor(x_object, cv2.COLOR_BGR2GRAY) # temporary
						backgrd = cv2.Sobel(backgrd, cv2.CV_32F, 0, 1, ksize=1) # temporary
						backgrd = backgrd.flatten()
						
						X_train[X_train_count] = backgrd
						Y_train[X_train_count] = 0
						X_train_count += 1

				try:
					filename = data[0]
					# x_objects = cv2.imread(data[0]) # image with possible multiple objects.
				except Exception as e:
					print(str(e))
				
				x_ar = [] 
				y_ar = []
				w_ar = []
				h_ar = []
			else:
				# print("processing image anno in textfile")

				x = data[0]
				x = math.floor(float(x))

				y = data[1]
				y = math.floor(float(y))

				w = data[2]
				w = math.floor(float(w))
				
				h = data[3]
				h = math.floor(float(h))

				clas = data[5]
				clas = int(clas)
				# print(clas)
				x_objects = cv2.imread(filename) # image with possible multiple objects.

				dim_x = x_objects.shape[1]
				dim_y = x_objects.shape[0]

				x_objects = cv2.cvtColor(x_objects, cv2.COLOR_BGR2RGB)
				if clas == train_class:
					x_ar.append(x)
					y_ar.append(y)
					w_ar.append(w)
					h_ar.append(h)

					if test_no_dup < 5:
						img = Image.fromarray(x_objects)
						# img.show()

						x_object = x_objects[y:y+h, x:x+w]
						img = Image.fromarray(x_object)
						# img.show()
						test_no_dup += 1

					x_object = x_objects[y:y+h, x:x+w]
					img = Image.fromarray(x_object)
					# img.show()

					# x_object = x_objects[y:y+h, x:x+w, 0:1]
					x_object = cv2.cvtColor(x_object, cv2.COLOR_BGR2GRAY) # temporary
					x_object = cv2.resize(x_object, (100, 200))
					x_object = np.float32(x_object)/255
					# x_objects = cv2.cvtColor(x_object, cv2.COLOR_BGR2GRAY) # temporary
					x_object = cv2.Sobel(x_object, cv2.CV_32F, 0, 1, ksize=1) # temporary
					x_object = x_object.flatten()
					
					X_train[X_train_count] = x_object
					Y_train[X_train_count] = train_class
					X_train_count += 1

			line = file.readline()
	return X_train, Y_train

def train_and_predict(anchor_box, training_for):
	X_data = []
	Y_data = []
	for my_class in anchor_box:
		a_X_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_X.npy')
		X_data.append(a_X_data)
		
		if my_class == training_for:
			a_Y_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_Y.npy')
		else:
			a_Y_data = np.zeros((a_X_data.shape[0]))
		Y_data.append(a_Y_data)

	X_data = np.concatenate(([a_X_data for a_X_data in X_data]))
	Y_data = np.concatenate(([a_Y_data for a_Y_data in Y_data]))

	svc = svm.SVC()

	# sanity check

	print(Y_data[0])
	print(Y_data[100])
	print(Y_data[200])

	print(svc.fit(X_data, Y_data))

	test_p = cv2.imread('/home/charlie/Desktop/potted_plant_test.png') 
	test_v = cv2.imread('/home/charlie/Desktop/vase_test.png') 
	test_r = cv2.imread('/home/charlie/Desktop/refrigerator_test.png') 
	test_b = cv2.imread('/home/charlie/Desktop/bottle_test.png')
	test_pe = cv2.imread('/home/charlie/Desktop/xxx.jpg')
	test_k = cv2.imread('/home/charlie/Desktop/knife_test.png')

	tst_ar = []

	tst_ar.append(test_p)
	tst_ar.append(test_v)
	tst_ar.append(test_r)
	tst_ar.append(test_b)
	tst_ar.append(test_pe)
	tst_ar.append(test_k)

	new_ar = []

	for tst in tst_ar:
		tst = cv2.resize(tst, (100, 200))
		tst = np.float32(tst)/255
		# tst = tst[0:tst.shape[0], 0:tst.shape[1], 0:1]
		tst = cv2.cvtColor(tst, cv2.COLOR_BGR2GRAY)
		tst = cv2.Sobel(tst, cv2.CV_32F, 0, 1, ksize=1)
		tst = tst.flatten()
		new_ar.append(tst)

	for new in new_ar:
		print(svc.predict([new]))

def train_and_predict_multiclass(anchor_box, suffix, class_no, hog_windows=None): # classes to train
	# classes = anchor_box[anchor_box]

	X_data = []
	Y_data = []

	for my_class in anchor_box:
		a_X_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_X'+suffix+'.npy')
		X_data.append(a_X_data)
		a_Y_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_Y'+suffix+'.npy')
		Y_data.append(a_Y_data)

	X_data = np.concatenate(([a_X_data for a_X_data in X_data]))
	Y_data = np.concatenate(([a_Y_data for a_Y_data in Y_data]))

	svc = svm.LinearSVC()
	print(svc.fit(X_data, Y_data))

	preds = []
	gts = []

	if hog_windows:
		X_test = []
		for hog_window in hog_windows:
			X_test.append(hog_window)

	else:
		X_test, X_throw_away = construct_dataset(0, 30, class_no)
		print(X_test[0].shape)
		print(X_test[0][1000])
		print(X_test[0][100])
		print(X_test[0][10530])
		print(X_test[15].shape)
		print(X_test[15][1000])
		print(X_test[15][1043])
		print(X_test[15][120])
		print(X_test[29].shape)
		print(X_test[29][1000])
		print(X_test[29][1340])
		print(X_test[29][14])

	i = 1

	for new in X_test:
		print("\n")
		print("no. "+str(i))
		i+=1
		print(int(svc.predict([new])[0]))
		preds.append(int(svc.predict([new])[0]))
		gts.append(int(class_no))
		# gts.append(class_no)
		# print(class_no)

	print(classification_report(preds, gts))

def save_train_data(class_name, X, Y, suffix):
	np.save('data/'+str(class_name)+'/'+str(class_name)+'_X'+suffix, X)
	np.save('data/'+str(class_name)+'/'+str(class_name)+'_Y'+suffix, Y)

if __name__ == "__main__":
	# ---------------------------------------------------------------------------
	X, Y = construct_dataset(0, 200, "1")
	# # print(X[0])
	# # print(X[99])
	# # print(X[199])
	save_train_data('person', X, Y, '200_with_backgrd')

	# X, Y = construct_dataset(0, 200, "43")
	# # print(X[0])
	# # print(X[99])
	# # print(X[199])
	# save_train_data('tennis racket', X, Y, '200_with_backgrd')

	# X, Y = construct_dataset(0, 200, "2")
	# # print(X[0])
	# # print(X[99])
	# # print(X[199])
	# save_train_data('bicycle', X, Y, '200_with_backgrd')

	# train_and_predict_multiclass(['orange', 'tennis racket', 'bicycle', 'apple'], '200')
	# train_and_predict_multiclass(['apple', 'carrot', 'tvmonitor', 'sofa'], '200')
	# train_and_predict_multiclass(['bicycle', 'orange', 'car', 'cake'], '200')
	# train_and_predict_multiclass(['aeroplane', 'person', 'motorbike', 'umbrella'], '200')
	# train_and_predict_multiclass(['tennis racket', 'sofa', 'bus', 'train'], '200', '7')
	# ---------------------------------------------------------------------------





































# -----------------------------------------------------------------------
# 	iris = datasets.load_iris()
# 	X = iris.data[:, :2]
# 	Y = iris.target

# 	C = 1.0
# 	models = (svm.SVC(kernel='linear', C=C),
# 			  svm.LinearSVC(C=C, max_iter=10000),
# 			  svm.SVC(kernel='rbf', gamma=0.7, C=C),
# 			  svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
# 	models = (clf.fit(X, Y) for clf in models)

# 	titles = ('aaa',
# 			  'bbb',
# 			  'ccc',
# 			  'ddd')

# 	fig, sub = plt.subplots(2, 2)
# 	plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 	X0, X1 = X[:, 0], X[:, 1]
# 	xx, yy = make_meshgrid(X0, X1)

# 	for clf, title, ax in zip(models, titles, sub.flatten()):
# 		plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# 		ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# 		ax.set_xlim(xx.min(), xx.max())
# 		ax.set_ylim(yy.min(), yy.max())
# 		ax.set_xlabel('Sepal length')
# 		ax.set_ylabel('Sepal width')
# 		ax.set_xticks(())
# 		ax.set_yticks(())
# 		ax.set_title(title)

# plt.show()
# -------------------------------------------------------------------------