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
	fname = "./data/images_annos_0_4999_subset.txt"
	# fname = "./data/images_annos_0_14999.txt"
	print(os.path.exists(fname))

	# with open("./data/images_annos_0_9999.txt") as file:
	with open(fname) as file:
		line = file.readline()
		while line and X_train_count < (upper-lower):
			# print(line)
			data = line.split()
			if len(data) == 1:
				try:
					filename = data[0]
					# x_objects = cv2.imread(data[0]) # image with possible multiple objects.
				except Exception as e:
					print(str(e))
			else:
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
				x_objects = cv2.cvtColor(x_objects, cv2.COLOR_BGR2RGB)
				if clas == train_class:
					if test_no_dup < 5:
						img = Image.fromarray(x_objects)
						img.show()

						x_object = x_objects[y:y+h, x:x+w]
						img = Image.fromarray(x_object)
						img.show()
						test_no_dup += 1

					x_object = x_objects[y:y+h, x:x+w]

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

def train_and_predict_multiclass(anchor_box, suffix): # classes to train
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

	# --------------------------------------------------
	X_test, X_throw_away = construct_dataset(0, 60, '43')
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

	for new in X_test:
		print(svc.predict([new]))
	# --------------------------------------------------

def save_train_data(class_name, X, Y, suffix):
	np.save('data/'+str(class_name)+'/'+str(class_name)+'_X'+suffix, X)
	np.save('data/'+str(class_name)+'/'+str(class_name)+'_Y'+suffix, Y)

# def plot_model(svc, X, Y, anchor_box):
# def plot_model(svc, X, Y):
# 	# classes = anchor_box[anchor_box]
# 	# for my_class in classes:

# 	# for i in len(anchor_box):

# 	# for my_class in anchor_box:
# 	# 	if my_class == 76:
			
# 	# 	elif my_class == 40:
			
# 	# 	elif my_class == 59:
			
# 	# 	else:

# 	color = ['pink' if c == '76' else 'lightgrey' for c in Y]
# 	plt.scatter(X[:,0], X[:,1], c=color)

# 	w = svc.coef_[0]
# 	a = -w[0] / w[1]
# 	xx = np.linspace(-2.5, 2.5)
# 	yy = a * xx - (svc.intercept_[0]) / w[1]

# 	plt.plot(xx, yy)
# 	plt.axis("off"), plt.show()

# def sanity_check(X, Y):
	# --------------------------------
	# print(X_train[9].shape)
	# img = mpimg.imread(data[0])
	# imgplot = plt.imshow(img)
	# plt.show()
	# print(Y_train[9])
	# --------------------------------

# def make_meshgrid(X, Y, h=.02):
# 	x_minimum = X.min() - 1
# 	x_maximum = X.max() + 1
# 	y_minimum = Y.min() - 1
# 	y_maximum = Y.max() + 1
# 	xx, yy = np.meshgrid(np.arange(x_minimum, x_maximum, h),
# 						 np.arange(y_minimum, y_maximum, h))

# 	return xx, yy

# def plot_contours(ax, classifier, xx, yy, **params):
# 	Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
# 	Z = Z.reshape(xx.shape)
# 	out = ax.contourf(xx, yy, Z, **params)
# 	return out

if __name__ == "__main__":
	# ---------------------------------------------------------------------------
	# X, Y = construct_dataset(0, 200, "28")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('umbrella', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "4")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('motorbike', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "1")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('person', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "5")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('aeroplane', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "7")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('train', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "6")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('bus', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "63")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('sofa', X, Y, '200')

	# X, Y = construct_dataset(0, 200, "43")
	# print(X[0])
	# print(X[99])
	# print(X[199])
	# save_train_data('tennis racket', X, Y, '200')

	train_and_predict_multiclass(['brocoli', 'tennis racket', 'hotdog', 'cup'], '200')
	# train_and_predict_multiclass(['apple', 'carrot', 'tvmonitor', 'sofa'], '200')
	# train_and_predict_multiclass(['bicycle', 'orange', 'car', 'cake'], '200')
	# train_and_predict_multiclass(['aeroplane', 'person', 'motorbike', 'umbrella'], '200')
	# train_and_predict_multiclass(['tennis racket', 'sofa', 'bus', 'train'], '200')
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