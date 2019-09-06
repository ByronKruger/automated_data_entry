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
os.urandom(55)
# def compose_dataset_file():

anchor_boxes = {
	'1:2': ['bottle', 'vase', 'pottted_plant', 'refrigerator'],
	'2:1': ['microwave', 'tvmonitor', 'sofa', 'bed'],
	'1:1': ['cup', 'oven', 'toilet', 'clock'],
}

switcher = {
	'carrot':'57',
	'chair':'62',
	'cup':'47',
	'tvmonitor':'72',
	'surfboard':'42',
	'hotdog':'58',
	'tennis racket':'43',
	'brocoli':'56',
	'cup':'47',
}

def construct_dataset(lower, upper, train_class, clss_fn):
	test_no_dup = 0
	train_class = int(train_class)
	print(train_class)
	X_train = np.zeros((upper-lower, 20000))  # make it dynamic size, make it 3 channels if found to increase accuracy else this will be faster
	Y_train = np.zeros((upper-lower))
	X_train_count = 0

	# with open('./data/'+str(lower)+str(upper)+".txt") as file:
	# fname = "./data/images_annos_0_4999_subset.txt"
	# fname = "./data/images_annos_0_14999.txt"
	# fname = "./data/images_annos_0_2999_toilet.txt"
	fname = clss_fn

	# with open("./data/images_annos_0_9999.txt") as file:

	x_ar = [] 
	y_ar = []
	w_ar = []
	h_ar = []
	cls_ar = []
	backgrd_amount = 0

	dim_x = 0
	dim_y = 0

	with open(fname) as file:
		line = file.readline()
		while line and X_train_count < (upper-lower):
			data = line.split()

			if len(data) == 1:
				# if len(x_ar) != 0:
				if backgrd_amount != 0:
					# backgrd_amount = len(x_ar)
					print("backgrd_amount")
					print(backgrd_amount) # sae amount as amount of desired class

					# boxes_o_any = []
					# four_attrs = zip(x_ar, y_ar, w_ar, h_ar)
					# for attr in four_attrs:

					# for i in range(backgrd_amount):
					# 	# reconstruct all boxes of all class in current image
					# 	b_x = random.randint(0, int(dim_x))
					# 	b_y = random.randint(0, int(dim_y))
						
					for i in range(backgrd_amount):
						# print("in for finding background boxes")
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
				except Exception as e:
					print(str(e))
				
				x_ar = [] 
				y_ar = []
				w_ar = []
				h_ar = []

				backgrd_amount = 0
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
				x_objects = cv2.imread(filename) # image with possible multiple objects.

				dim_x = x_objects.shape[1]
				dim_y = x_objects.shape[0]

				x_objects = cv2.cvtColor(x_objects, cv2.COLOR_BGR2RGB)
				
				x_ar.append(x)
				y_ar.append(y)
				w_ar.append(w)
				h_ar.append(h)				

				if clas == train_class:
					backgrd_amount += 1
					# x_ar.append(x)
					# y_ar.append(y)
					# w_ar.append(w)
					# h_ar.append(h)

					if test_no_dup < 5:
						img = Image.fromarray(x_objects)
						# img.show() #
						x_object = x_objects[y:y+h, x:x+w]
						img = Image.fromarray(x_object)
						# img.show() #
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

def construct_dataset_no_back(lower, upper, train_class, clss_fn):
	test_no_dup = 0
	train_class = int(train_class)
	print(train_class)
	X_train = np.zeros((upper-lower, 20000))  # make it dynamic size, make it 3 channels if found to increase accuracy else this will be faster
	Y_train = np.zeros((upper-lower))
	X_train_count = 0

	# with open('./data/'+str(lower)+str(upper)+".txt") as file:
	# fname = "./data/images_annos_0_4999_subset.txt"
	# fname = "./data/images_annos_0_2999_chair.txt"
	fname = clss_fn

	# with open("./data/images_annos_0_9999.txt") as file:

	with open(fname) as file:
		line = file.readline()
		while line and X_train_count < (upper-lower):
			data = line.split()

			if len(data) == 1:
				filename = data[0]
				print("__________________________")
				print(filename)
				print("__________________________")
				
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
				x_objects = cv2.imread(filename) # image with possible multiple objects.
				x_objects = cv2.cvtColor(x_objects, cv2.COLOR_BGR2RGB)
				
				if clas == train_class:
					if test_no_dup < 51:
						img = Image.fromarray(x_objects)
						# img.show()
						x_object = x_objects[y:y+h, x:x+w]
						img = Image.fromarray(x_object)
						# img.show()
						test_no_dup += 1

					x_object = x_objects[y:y+h, x:x+w]

					x_object = cv2.cvtColor(x_object, cv2.COLOR_BGR2GRAY) # temporary
					x_object = cv2.resize(x_object, (100, 200))
					x_object = np.float32(x_object)/255
					x_object = cv2.Sobel(x_object, cv2.CV_32F, 0, 1, ksize=1) # temporary
					x_object = x_object.flatten()
					
					X_train[X_train_count] = x_object
					Y_train[X_train_count] = train_class
					X_train_count += 1

			line = file.readline()
	print("no_back, len(X_train):"+str(len(X_train)))
	return X_train, Y_train

def is_box_in_box(x_ar, y_ar, w_ar, h_ar, x, y, w, h):
	c = 0
	for xx,yy,ww,hh in zip(x_ar,y_ar,w_ar,h_ar):
		# print(c)
		if (x >= xx) and (x <= xx+ww) and (y >= yy) and (y <= yy+hh):
			c+=1
			return True

		if (x+w >= xx) and (x+w <= xx+ww) and (y+h >= ww) and (y+h <= yy+hh):
			c+=1
			return True
			
		return False

def construct_dataset_bckgrd(lower, upper, bb_w, bb_h, clss_fn):
	sanity_check = 0
	sanity_check1 = 0

	X_train = np.zeros((upper-lower, 20000))  # make it dynamic size, make it 3 channels if found to increase accuracy else this will be faster
	Y_train = np.zeros((upper-lower))

	X_train_count = 0

	# fname = "./data/images_annos_0_14999.txt"
	fname = clss_fn

	x_ar = [] 
	y_ar = []
	w_ar = []
	h_ar = []		
	x_dim = 0			
	y_dim = 0		
	filename = ''
	c = 0

	with open(fname) as file:
		line = file.readline()
		while line and X_train_count < (upper-lower):
			data = line.split()

			if len(data) == 1:
				if len(x_ar) > 0 and len(y_ar) > 0 and len(w_ar) > 0 and len(h_ar) > 0:
					# x = 0
					# y = 0
					w = bb_w
					h = bb_h
					# xywhs = zip(x_ar, y_ar, w_ar, h_ar)
					# for xywh in xywhs: # just ot match the same amount of objects with the amount of background samples
					# while x in x_ar or y in y_ar: # just to make sure we don't cut backgrounds as objects (still not robust enuf)
					for _ in x_ar:
						while True:  
							x = random.randint(0, x_dim)
							y = random.randint(0, y_dim)
							if is_box_in_box(x_ar, y_ar, w_ar, h_ar, x, y, w, h):  
								break;
						# while is_box_in_box(x_ar, y_ar, w_ar, h_ar, x, y, w, h): # just to make sure we don't cut backgrounds as objects (still not robust enuf)
							# x = random.randint(0, x_dim)
							# y = random.randint(0, y_dim)
						
						if sanity_check1 < 5:
								print("x:"+str(x))
								print("y:"+str(y))
								sanity_check1+=1
							# w = bb_w
							# h = bb_h

						# if c < 1:
						if sanity_check < 10: # display purposes only
							x_objects = cv2.imread(filename)
							x_objects = cv2.cvtColor(x_objects, cv2.COLOR_BGR2RGB) # temporary
							img = Image.fromarray(x_objects)
							# img.show()
							bckgrd = x_objects[y: y+h, x: x+w]
							img = Image.fromarray(bckgrd)
							# img.show()
							sanity_check += 1

						bckgrd = x_objects[y: y+h, x: x+w]
						bckgrd = cv2.cvtColor(bckgrd, cv2.COLOR_BGR2GRAY) # temporary
						bckgrd = cv2.resize(bckgrd, (100, 200))
						bckgrd = np.float32(bckgrd)/255
						bckgrd = cv2.Sobel(bckgrd, cv2.CV_32F, 0, 1, ksize=1) # temporary
						bckgrd = bckgrd.flatten()
							
						if X_train_count < (upper-lower):
							X_train[X_train_count] = bckgrd
							Y_train[X_train_count] = '0' # background
							X_train_count += 1
							# print(X_train_count)

				# -------------------- reset all these vars ----------------------
				c = 0
				x_ar = [] 
				y_ar = []
				w_ar = []
				h_ar = []

				filename = data[0]
				# -------------------- reset all these vars ----------------------

			else:
				x = data[0]
				x = math.floor(float(x))

				y = data[1]
				y = math.floor(float(y))

				w = data[2]
				w = math.floor(float(w))
				
				h = data[3]
				h = math.floor(float(h))

				x_ar.append(x)
				y_ar.append(y)
				w_ar.append(w)
				h_ar.append(h)	

				clas = data[5]
				clas = int(clas)

				x_objects = cv2.imread(filename) # image with possible multiple objects.
				x_objects = cv2.cvtColor(x_objects, cv2.COLOR_BGR2RGB)
				x_dim = x_objects.shape[1]
				y_dim = x_objects.shape[0]

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

def save_train_data(class_name, X, Y, suffix):
	np.save('data/'+str(class_name)+'/'+str(class_name)+'_X_'+suffix, X)
	np.save('data/'+str(class_name)+'/'+str(class_name)+'_Y_'+suffix, Y)

def train_and_predict_multiclass(anchor_box, suffix, clss_name, clss_fn, hog_windows=None): # classes to train
	# classes = anchor_box[anchor_box]

	X_data = []
	Y_data = []

	for my_class in anchor_box:
		a_X_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_X_'+suffix+'.npy')
		X_data.append(a_X_data)
		a_X_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_X_'+suffix+'_back.npy')
		X_data.append(a_X_data)
		a_Y_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_Y_'+suffix+'.npy')
		Y_data.append(a_Y_data)
		a_Y_data = np.load('data/'+str(my_class)+'/'+str(my_class)+'_Y_'+suffix+'_back.npy')
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

		i = 1
		for new in X_test:
			print("no. "+str(i)+": "+str(int(svc.predict([new])[0])))
			i+=1
			preds.append(int(svc.predict([new])[0]))

		return preds

	else:
		X_test = []
		amt = 50
		switched = switch_name_to_num(clss_name)
		X_objs, _ = construct_dataset_no_back(0, amt, switched, clss_fn)
		X_test.append(X_objs)
		X_backs, _ = construct_dataset_bckgrd(0, amt, 50, 50, clss_fn)
		X_test.append(X_backs)

		X_test_objs_back = np.concatenate(([a_X_data for a_X_data in X_test]))

		# print("X_test[]:")
		# print(len(X_test))

		i = 1
		for test in X_test_objs_back:
			# print("test[0]")
			# print(test[0])
			print("no. "+str(i)+": "+str(int(svc.predict([test]))))
			# print("\n")
			i+=1
			preds.append(int(svc.predict([test])[0]))
			gts.append(int(switched))

		print(classification_report(preds, gts))

def switch_name_to_num(name):
	return switcher.get(name, 'invalid_name')

def create_xy_and_save(amt, fn, clss_name):
	switched = switch_name_to_num(clss_name)
	X, Y = construct_dataset_no_back(0, amt, switched, fn)
	save_train_data(clss_name, X, Y, str(amt))
	X, Y = construct_dataset_bckgrd(0, amt, 50, 50, fn)
	save_train_data(clss_name, X, Y, str(amt)+'_back')

def create_xy():
	pass

if __name__ == "__main__":

	create_xy_and_save(100, 'data/images_annos_0_299_brocoli.txt', 'brocoli')
	create_xy_and_save(100, 'data/images_annos_0_299_tennis racket.txt', 'tennis racket')
	create_xy_and_save(100, 'data/images_annos_0_299_hotdog.txt', 'hotdog')
	create_xy_and_save(100, 'data/images_annos_0_299_cup.txt', 'cup')

	train_and_predict_multiclass(["cup", "hotdog", "tennis racket", "brocoli"], "100", 'tennis racket', "data/images_annos_0_49_tennis racket.txt") # trains on 300, tests with 50