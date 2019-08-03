import tensorflow as tf
from tflearn import DNN
import time
import numpy as np
import argparse
import dlib
import cv2
import os
from skimage.feature import hog

from skimage import io
from skimage.transform import resize
import dlib
import scipy
import imutils
from PIL import Image
from os.path import isfile, join
from os import *
import pandas as pd


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def detect_faces(image):
	face_detector = dlib.get_frontal_face_detector()
	detected_faces = face_detector(image, 1)
	face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
	return face_frames



def preprocess_images():

	image_files = [f for f in listdir(mypath_source) if isfile(join(mypath_source, f))]

	image_files = image_files[588:]

	for img_path in image_files:
		name = img_path.split('.')[0] 
		image = cv2.imread(join(mypath_source, img_path), 0)

		detected_faces = detect_faces(image)
		print(len(detected_faces))

		if len(detected_faces) == 0:
			print("***"*5,"Normal: ",len(detected_faces))
			image = imutils.rotate_bound(image, 90)
			detected_faces = detect_faces(image)

		if len(detected_faces) == 0:
			print("***"*5,"90 degrees: ",len(detected_faces))
			image = imutils.rotate_bound(image, 90)
			detected_faces = detect_faces(image)

		if len(detected_faces) == 0:
			print("***"*5,"180 degrees: ",len(detected_faces))
			image = imutils.rotate_bound(image, 90)
			detected_faces = detect_faces(image)
			print("***"*5,"270 degrees: ",len(detected_faces))

		if len(detected_faces) > 0:
			for n, face_rect in enumerate(detected_faces):
				face = Image.fromarray(image).crop(face_rect)
				scipy.misc.imsave(processed_test_images_path + name + '_' + str(n) + '.jpg' , face)




def predict_from_img(image, shape_predictor=None):

	tensor_image = image.reshape([-1, 48, 48, 1])
#	print(tensor_image.shape)

	predicted_label1 = model1.predict(tensor_image)
	predicted_label2 = model2.predict(tensor_image)
	predicted_label3 = model3.predict(tensor_image)

	emotions = ["Non", "Pos"]
	label1 = list(predicted_label1[0])
	label2 = list(predicted_label2[0])
	label3 = list(predicted_label3[0])

	model1_emotion = emotions[label1.index(max(label1))]
	model1_conficence = max(label1)*100

	model2_emotion = emotions[label2.index(max(label2))]
	model2_conficence = max(label2)*100
	
	model3_emotion = emotions[label3.index(max(label3))]
	model3_conficence =  max(label3)*100

	list_emotions = [model1_emotion, model2_emotion, model3_emotion]
	list_confidence = [model1_conficence, model2_conficence, model3_conficence]

	return list_emotions, np.array(list_confidence)



def predict_emotion():

	image_files = [f for f in listdir(processed_test_images_path) if isfile(join(processed_test_images_path, f))]
	image_files = sorted(image_files)
	try:
		data = []
		for image in image_files[:]:
			print(image)
			name = image
			image = cv2.imread(join(processed_test_images_path,name), 0)
			image = cv2.resize(image, (48, 48))

			emotion_list, confidence_list = predict_from_img(image)

			row = [name]
			for i in range(len(model_names)):
				row.append(str(emotion_list[i]) + "( " +str(confidence_list[i])[:5] + ")")
				data.append(row)

		columns = ['name']
		for name in model_names:
			columns.append(name)

			results_df = pd.DataFrame(data)
			results_df.columns = columns 
			results_df.to_csv("model_results_df_old_comb.csv", index=False)
		
	except Exception as e:
		print("no face", 0, e)
		# return "no face", 0




#mypath_source = "test_source_dir/images/"
mypath_source = "/mnt/data/shubham/new_model/facial-expression-recognition-using-cnn/test_data_images2/neg/"
mypath_end = "api_end_dir/"
processed_test_images_path = "processed_test_images_path/" 

if not os.path.exists(mypath_source):
	os.makedirs(mypath_source)
if not os.path.exists(mypath_end):
	os.makedirs(mypath_end)
if not os.path.exists(processed_test_images_path):
	os.makedirs(processed_test_images_path)



from models import get_model_smaller2 as get_model1
from models import get_model_smaller3 as get_model2
from models import get_model_smaller4 as get_model3


save_model_path1 = 'Model_smaller2/' + 'best_model.h5'
save_model_path2 = 'Model_smaller3/' + 'best_model.h5'
save_model_path3 = 'Model_smaller4/' + 'best_model.h5'

model_names = [save_model_path1, save_model_path2, save_model_path3]

model1 = get_model1()
model1.load_weights(save_model_path1)
model1._make_predict_function()

model2 = get_model2()
model2.load_weights(save_model_path2)
model2._make_predict_function()

model3 = get_model3()
model3.load_weights(save_model_path3)
model3._make_predict_function()


# preprocess_images()
predict_emotion()

