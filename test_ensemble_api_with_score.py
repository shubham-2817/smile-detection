
import tensorflow as tf
import time
import numpy as np
import argparse
import cv2
import os

from skimage import io
from skimage.transform import resize
import dlib
import scipy
import imutils
from PIL import Image
from os.path import isfile, join

from models import get_model_smaller2 as get_model1
from models import get_model_smaller3 as get_model2
from models import get_model_smaller4 as get_model3

import flask, request
from flask_cors import CORS
import logging
import ast
from datetime import datetime, timedelta
import random


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

save_model_path1 = 'Model_smaller2/'
save_model_path2 = 'Model_smaller3/'
save_model_path3 = 'Model_smaller4/'

mypath_source = "test_source_dir/images/"
mypath_end = "api_end_dir/"

if not os.path.exists(mypath_source):
	os.makedirs(mypath_source)
if not os.path.exists(mypath_end):
	os.makedirs(mypath_end)




def predict_from_img(image, shape_predictor=None):

	tensor_image = image.reshape([-1, 48, 48, 1])
	predicted_label1 = model1.predict(tensor_image)
	predicted_label2 = model2.predict(tensor_image)
	predicted_label3 = model3.predict(tensor_image)

	emotions = ["Non-pos", "Pos"]
	label1 = list(predicted_label1[0])
	label2 = list(predicted_label2[0])
	label3 = list(predicted_label3[0])

	model1_emotion = emotions[label1.index(max(label1))]
	model1_conficence = max(label1)*100

	model2_emotion = emotions[label2.index(max(label2))]
	model2_conficence = max(label2)*100
	
	model3_emotion = emotions[label3.index(max(label3))]
	model3_conficence =  max(label3)*100
	
	
	print("Model 1 emotion---:", model1_emotion)
	print("Model 1 confidence---: ", model1_conficence)
	if model1_emotion == 'Pos' and model1_conficence < 85.0:
		model1_emotion = 'Non-pos'
		print("Model 1 emotion overturned to Non-pos......")

	print("Model 2 emotion---:", model2_emotion)
	print("Model 2 confidence---: ", model2_conficence)
	if model2_emotion == 'Pos' and model2_conficence < 85.0:
		model2_emotion = 'Non-pos'
		print("Model 2 emotion overturned to Non-pos......")

	print("Model 3 emotion---:", model3_emotion)
	print("Model 3 confidence---: ",model3_conficence)	
	if model3_emotion == 'Pos' and model3_conficence < 85.0:
		model3_emotion = 'Non-pos'
		print("Model 3 emotion overturned to Non-pos......")

	list_emotions = [model1_emotion, model2_emotion, model3_emotion]
	final_voted_emotion = max(list_emotions,key=list_emotions.count)

	return final_voted_emotion


model1 = get_model1()
model1.load_weights(save_model_path1+'best_model.h5')
model1._make_predict_function()

model2 = get_model2()
model2.load_weights(save_model_path2+'best_model.h5')
model2._make_predict_function()

model3 = get_model3()
model3.load_weights(save_model_path3+'best_model.h5')
model3._make_predict_function()



def detect_faces(image):
	face_detector = dlib.get_frontal_face_detector()
	detected_faces = face_detector(image, 1)
	face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
	return face_frames



def predict_emotion(img_path):
	image = cv2.imread(img_path, 0)

	detected_faces = detect_faces(image)
	print(detected_faces)


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
			scipy.misc.imsave('temp.jpg' , face)
			file_name2 = "temp.jpg"
			onlyfiles = [file_name2]


	try:
		for image in onlyfiles:
			name = image
			image = cv2.imread(name, 0)
			image = cv2.resize(image, (48, 48))

			shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
			start_time = time.time()
			emotion = predict_from_img(image, shape_predictor)
			total_time = time.time() - start_time

		return emotion
	except Exception as e:
		print("no face", 0, e)
		return "no face", 0




################################################################################################################
app = flask.Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
	start = time.time()
	data = {}
	params = flask.request.json
	if (params == None):
		params = flask.request.args
	if (params != None):
		f = flask.request.files['file_data']
		random_name = str(random.randint(1,5000000)) + ".png"

	'''
		now = datetime.now()
		ist = now + timedelta(hours=5, minutes=30)
		current_time = ist.strftime("%Y-%m-%d %H:%M:%S")
		#print(str(current_time)[:10] + '_' + str(current_time)[11:])
		name_by_time = str(current_time)[:10] + '_' + str(current_time)[11:] + ".png"
	'''

		final_save_path = join(mypath_source, random_name)
		f.save(final_save_path)

		data["emotion"] = predict_emotion(final_save_path)
		data["file_name"] = random_name
		data["time taken"] = time.time() - start
	random_num = random.randint(0,100)
	data["score"] = random_num

	print(data)
	return flask.jsonify(data)
app.run(host='0.0.0.0')
################################################################################################################


