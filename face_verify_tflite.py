import tensorflow as tf
import onnx
import numpy as np
import cv2
from face_verify import FaceVerify
from utils import norm_crop, compute_sim

def tflite_face_verification(img_path1,img_path2):
	TFLITE_QUANT_MODEL = "checkpoints/buffalo_sc/w600k_mbf.tflite"

	interpreter = tf.lite.Interpreter(model_path=TFLITE_QUANT_MODEL)

	# Learn about its input and output details
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# print("== Input details ==")
	# print("name:", input_details[0]['name'])
	# print("shape:", input_details[0]['shape'])
	# print("type:", input_details[0]['dtype'])

	# print("\n== Output details ==")
	# print("name:", output_details[0]['name'])
	# print("shape:", output_details[0]['shape'])
	# print("type:", output_details[0]['dtype'])

	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# input_shape = np.array([1,3,112,112])
	input_data1 = face_detection(img_path1)
	interpreter.set_tensor(input_details[0]['index'], input_data1)
	interpreter.invoke()
	output_data1 = interpreter.get_tensor(output_details[0]['index'])
	# print(output_data1.shape)

	input_data2 = face_detection(img_path2)
	interpreter.set_tensor(input_details[0]['index'], input_data2)
	interpreter.invoke()
	output_data2 = interpreter.get_tensor(output_details[0]['index'])
	# print(output_data2.shape)

	sim_val = compute_sim(output_data1,output_data2)
	print(sim_val)
	return sim_val


FV = FaceVerify()
def face_detection(img_path):
	img = cv2.imread(img_path)
	img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
	bbox,kps=FV.detect_face(img)
	aimg = norm_crop(img, landmark=kps)
	aimg = [aimg]
	input_mean = 127.5
	input_std = 127.5
	blob = cv2.dnn.blobFromImages(aimg, 1.0 / input_std, (112,112),
										(input_mean, input_mean, input_mean), swapRB=True)
	# aimg = aimg.transpose(2,0,1)
	# aimg = np.expand_dims(aimg,axis=0)
	# aimg = aimg.astype(np.float32)

	
	return blob


if __name__=="__main__":

	tflite_face_verification(img_path1="data/Howard-1.jpg",img_path2="data/Howard-2.jpg")
