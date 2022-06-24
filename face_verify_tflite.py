import tensorflow as tf
import onnx
import numpy as np
import cv2
from face_verify import FaceVerify
from utils import norm_crop, compute_sim
from models.retinaface import FaceDetectProcessing


class FaceVerifyTflite():
	def __init__(self, fd_model_path='checkpoints/buffalo_sc/det_500m.tflite', fv_model_path='checkpoints/buffalo_sc/w600k_mbf.tflite'):

		self.fd_model = tf.lite.Interpreter(model_path=fd_model_path)
		self.fd_input_details = self.fd_model.get_input_details()
		self.fd_output_details = self.fd_model.get_output_details()
		self.fd_model.resize_tensor_input(self.fd_input_details[0]['index'], (1, 3, 640, 640))
		self.fd_model.allocate_tensors()


		self.fv_model = tf.lite.Interpreter(model_path=fv_model_path)
		self.fv_input_details = self.fv_model.get_input_details()
		self.fv_output_details = self.fv_model.get_output_details()
		self.fv_model.allocate_tensors()

	def face_verification(self,img_path1,img_path2):
		

		input_data1 = self.face_detection(img_path1)
		input_data2 = self.face_detection(img_path2)

		if input_data1 is None or input_data2 is None:

			print("Please provide single face Image")
			return None

		self.fv_model.set_tensor(self.fv_input_details[0]['index'], input_data1)
		self.fv_model.invoke()
		output_feat1 = self.fv_model.get_tensor(self.fv_output_details[0]['index'])
		# print(output_data1.shape)

		self.fv_model.set_tensor(self.fv_input_details[0]['index'], input_data2)
		self.fv_model.invoke()
		output_feat2 = self.fv_model.get_tensor(self.fv_output_details[0]['index'])

		sim_val = compute_sim(output_feat1,output_feat2)
		print(sim_val)
		return sim_val


	def face_detection(self,img_path):

		# LOAD AND PREPROCESS IMAGE DATA
		img = cv2.imread(img_path)
		img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
		img_processing = FaceDetectProcessing()
		blob = img_processing.preprocessing(img,input_size=(640,640))
		input_data = blob.astype(np.float32)

		# MAKE PREDICTION FOR FACE DETECTION
		self.fd_model.set_tensor(self.fd_input_details[0]['index'], input_data)
		self.fd_model.invoke()
		net_outs = []
		for i in range(3):
			net_outs.append(self.fd_model.get_tensor(self.fd_output_details[i]['index']))
			net_outs.append(self.fd_model.get_tensor(self.fd_output_details[i+3]['index']))
			net_outs.append(self.fd_model.get_tensor(self.fd_output_details[i+6]['index']))

		bboxes,kpss = img_processing.post_processing(img,blob,net_outs,max_num=2)
		# print(kpss[0])
		if kpss.shape[0] >1 :
			print("Multiple faces Detected")
			return None

		# Preprocess data for face verification input
		aimg = norm_crop(img, landmark=kpss[0])
		aimg = [aimg]
		input_mean = 127.5
		input_std = 127.5
		blob_fv = cv2.dnn.blobFromImages(aimg, 1.0 / input_std, (112,112),
										(input_mean, input_mean, input_mean), swapRB=True)
		
		# cv2.imwrite("text.png",aimg[0])

		return blob_fv

if __name__=="__main__":

	FV_instance = FaceVerifyTflite()
	FV_instance.face_verification(img_path1="data/Howard-2.jpg",img_path2="data/Howard-2.jpg")
