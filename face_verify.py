import os
import cv2
import numpy as np

from models.arcface_onnx import ArcFaceONNX
from models.retinaface import RetinaFace
from utils import PickableInferenceSession, norm_crop, compute_sim
import csv
import argparse

class FaceVerify():

	def __init__(self,fd_model_Path = "checkpoints/buffalo_sc/det_500m.onnx", fv_model_Path = "checkpoints/buffalo_sc/w600k_mbf.onnx") -> None:

		'''
		fd_model_Path: Onnx file path for face detection (fd)
		fv_model_Path: Onnx file path for face matching (fv)
		'''
		
		session_fd = PickableInferenceSession(fd_model_Path)
		session_fv = PickableInferenceSession(fv_model_Path)
		
		self.fd_model = RetinaFace(model_file=fd_model_Path, session=session_fd)
		self.fv_model = ArcFaceONNX(model_file=fv_model_Path, session=session_fv)

		self.fd_model.prepare(ctx_id=-1)
		self.fv_model.prepare(ctx_id=-1)

	
	def detect_face(self,img):
		bboxes, kpss= self.fd_model.detect(img,max_num=1,input_size=(640,640))
		bbox = bboxes[0].astype(int)
		# dimg = img.copy()
		# cv2.rectangle(dimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
		# cv2.imwrite("check2.png",dimg)
		return bbox,kpss[0]

	def face_embedding(self,img, bbox,kps):
		aimg = norm_crop(img, landmark=kps)
		embed = self.fv_model.get_feat(aimg).flatten()
		return embed
	
	def pair_match(self,img_path1, img_path2,thres=0.3):
		img1 = cv2.imread(img_path1)
		img1 = cv2.fastNlMeansDenoisingColored(img1, None, 3, 3, 7, 21)
		bbox1,kps1=self.detect_face(img1)
		feat1 = self.face_embedding(img1,bbox1,kps1)

		img2 = cv2.imread(img_path2)
		img2 = cv2.fastNlMeansDenoisingColored(img2, None, 3, 3, 7, 21)
		bbox2,kps2=self.detect_face(img2)
		feat2 = self.face_embedding(img2,bbox2,kps2)

		simval = compute_sim(feat1,feat2)

		if simval>thres:
			print("\n*** Faces matched ***\n")
		else:
			print("\n*** Faces did not matched ***\n")
			
		return simval
		

	def draw_on(self, img, faces):
		dimg = img.copy()
		for i in range(len(faces)):
			face = faces[i]
			box = face.bbox.astype(int)
			color = (0, 0, 255)
			cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
			if face.kps is not None:
				kps = face.kps.astype(int)
				#print(landmark.shape)
				for l in range(kps.shape[0]):
					color = (0, 0, 255)
					if l == 0 or l == 3:
						color = (0, 255, 0)
					cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
								2)
			if face.gender is not None and face.age is not None:
				cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

		return dimg


if __name__=="__main__":
	
	parser = argparse.ArgumentParser()

	# train or test for this experiment
	parser.add_argument('-i1', '--image_path1', default='data/Howard-1.jpg', required=False)
	parser.add_argument('-i2', '--image_path2', default='data/Howard-2.jpg', required=False)
	parser.add_argument('-t', '--threshold', default=0.3,type=float)

	args = parser.parse_args()

	facever = FaceVerify()
	simval1 = facever.pair_match(args.image_path1,args.image_path2,thres = args.threshold)

	print("Similarity score : ", simval1)
