
import os 
import csv
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from face_verify import FaceVerify
from face_verify_tflite import tflite_face_verification

def prediction(csv_file = "data/LFW/pairs_train.csv"):
	
	file_reader = csv.reader(csv_file)
	root = "data/LFW/lfw-deepfunnel"
	FV = FaceVerify()

	f = open("LFW_Results_tflite.csv", 'w')	
	outfile = csv.writer(f)
	with open(csv_file, newline='\n') as csvfile:
		lines = len(csvfile.readlines())

	with open(csv_file, newline='\n') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		# lines = len(csvfile.readlines())
		i=0
		for row in tqdm(reader,total=lines-1):
			i+=1
			if i>10:
				break

			img_path1 = os.path.join(root,row[0],row[0]+"_"+row[1].zfill(4)+".jpg")
			img_path2 = os.path.join(root,row[2],row[2]+"_"+row[3].zfill(4)+".jpg")
			gt = row[-1]
			try:
				# match_val = FV.pair_match(img_path1,img_path2)
				match_val = tflite_face_verification(img_path1,img_path2)
				row_out = [img_path1, img_path2 , str(match_val), gt]
				outfile.writerow(row_out)
			except:
				print("problem : "+ str(i+1))
				continue
			
	f.close()
	
	print(match_val)


def calculate_perf(csv_file = "LFW_Results.csv"):

	with open(csv_file, newline='\n') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		output =[]
		gt=[]
		for row in reader:
			output.append(float(row[2]))
			gt.append(int(row[3]))

	print("end")

	fpr, tpr, threshold = metrics.roc_curve(gt,output,pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	print(eer_threshold)
	eer_threshold = 0.3
	y_pred = []
	for v in output:
		if v> eer_threshold:
			y_pred.append(1)
		else:
			y_pred.append(0)
	matrix_confusion = metrics.confusion_matrix(gt, y_pred)
	f1_sc = metrics.f1_score(gt,y_pred)
	print("F1 Score",f1_sc)
	print(matrix_confusion)

if __name__=="__main__":
	prediction()