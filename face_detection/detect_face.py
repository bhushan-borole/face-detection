import numpy as np
import argparse
import cv2

def start_detecting_faces(image, args, detections, dim):
	(h, w) = dim
	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
	 
		# filter out weak detections
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	 
	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)

def load_model_and_compute_detections(args):
	# laoding caffe model
	net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt',
							'models/res10_300x300_ssd_iter_140000.caffemodel')
	print('Loading Model...')

	# read image and create blob from that image
	image = cv2.imread(args['image'])
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
								(300, 300), (104.0, 177.0, 123.0))
	#obtain detections and predictions
	net.setInput(blob)
	detections = net.forward()
	print('Computing object detections...')

	return image, detections, (h, w)

def get_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image',required=True, 
				help='path to input image')
	ap.add_argument('-c', '--confidence', type=float, default=0.5,
				help='minimum probability to filter weak detections')
	return vars(ap.parse_args())

def main():
	args = get_arguments()
	image, detections, (h, w) = load_model_and_compute_detections(args)
	start_detecting_faces(image, args, detections, (h, w))

if __name__ == '__main__':
	main()