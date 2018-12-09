from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# costructing argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

'''
detector: We’re using a Caffe based DL face detector to localize faces in an image.
embedder: This model is Torch-based and is responsible for extracting facial embeddings via deep learning feature extraction.
'''
# loading face detector
print('Loading Face Detector...')
prototype_path = os.path.join(args['detector'], 'deploy.prototxt')
model_path = os.path.join(args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel')
detector = cv2.dnn.readNetFromCaffe(prototype_path, model_path)

# loading face embedding model
print('Loading Face recognizer...')
embedder = cv2.dnn.readNetFromTorch(args['embedding_model'])

# grabbing paths to the input image in our dataset
print('Reading faces...')
image_paths = list(paths.list_images(args['dataset']))

known_embeddings = []
known_names = []

total = 0 # track of how many faces processed

# extracting embeddings from faces found in each image
for i, image_path in enumerate(image_paths):
	# extracting person's name
	print('Processing Image {}/{}'.format(i + 1, len(image_paths)))
	name = image_path.split(os.path.sep)[-2]

	image = cv2.imread(image_path)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	image_blob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# applying opencv's deep-learning face detector
	detector.setInput(image_blob)
	detections = detector.forward()

	# ensure atleast one face was found
	if len(detections) > 0:
		# we're assuming that each image has only one face
		# so find the bounding box with largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]


		# ensuring detection with largest probability
		# thus filtering out weak detections

		if confidence > args['confidence']:
			# computing (x,y) coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype('int')

			# extract face ROI and grab ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are suficiently large
			if fW < 20 or fH < 20:
				continue


			'''
			construct a blob for the face ROI, then pass the blob
			through our face embedding model to obtain the 128-d
			qunatification of the face
			'''
			face_blob = cv2.dnn.blobFromImage(face, 1.0/255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(face_blob)
			vec = embedder.forward()

			'''
			add the name of the person + corresponding face
			embedding to their respective lists
			'''
			known_names.append(name)
			known_embeddings.append(vec.flatten())
			total += 1

# dumping facial embeddings and names
print("Serializing {} encodings...".format(total))
data = {
	"embeddings" : known_embeddings,
	"names" : known_names
}
f = open(args['embeddings'], 'wb')
f.write(pickle.dumps(data))
f.close()