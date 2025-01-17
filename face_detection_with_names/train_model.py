from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# loading face embeddings
print('Loading face embeddings...')
data = pickle.loads(open(args['embeddings'], 'rb').read())

# encode the labels
print('Encoding labels...')
le = LabelEncoder()
labels = le.fit_transform(data['names'])

# train the model and produce actual face recognition
print('Traning Model...')
recognizer = SVC(C=1.0, kernel='linear', probability=True)
recognizer.fit(data['embeddings'], labels)

# write the actual face recognition model to disk
f = open(args['recognizer'], 'wb')
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args['le'], 'wb')
f.write(pickle.dumps(le))
f.close()
