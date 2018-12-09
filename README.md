# Face Detection

### Usage
1. You can make your custom dataset in the ```dataset``` folder.
2. Similarly you can add test images in the ```images``` folder.
3. Running this Script: 
	- First you have to extract embeddings of the images using the ```extract_embeddings.py``` file

	```python
	python extract_embeddings.py -i dataset -e output/embeddings.pickle \
	-d face_detection_model -m openface_nn4.small2.v1.t7
	```

	- Next you have to train your model using ```train_model.py``` file

	```python
	python train_model.py -e output/embeddings.pickle -r output/recognizer.pickle -l output/le.pickle
	```

	- At the end if you wish to do face detection of static images use the ```recognize.py``` file

	```python
	python recognize.py -d face_detection_model -m openface_nn4.small2.v1.t7 \
	-r output/recognizer.pickle -l output/le.pickle -i images/img1.jpg
	```

	- If you wish to detect faces live using the video camera run the ```recognize_video.py``` file

	```python
	python recognize_video.py -d face_detection_model -m openface_nn4.small2.v1.t7 \
	-r output/recognizer.pickle \
	```
