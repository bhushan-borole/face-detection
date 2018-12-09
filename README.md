# Face Detection

### Usage
1. You can make your custom dataset in the ```dataset``` folder.
2. Similarly you can add test images in the ```images``` folder.
3. Running this Script: 
	- First you have to extract embeddings of the images using the ```extract_embeddings.py``` file
```python
python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7
```
