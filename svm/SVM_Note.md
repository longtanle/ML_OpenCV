# SVM Example: Automatic Object Inspection Classification

The example of object detection and classification of three object types: a screw, a packing ring, and a nut.

## Process Stages
	
1. Pre-process
	- Noise removal
	- Lighting removal
	- Binarization

2. Segmentation
	- Contour detection
	- Connected component extraction (Labeling)

3. Feature Extracion: 
	- The area of an object
	- The aspectio ratio, which is the width divided by the heigt of the bouding rectangle
	- The number of holes
	- THe number of contour sides

4. ML Classification (SVM)
	- Training SVM model: require images of each object and their corresponding labels
		+ Find datasets in "data" folder (screw, nut, & ring)
		+ Image in .pgm (Portable graymap format)
	- Prediction: Load input images

5. Post-process

## Procedure:

1. For training each image:
	- Preprocess an image
	- Segment an image

2. For each object in an image:
	- Extract the features
	- Add the object to the training feature vector with its label

3. Create an SVM model
4. Train the SVM model with the training feature vector
5. Preprocess an input image to be classified
6. Segment an input image
7. For each object detected:
	- Extract the features
	- Predict with an SVM model
	- Pain the result in an output image
