# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the len() and .shape functions to find out the summary statistics of the traffic signs data set:

* The size of training set -> len(X_train)
* The size of the validation set -> len(X_valid)
* The size of test set is -> len(X_test)
* The shape of a traffic sign image -> X_train[0].shape
* The number of unique classes/labels in the data set is retrieved based on the number of rows in the 'signnames.csv' file. The code:
`import csv

all_labels = []

with open('signnames.csv', 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        all_labels += [row[1]]
        
all_labels.remove('SignName') `


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the distribution of the data set. Most of the data belongs to the first 20 classes, which shows that the data set is biased.

![Dist_dataset][./hist.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. Personally, I think there is no specific reason not to use the color image. But since the traffic sign we dealing with here are either black, red or blue, the color has no significance in this classifier, then I decided to go for grey scale images to avoid false classification and complexities. It shrinks down the depth of the data from 3 to 1, which decrease the amount of information for the CNN to process. Then I normalized the image data to avoid inconsistency. Since the CNN corrections are based on the multiplication of learning rate and feature values, it is essential to keep the feature values in the same scale by using `(data - 128)/128`.

Here is an example of a traffic sign image after grayscaling and normalizing.

![Image_preprocess][./img_visual_gray.png]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAY image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5   	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 					|
| Fully connected 01	| Input = 1600. Output = 480					|
| Drop Out 			 	| Keep Prob = 0.6								|
| Fully connected 02	| Input = 480. Output = 120						|
| Drop Out 			 	| Keep Prob = 0.6								|
| Fully connected 03	| Input = 120. Output = 43						|
| Softmax				| etc.        									|
| Cross Entropy			| etc.        									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a training set with 20 epochs and 200 as batch size. Learning rate 0.0005, and a keep probability of 70%. The cross-entropy results are then fed into an Adam optimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 93.4%
* test set accuracy of 92.6%

An iterative approach was chosen:
* I started with the LeNet model provided by Udacity. This model can successfully train the recognition of the character. I think the structure should be a good fit for this project. Plus, the convolution layers should work very well for pattern recognition, which is the primary objective of the traffic sign classification.
* With gray pictures as input, with a training setup of 20 epochs, 200 as batch size and 0.001 learning rate, the results could not exceed 90% accuracy. 
* To improve that, I increased the number of epochs to 25 and decreased the learning rate to 0.0005. The results can reach 90% but still not good enough.
* I then made the first two convolution layer deeper, which also leads to a higher volume of the fully-connected layers. The results are now capable of fluctuating around 93% accuracy and 99% for the training set accuracy(indicating over-fitting). 
* To make the results more consistent, I then added two dropout layers to prevent overfitting, with 0.6 keep probability the results can maintain at a steady state of 94% accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Test_img_01][./Test_img/general_cuation.jpg]
![Test_img_01][./Test_img/keep_right.jpg] 
![Test_img_01][./Test_img/no_entry.jpg] 
![Test_img_01][./Test_img/speed_30_limits.jpg] 
![Test_img_01][./Test_img/turn_right.jpg] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed_30_limits      	| Speed limit (30km/h)							| 
| Keep_right   			| Keep right									|
| General_cuation 		| General caution								|
| Turn_right     		| Turn right ahead				 				|
| No_entry				| No entry     									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. I think this is a reasonable answer since the accuracy on the test set is already 92% plus. 
The `Keep_right` and `Turn_right` image got a poor prediction at the beginning when the hyperparameters are not adjusted yet. The reason could be:
    1. Lack of information -> The `Keep_right` sign in the image is not a perfect circle, and the `Turn_right` image is missing a small bottom part of the sign.
    2. Since the training set is biased and the high volume of CNN layer/depth may introduce over-fitting, this is why I used Dropout layer later in the modification.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16-18th cell of the Ipython notebook.
I used a top 3 prediction soft max probabilities, and the result is showing in the image below:

![Top_3_Prediction][./Top3Prediction.png]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


