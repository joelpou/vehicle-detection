## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_window.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./output_images/heat_map1.png
[video1]: ./test_video_out-primitive.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images on the second cell of the `carTraining.ipynb` jupyter notebook after importing all the libraries needed for the pipeline.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

`pipelineFunctions.py`, contains all the helper functions used to run this pipeline, including the `extract_features()` and `single_img_features()` functions. The former helps me extract car and non-car features to train my SVC. The latter is a similar implementation of the other one but for single images. It will be used for  For these I'm using Udacity's Vehicle Tracking datasets. HOG parameters chosen were `orientations of 9`, `pixels per cell of 8`, `cells per block of 2` and `ALL channels`. On the eighth cell I added some code to visualize HOG feature extraction for car and non-car datasets like so:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

At first I tried various combinations of parameters using a combination of spatial binning with color histogram and hog feature extraction. I mostly used a colorspace of `YUV`, `LUV`, `YCrCb` and `GRAY` to experiment. For color parameters I used a `spatial size of 32x32` and `128 histogram bins`. HOG parameters like orientations and pixels per cell I varied between 8 and 16 while cells per block varied between 1 and 4 and   ...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a simple linear SVM using sklearn's LinearSVC. Code can be found at cell number six of `carDetection.ipynb`. I split the data on previous cell using 20% for testing set. Using `YCrCb ALL channel HOG` I was able to achieve an accuracy of 99%. I tried various combinations of `YUV`, `GRAY` and `HSV` but got better car detection with `YCrCb` on video pipeline. Still trying to figure out why...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search in lower areas of the picture where cars are more likely to appear and wounded up creating dimensions for a triangle shape on a list prior to searching for windows. This can be seen on third cell of `carDetection.ipynb`. This cell launches `create_windows()`   On cell number 9, `Heatmaps tests with example images`, I experimented with four different combinations of x and y limits, window size and overlap. I perform the `slide_window()` four times and sum all windows together. I then perform `search_window()` and this function then runs `single_img_features` with parameters loaded from pickle file. Afterwards use a threshold to validate candidate predictions.
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched windosWS using YCrCb all-channel HOG features, which provided a decent result. To optimize the performance I turned off spatial and histogram feature extraction. Not only did it save time, but classifier greatly improved.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive thresholded detections I created a cache on fourth cell of `carDetection.ipynb` using `collections.deque`. I then use `itertools.chain` to iterate previous window results stored on `params` dictionary onto a single sequence.  I then run `window_detect()` which launches `search_window()` for each frame of the video. Afterwards, heatmaps are updated and cached instances are appended to list,  thresholded and blurred. With this I generated labels and with these labels I draw boxes. For video pipeline I usued a cache length of 110 and a heatmap threshold of 14. I also implemented a `decision function` threshold.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. I began experimenting a lot with combinations of the three extraction methods discussed. However, the most efficient and fastest classifier was to only extract hog features using grayscale color space. This project was kind of a hassle since I struggled so much with debugging the combination of feature extraction only to find out that this wasn't the best alternative. I experienced a lot of broadcasting errors whenever I tried appending features from different methods onto a list. I felt like I spent a greater portion of this project debugging problems rather than experimenting with actual classification.

My pipeline still needs work. Still need to improve how some false positives are being dealt with and experiment with more hyperparameter tunning.
