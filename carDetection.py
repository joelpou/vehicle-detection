import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """

# Define a function to compute histogram of gradients (hog) features
def get_hog(img, orient, pix_per_cell, cell_per_block, vis,
                     feature_vec):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return_list = hog(gray, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False,
                                  visualise= vis, feature_vector= feature_vec)
    # name returns explicitly
    hog_features = return_list[0]
    if vis:
        hog_features = features.reshape(-1)
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        hog_features = features.reshape(-1)
        return hog_features


def extract_color_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        image = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                featureImg = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                image = img * 255
                image = img.astype(np.uint8)
                featureImg = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
            elif cspace == 'HLS':
                featureImg = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                featureImg = cv2.cvtColor(image, cv2.COLOR_RGBYUV)
        else: featureImg = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatialFtrs = bin_spatial(featureImg, spatial_size)
        # Apply color_hist() to get color histogram features
        histFtrs = color_hist(featureImg, nbins = hist_bins, bins_range = hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatialFtrs, histFtrs)))
        # Apply hog to get HOG features
        # hogFtrs = get_hog(img, orient= 9,
        #                 pix_per_cell= 8, cell_per_block= 2,
        #                 vis=False, feature_vec=True)

    # Return list of feature vectors
    return features

def extract_hog_features(imgs, cspace='RGB', orient=9,
                    pix_per_cell=8, cell_per_block=2, hog_channel=0):
# Create a list to append feature vectors to
features = []
# Iterate through the list of images
for file in imgs:
    # Read in each one by one
    image = mpimg.imread(file)
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # Append the new feature vector to the features list
    features.append(hog_features)
# Return list of feature vectors
return features

# spatial = 32
# histbin = 32
#
# carFtrs = extract_features(cars, cspace='RGB', spatial_size=(spatial, spatial),
#                         hist_bins=histbin, hist_range=(0, 256))
# notcarFtrs = extract_features(notcars, cspace='RGB', spatial_size=(spatial, spatial),
#                         hist_bins=histbin, hist_range=(0, 256))

colorspace = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 16
pix_per_cell = 15
cell_per_block = 4
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

t=time.time()
car_features = extract_hog_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
notcar_features = extract_hog_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Define array stack of feature vectors
X = np.vstack((carFtrs, notcarFtrs)).astype(np.float64)

# Define the labels vectors
y = np.hstack((np.ones(len(carFtrs)), np.zeros(len(notcarFtrs))))

# Split up data into randomized training and test snippets
randState = np.random.randint(0, 100)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = randState)

# Fit a per-column scaler only on the training data
Xscaler = StandardScaler().fit(Xtrain)
# Apply the scaler to Xtrain and Xtest
Xtrain = Xscaler.transform(Xtrain)
Xtest = Xscaler.transform(Xtest)

print('Using spatial binning of:', spatial, 'and', histbin, 'histogram bins')
print('Feature vector length:', len(Xtrain[0]))

# Use a liner SVC
svc = LinearSVC()

# Check the training time for the SVC
t = time.time()
svc.fit(Xtrain, ytrain)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')

# Check the score  of the SVC
print('Test Accuracy of SVC = ', round(svc.score(Xtest, ytest), 4))
# Check the prediction time for a single sample
t = time.time()
nPredict = 10
print('My SVC predicts: ', svc.predict(Xtest[0:nPredict]))
print('For these', nPredict, 'labels: ', ytest[0:nPredict])

t2 = time.time()
print(round(t2 - t, 5), 'Seconds to predict ', nPredict, ' labels with SVC')
# if len(car_features) > 0:
#     # Create an array stack of feature vectors
#     X = np.vstack((car_features, notcar_features)).astype(np.float64)
#     # Fit a per-column scaler
#     X_scaler = StandardScaler().fit(X)
#     # Apply the scaler to X
#     scaled_X = X_scaler.transform(X)
#     car_ind = np.random.randint(0, len(cars))
#     # Plot an example of raw and scaled features
#     fig = plt.figure(figsize=(12,4))
#     plt.subplot(131)
#     plt.imshow(mpimg.imread(cars[car_ind]))
#     plt.title('Original Image')
#     plt.subplot(132)
#     plt.plot(X[car_ind])
#     plt.title('Raw Features')
#     plt.subplot(133)
#     plt.plot(scaled_X[car_ind])
#     plt.title('Normalized Features')
#     fig.tight_layout()
# else:
#     print('Your function only returns empty feature vectors...')
