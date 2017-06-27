import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import *


# First I am going to train a SVM classifier to recognize what is a car and what is not.
# To train this classifier I am using the GTI and KITTI data.
# This function loads all the paths to these images.
# It also ensure that I have the same number of car images and non car images.
def load_paths():
	# I load all the paths of the car images
	car_far_paths = sorted(glob.glob('vehicles/GTI_Far/image*.png'))
	car_left_paths = sorted(glob.glob('vehicles/GTI_Left/image*.png'))
	car_right_paths = sorted(glob.glob('vehicles/GTI_Right/image*.png'))
	car_middle_paths = sorted(glob.glob('vehicles/GTI_MiddleClose/image*.png'))
	car_kitti_paths = sorted(glob.glob('vehicles/KITTI_extracted/*.png'))
	
	# I load all the paths of the non car images
	not_car_gti_paths = sorted(glob.glob('non-vehicles/GTI/image*.png'))
	not_car_extras_path = sorted(glob.glob('non-vehicles/Extras/extra*.png'))

	# I put all these paths in 2 lists
	car_paths = car_far_paths+car_left_paths+car_right_paths+car_middle_paths+car_kitti_paths
	not_car_paths = not_car_gti_paths+not_car_extras_path
	
	# I make sure to use the same number of car and non car images
	nb_paths = min(len(not_car_paths),len(car_paths))
	car_paths = car_paths[:nb_paths]
	not_car_paths = not_car_paths[:nb_paths]

	return car_paths, not_car_paths

# Once the images are loaded, I need to extract some features that will be the inputs of my classifier
# The first feature is going to be the raw pixel values of the images.
# Since the pictures of the video have high resolution, I need to use spatial binning to be
# able to extract this feature quickly. I chose to resize the image to 32x32 pixels.
# Even with this resolution, I can still identify the cars by eye so it is relevant to use
# this feature with spatial binning to 32x32.
def get_spatial_features(img, size=(32, 32)):
	# I resize the image and use .ravel() to create a feature vector (1D)
	spatial_features = cv2.resize(img, size).ravel() 
	return spatial_features

# The second part of my feature vector is composed of the histogram of pixels from each color channel
# I perform these histograms for all 3 channels and use 32 bins
def get_color_features(img, nbins=32, bins_range=(0, 256)):
	# I compute the histograms over the 3 color channels
	ch1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	ch2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	ch3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

	# I concatenate them to create a feature vector
	color_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
	return color_features

# Finnally, the last part of my feature vector is going to be a histogram oriented gradient (HOG).
def get_hog_features(img, orient=8, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
	# If vis is True, I am going to output an image representing the HOG feature
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),\
									cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
									visualise=True, feature_vector=False)
		return features, hog_image
	# Otherwise, I just output the feature vector
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), \
						cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
						visualise=False, feature_vector=feature_vec)
		return features

# Now I am going to combine these 3 features vectr into 1.
# Note that I am going to use the YCrCb color space instead of the RGB one.
def get_features(img, color_space='YCrCb', size=(32,32), nbins=32, orient=8, pix_per_cell=8, \
					cell_per_block=2, hog_channel='ALL'):
	
	features = []
	
	# First I convert the image to the color space I want to use (here YCrCb)
	if color_space == 'HSV':
		image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	elif color_space == 'HLS':
		image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	elif color_space == 'YUV':
		image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	elif color_space == 'YCrCb':
		image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else:
		image = np.copy(img)

	# Then I extract the first part of the features vector (raw pixel values with spatial binning)
	spatial_features = get_spatial_features(image, size)
	features.append(spatial_features)

	# Then I extract the second part of the features vector (color histogram over the 3 channels)
	color_features = get_color_features(image, nbins)
	features.append(color_features)

	# Then I extract the last part of the features vector (HOG)
	if hog_channel == 'ALL':
		# If I want to use all the color channel, I need to extract the 
		# HOG feature of each channel and then concatenate them
		hog_features = []

		for channel in range(image.shape[2]):
			ch_hog_features = get_hog_features(image[:,:,channel],orient,pix_per_cell,cell_per_block)
			hog_features.append(ch_hog_features)
		hog_features = np.ravel(hog_features)

	else:
		hog_features = get_hog_features(image[:,:,hog_channel],orient,pix_per_cell,cell_per_block)

	features.append(hog_features)

	# I concatenate all the 3 features into 1 feature vector
	features = np.concatenate(features)
	return features

# Now that I can get a feature vector out of any image, I am going to write a function 
# that takes in input a list of paths to images and oututs the list of corresponding feature vectors
def extract_features(paths, color_space='YCrCb', size=(32,32), nbins=32, orient=8, pix_per_cell=8, \
						cell_per_block=2, hog_channel='ALL'):

	features = []
	for path in paths:
		img = cv2.imread(path)
		# OpenCV uses BGR images, so I first need to convert it to RGB
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		features.append(get_features(img,color_space,size,nbins,orient,pix_per_cell,cell_per_block,hog_channel))

	return features

# Now let's prepare the training, validation and test sets to train the SVM classifier
def prepare_sets(proportion = 0.15):
	# First I load all the paths of car and non car images (and make sure I have the same number)
	car_paths, not_car_paths = load_paths()

	# Then I convert them into lists of feature vectors
	car_features = extract_features(car_paths)
	not_car_features = extract_features(not_car_paths)

	# Then I put them together to have a list of all the feature vectors (cars and non cars)
	X_unscaled = np.vstack((car_features,not_car_features)).astype(np.float64)

	# I scale and normalize that list thanks to StandardSacler()
	X_scaler = StandardScaler().fit(X_unscaled)
	X_scaled = X_scaler.transform(X_unscaled)

	# I create the list of labels (1 for cars and 0 for non cars)
	y = np.hstack((np.ones(len(car_features)),np.zeros(len(not_car_features))))

	# I split these lists into a training, validation and test sets.
	# The training set has 70% of the data.
	# The test and validation sets have each 15% of the data.
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=2*proportion)
	X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

	return X_train, y_train, X_val, y_val, X_test, y_test, X_scaler

# Now that I have my training, validation and test sets, I can train my classifier
# Note that I used the validation set to fine tune the parameters of my extract_features function
# (the color space, the size of space binning, the number of orientation for HOG features ...)
# I use the test set only at the end to test (after fine tuning the parameter) to avoid overfitting
def fit_svc():
	# First I create my training, validation and test sets
	X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_sets()

	# Then I train the classifier on the training set
	svc = LinearSVC()
	svc.fit(X_train, y_train)

	print('Train Accuracy of SVC: ' + str(round(svc.score(X_train,y_train),4)))
	# print('Cross Validation Accuracy of SVC: ' + str(round(svc.score(X_val,y_val),4)))
	print('Test Accuracy of SVC: ' + str(round(svc.score(X_test,y_test),4)))

	return svc, scaler

# Now that my classifier is trained, given an image, it should be able to tell if it is a car or not.
# So I need to slice the picture taken from the dashboard into smaller images so that the classifier
# can analyze them and tell if ther is a car or not.
# To do that I am going to use a sliding window method.

# This function is going to return windows of sizes xy_window with overlap xy_overlap between 
# x_start,y_start and x_stop, y_stop (This function is the one from the lesson)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]

	# Compute the span of the region to be searched    
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]

	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

	# Compute the number of windows in x/y
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	
	return window_list

# Here I define the windows that I am going to feed the classifier for each frame of the video.
# Since the farther a car is the smaller it appears, the far windows are smaller than the close ones.
def boxes_to_scan(img):
	far_windows = slide_window(img, x_start_stop=[None,None], y_start_stop=[400,500], xy_window=(96,96))
	mid_windows = slide_window(img, x_start_stop=[None,None], y_start_stop=[400,550], xy_window=(144,144))
	close_windows = slide_window(img, x_start_stop=[None,None], y_start_stop=[430,650], xy_window=(192,192))
	close_windows_2 = slide_window(img, x_start_stop=[None,None], y_start_stop=[460,680], xy_window=(192,192))
	return far_windows+mid_windows+close_windows+close_windows_2

# So, the classifier will analyze each of these boxes and assign 1 if there is a car and 0 otherwise.
# Then, by adding all these boxes labeled by 0 or 1, I can create a heatmap where the heat represents
# the number of times the classifier found a car in the region. 
# (This function is the one from the lesson)
def add_heat(heatmap, boxes):
	# Iterate through list of boxes
	for box in boxes:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap

# Then the more heat there is, the greater the probability that there is a car in this region.
# So, by thresholding this heatmap, I can choose only the regions where the probability of a car is high
# This allows me to dismiss false positive (This function is the one from the lesson)
def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

# Once I have a thresholded heatmap, I want to draw boxes arround the regions where there is heat.
# For that I use the label() function and then this function to define the boxes from the labels.
# (This function is the one from the lesson)
def labeled_boxes(labels):
	boxes = []
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		boxes.append(box)
	# Return the image
	return boxes

# This function simply draws the specified rectangular boxes on the image
def draw_boxes(img, boxes, color=(0,0,255), thick=5):
	image = np.copy(img)
	for box in boxes:
		cv2.rectangle(image, box[0], box[1], color, thick)
	return image

# Now I know how to find and draw boxes around a car in an image.
# In a video, from a frame to the next, a car will not move a lot.
# So I am going to want to focus on the region of the frame where I already found a car.
# This function will give me windows to feed the classifier 
# around the position where I found a car in the previous frame.
# I take in input the box where I found the car in the previous frame 
# and outputs 5 boxes arround that position where I should look for a car in the next frame
def add_boxes_around(box, margin=25, size=(64,64)):
	# First I find the center of the box where I found the car in the previous frame
	x1,y1,x2,y2 = box[0][0],box[0][1],box[1][0],box[1][1]
	center_x = int(0.5*(x1+x2))
	center_y = int(0.5*(y1+y2))
	
	boxes = []
	# Then I define the centers of the 5 new boxes (I move from margin in every direction)
	new_centers = [[center_x,center_y], [center_x+margin,center_y], [center_x-margin,center_y],\
					[center_x,center_y+margin], [center_x,center_y-margin]]

	# For each new center I define a window of the given size arround this center
	for center in new_centers:
		x1_new = center[0] - int(size[0]/2)
		x2_new = center[0] + int(size[0]/2)
		y1_new = center[1] - int(size[1]/2)
		y2_new = center[1] + int(size[1]/2)
		boxes.append(((x1_new,y1_new),(x2_new,y2_new)))

	return boxes

# Now I am finally ready to process a frame of the video.
# previous_boxes contains the boxes where I found a car in the previous frame
# previous_heat contains the heatmaps of the previous nb_previous frames
# threshold is my final threshold over the heatmap
def process_image(img,svc,scaler,previous_boxes=[],previous_heat=[],nb_previous=10,threshold=8):
	# First I define the windows I am going to feed my classifier
	boxes = boxes_to_scan(img)
	# I add the windows around the region where I found the car in the previous frame 
	# (These are my region of interest = roi)
	for box in previous_boxes:
		roi_boxes = add_boxes_around(box)
		boxes = boxes + roi_boxes
	
	car_boxes = []

	# I am going to feed each window to my classifier
	for box in boxes:
		x1,y1,x2,y2 = box[0][0],box[0][1],box[1][0],box[1][1]
		crop = img[y1:y2,x1:x2]
		# I resize the image because the get_features function is expecting 64x64 images
		image = cv2.resize(crop, (64,64))
		# I extract the feature vector from the image
		features = get_features(image)
		# I scale and normalize the feature vector
		scaled_features = scaler.transform(features)
		# I use the classifier to know if it is a car or not
		if svc.predict(scaled_features) == 1:
			# If it is a car, I keep this window to use it in my heatmap
			car_boxes.append(box)

	# I create a heatmap of the current frame
	heatmap = np.zeros_like(img[:,:,0]).astype(np.float64)
	heatmap = add_heat(heatmap, car_boxes)
	final_heatmap = np.copy(heatmap)

	# I add all the previous heatmaps to avoid false positive
	for prev_heat in previous_heat:
		final_heatmap = final_heatmap + prev_heat

	# I threshold this final heatmap, and draw boxes around the regions of heat
	final_heatmap = apply_threshold(final_heatmap, threshold)
	labels = label(final_heatmap)
	final_boxes = labeled_boxes(labels)
	final_image = draw_boxes(img, final_boxes)

	# I add the heatmap of this frame to the list of previous heatmaps to pass on to the next frame
	# I take out the least recent heatmap
	previous_heat.append(heatmap)
	if len(previous_heat) > nb_previous:
		previous_heat.pop(0)

	# I return the image with boxes drawn,
	# the list of boxes to use as my region of interest in the next frame
	# and the list of previous heatmaps
	return final_image, final_boxes, previous_heat

# Finaly I can process the video frame by frame
def process_video(clip, svc, scaler, nb_previous=10, threshold=8):
	# I initialize my parameters
	new_frames = []
	previous_heat=[]
	previous_boxes = []
	counter = 1
	nb_frames = clip.fps * clip.duration

	# Then I go through each frame and process it
	for frame in clip.iter_frames():
		new_frame, previous_boxes, previous_heat = process_image(frame, svc, scaler, previous_boxes, previous_heat, nb_previous, threshold)
		new_frames.append(new_frame)
		print('Processing image: ' + str(counter) + '/' + str(nb_frames), end='\r')
		counter = counter + 1

	print('')
	# I put the clip back together
	new_clip = ImageSequenceClip(new_frames, fps=clip.fps)

	return new_clip


svc, scaler = fit_svc()

clip = VideoFileClip('project_video.mp4')

process_clip = process_video(clip, svc, scaler)
process_clip.write_videofile('project_video_out.mp4')


