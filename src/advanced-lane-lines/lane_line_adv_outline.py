# https://www.youtube.com/watch?v=VyLihutdsPk
###################################
## undistort for lens curvature
# get camera calibration with a checkerboard image
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                                    objpoints
                                    imgpoints,
                                    gray.shape[::-1],
                                    None,
                                    None)
# undistort original image
undist_img = cv2.undistory(img, mtx, dist, None, mtx)

###################################
## perspective transform, switch to birds-eye view s.t. perspective lines become parallel
M = cv2.getPerspectiveTransform(src, dst)
warped_img = cv2.warpPerspective(img, M, img_size)

###################################
## finding lane lines
# color selection
# some color spaces may be more useful for detection of lane lines
# check them out and see what looks good, other spaces include: HLS, HSV, LAB, LUV

# convert RGB to Hue Light Saturation (HLS)
# HSL_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2HLS)

# since yellow lane lines are bright in the red channel
# apply thresholds to red channel of warped img
channel = warped_img[:,:, 0]
thresh = (200, 255)
binary_red_img = np.zeros_like(channel)

# edge detection
# apply a binary threshold to get clean lane lines
binary_red_img[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

# apply Canny Edge detection to the warped image to detect some edges
# TODO select region of interest
low_threshold = 100
high_threshold = 300
canny_img = np.copy(warped_img)
canny_img = cv2.Canny(canny_img, low_threshold, high_threshold)

combined_binary_img = np.zeros_like(binary_red_img)
combined_binary_img[(red == 1) | (sobel == 1)] = 1
# now we have a binary image that should only show the lane lines
# you will have to tune parameters to deal with shadows

###################################
## fit a polynomial through the identified lane pixels

# method 1 - more robust
# take a histogram of white pixels along the x-dimension of the image
# the first peak is hypothesized to be where the left lane marker starts
# // is the integer division operator (divides, then takes the floor of the result)
histogram = np.sum(img[img.shape[0]//2:, :])

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# create a box (range of x and y values) around these maxima, any white pixels in the box are added to a list of xy values

# separate the image so you can have an integer number of boxes stacked in the y dimension

# add a box on top of the previous one, the middle point of the next box is the mean x position of the box below

# fit a polynomial to the xy positions of the pixels in the boxes
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# can convert pixels to meters by knowing the lanes are 3 meters wide


## method 2 - faster, uses a previous fit
# take the previous fit, move left and right by 100 pixels to create a sweeping area
# any pixels within the sweeping areas are added to the list of xy positions, and then a new fit is generated

###################################
# calculate radius of curvature using fit polynomial
# Ax**2 + Bx + C
fit = [A, B, C]
curve_rad = ((1+(2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])

# find distance to lane center
# rightPos, leftPos = x coordinate of fit polynomial at bottom of image
# imageCenter = img_x_dim / 2
# assume camera on centerline of car
laneCenter = (rightPos - leftPos) / 2 + leftPos
distanceToCenter = laneCenter - imageCenter

###################################
# plot on warped image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
cv2.polylines(color_warp, np.int32([pts_left]))

result = cv2.addWeighted(img, 1, new_warp, 0.5, 0)
# put text on the final image
cv2.putText(img, text, (50, 100), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

###################################
# making it more robust, doesn't work well on non-highway roads
# doesn't work well if a car is passing

