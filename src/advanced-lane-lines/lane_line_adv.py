import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import moviepy.editor as mp

print('Current dir: ' + os.getcwd())
##################################
# camera calibration
## go over the chessboard images and show the points that CV2 detects
# size of the chessboard image
grid_size = (9,6)

# arrays to store points from the checkerboard images
imgpoints = []
objpoints = []

# set the 3D points on the checkerboard images
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# make a list of all filepaths of the calibration images
image_paths = glob.glob('./camera_cal/calibration*.jpg')

# fig, axs = plt.subplots(5,4, figsize=(16, 11))
# fig.subplots_adjust(hspace = .2, wspace=.001)
# axs = axs.ravel()

# for i, fn_img in enumerate(image_paths):
#     img = cv2.imread(fn_img)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # find Chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray_img, grid_size, None)
#     # axs[i].axis('off')

#     if (ret == True):
#         objpoints.append(objp)
#         imgpoints.append(corners)

# #         # draw the image
# #         # img = cv2.drawChessboardCorners(img, grid_size, corners, ret)
# #         # axs[i].imshow(img)

mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
                [0.00000000e+00, 1.15282291e+03, 3.86128937e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])


# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
print('Camera Calibration Matrix')
print(mtx)
print('Distortion Matrix')
print(dist)

##################################
# remove camera lens distortion
def undistort(img, matx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)
# ine transform so that a rectangle in the transformed view
#looks stretched along the road from the camera's perspective
fn = './test_images/straight_lines1.jpg'
img = mpimg.imread(fn)

height, width = img.shape[:2]

##################################
# perspective transform
# Points for the original image
start_points = np.float32([
    [210, 700],
    [570, 460],
    [705, 460],
    [1075, 700]
])
# Points for the new image
trans_points = np.float32([
    [400, 720],
    [400, 0],
    [width-400, 0],
    [width-400, 720]
])

##################################

def add_points(img, start_points):
    # draws circles at the coordinates ined in start_points
    img_out = np.copy(img)
    color = [255, 0 ,0]
    thickness = -1
    radius = 10
    for i in range(len(start_points)):
        cv2.circle(img_out, tuple(start_points[i]), radius, color, thickness)
    return img_out

def add_lines(img, start_points):
    img_out = np.copy(img)
    color = [255, 0, 0] # Red
    thickness = 3
    x0, y0 = start_points[0]
    x1, y1 = start_points[1]
    x2, y2 = start_points[2]
    x3, y3 = start_points[3]
    cv2.line(img_out, (x0, y0), (x1, y1), color, thickness)
    cv2.line(img_out, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img_out, (x2, y2), (x3, y3), color, thickness)
    cv2.line(img_out, (x3, y3), (x0, y0), color, thickness)
    return img_out

def add_points_lines(img, start_points):
    return add_lines(add_points(img, start_points), start_points)

def transform_img(img):
    # use start_points, trans_points to ine the perspective transformation
    img_size = (img.shape[1], img.shape[0])
    transform_matrix = cv2.getPerspectiveTransform(start_points, trans_points)
    img_out = cv2.warpPerspective(img, transform_matrix, img_size, flags = cv2.INTER_NEAREST)
    return img_out

def inverse_transform_img(img):
    # use start_points, trans_points to ine the inverse perspective transformation
    img_size = (img.shape[1], img.shape[0])
    transform_matrix = cv2.getPerspectiveTransform(trans_points, start_points)
    img_out = cv2.warpPerspective(img, transform_matrix, img_size, flags = cv2.INTER_NEAREST)
    return img_out

##################################
# color selection
def applyThreshold(channel, thresh):
    # Create an image of all zeros
    binary_output = np.zeros_like(channel)

    # Apply a threshold to the channel with inclusive thresholds
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output

def rgb_rthresh(img, thresh=(125, 255)):
    # Pull out the R channel - assuming that RGB was passed in
    channel = img[:,:,0]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

def hls_sthresh(img, thresh=(125, 255)):
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Pull out the S channel
    channel = hls[:,:,2]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

def lab_bthresh(img, thresh=(125, 255)):
    # Convert to HLS
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Pull out the B channel
    channel = lab[:,:,2]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

def luv_lthresh(img, thresh=(125, 255)):
    # Convert to HLS
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # Pull out the L channel
    channel = luv[:,:,0]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

##################################
# Canny edge detection
# Canny edge detector
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def run_canny(img, kernel_size=5, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian Blur
    gausImage = gaussian_blur(gray, kernel_size)

    # Run the canny edge detection
    cannyImage = canny(gausImage, low_thresh, high_thresh)

    return cannyImage

##################################
# binary image pipeline -
# undistort
# perspective transform
# rgb, hls, lab, luv thresholds
# canny edge detection
def binary_pipeline(img, show_images=False, \
                   canny_kernel_size=5 , canny_thresh_low=50, canny_thresh_high=150, \
                   r_thresh_low=225, r_thresh_high=255, \
                   s_thresh_low=220, s_thresh_high=250, \
                   b_thresh_low=175, b_thresh_high=255, \
                   l_thresh_low=215, l_thresh_high=255):
    fp = './output_images/doc/'

    # Copy the image
    img2 = np.copy(img)
    plt.imshow(img)
    # fn =  fp + 'img'
    # plt.savefig(fn, dpi=300)
    # Undistort the image based on the camera calibration
    undist = undistort(img, mtx, dist)

    # warp the image based on our perspective transform
    warped = transform_img(undist)

    ### COLOR SELECTION
    # Get the Red and saturation images
    r = rgb_rthresh(warped, thresh=(r_thresh_low, r_thresh_high))
    s = hls_sthresh(warped, thresh=(s_thresh_low, s_thresh_high))
    b = lab_bthresh(warped, thresh=(b_thresh_low, b_thresh_high))
    l = luv_lthresh(warped, thresh=(l_thresh_low, l_thresh_high))

    ### EDGE DETECTION

    # Run canny edge detector
    edge = run_canny(warped, kernel_size=canny_kernel_size, low_thresh=canny_thresh_low, high_thresh=canny_thresh_high)

    ### Create plots if we want them
    if show_images:
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 7))
        f.tight_layout()

        ax1.set_title('r', fontsize=10)
        ax1.axis('off')
        ax1.imshow(r, cmap='gray')

        ax2.set_title('s', fontsize=15)
        ax2.axis('off')
        ax2.imshow(s, cmap='gray')

        ax3.set_title('b', fontsize=15)
        ax3.axis('off')
        ax3.imshow(b, cmap='gray')

        ax4.set_title('l', fontsize=15)
        ax4.axis('off')
        ax4.imshow(l, cmap='gray')

        ax5.set_title('sobel', fontsize=15)
        ax5.axis('off')
        ax5.imshow(edge, cmap='gray')


    # combine these layers
    combined_binary = np.zeros_like(r)
    combined_binary[ (r == 1) | (s == 1) | (b == 1) | (l == 1) | (edge == 1) ] = 1

    return combined_binary

##################################
xm_per_pix = 3.7/550 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
ym_per_pix = 3*8/720 # meters per pixel in y dimension, 8 lines (5 spaces, 3 lines) at 10 ft each = 3m
# fn = './test_images/test1.jpg'
# img = mpimg.imread(fn)

# bin_img = binary_pipeline(img)

# plt.imshow(bin_img, cmap='gray')
# plt.show()

def calc_line_fits(img):
    """
    takes a binary image and returns a line of best fit using the bounding box method

    Args:
        img (np.array): binary image with white pixels as the lane lines
    """
    # number of sliding windows
    nwindows = 9
    # width of the windows
    margin = 100
    # minimum number of pixels needed to recenter the window
    minpix = 50

    # get a histogram of the white pixels in the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:, :], axis= 0)
    # plt.plot(histogram)
    # plt.show()
    out_img = np.dstack((img, img, img))*255

    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(img.shape[0]/nwindows)

    # identify x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # current positions of the windows
    left_current_x = left_base
    right_current_x = right_base

    left_lane_inds = []
    right_lane_inds = []

    for window_idx in range(nwindows):
        # create the window boundaries for left and right lane lines
        # y_high = (window_idx+1)*window_height
        # y_low = window_idx*window_height
        y_low = img.shape[0] - (window_idx+1)*window_height
        y_high = img.shape[0] - window_idx*window_height

        x_left_high = left_current_x + margin
        x_left_low = left_current_x - margin

        x_right_high = right_current_x + margin
        x_right_low = right_current_x - margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(x_left_low,y_low),(x_left_high,y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(x_right_low,y_low),(x_right_high,y_high),(0,255,0), 2)

        # identify indicies of white pixels in the box
        left_idx = ((nonzero_y >= y_low) &\
                        (nonzero_y < y_high) &\
                        (nonzero_x >= x_left_low) &\
                        (nonzero_x < x_left_high)).nonzero()[0]

        right_idx = ((nonzero_y >= y_low) &\
                        (nonzero_y < y_high) &\
                        (nonzero_x >= x_right_low) &\
                        (nonzero_x < x_right_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(left_idx)
        right_lane_inds.append(right_idx)

        # If you found more than minpix pixels, recenter next window on their mean position
        if len(left_idx) > minpix:
            left_current_x = np.int(np.mean(nonzero_x[left_idx]))
        if len(right_idx) > minpix:
            right_current_x = np.int(np.mean(nonzero_x[right_idx]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # get the pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    # fit a parabola through these bois
    # these are the parameters of the parabola
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # calculate the fit in meters too
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # highlight the pixels we identified in the boxes
    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, left_fit_m, right_fit_m, out_img

# left_fit, right_fit, out_img = calc_line_fits(bin_img)
# plt.imshow(out_img)

# Generate x and y values for plotting the fit parabola
# ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0] )
# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# plt.plot(left_fitx,ploty, color='yellow')
# plt.plot(right_fitx, ploty,color='yellow')
# plt.show()

##################################
# line fitting pipeline
# fn = './test_images/test1.jpg'
# img = mpimg.imread(fn)
def line_fitting_pipeline(img):
    bin_img = binary_pipeline(img)
    left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(bin_img)
    return left_fit, right_fit, left_fit_m, right_fit_m, out_img

def calc_radius(y_eval, current_fit_m):
    dy = (2*current_fit_m[0]*y_eval + current_fit_m[1])
    ddy = 2*current_fit_m[0]
    curve_radius = ((1 + dy**2)**1.5) / np.absolute(ddy)
    return curve_radius

def combine_radii(y_eval, left_fit_m, right_fit_m):
    left = calc_radius(y_eval, left_fit_m)
    right = calc_radius(y_eval, right_fit_m)

    return np.average([left, right])

def get_center_dist(left_fit, right_fit):

    # grab the x and y fits at px 700 (slightly above the bottom of the image)
    y = 700.
    image_center = 640. * xm_per_pix

    leftPos = left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2]
    rightPos = right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2]
    lane_middle = int((rightPos - leftPos)/2.)+leftPos
    lane_middle = lane_middle * xm_per_pix

    mag = lane_middle - image_center
    if (mag > 0):
        direction = "Right"
    else:
        direction = "Left"

    return direction, mag


##################################
# final pipeline, including drawing on original image
y_eval = 720. * ym_per_pix

def create_lane_image(img, bin_img, left_fit, right_fit):
    # plot the lines on the transformed image, then do the
    # inverse transform to get the lines on the original image
    warp_zero = np.zeros_like(bin_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = inverse_transform_img(color_warp)

    # Combine the result with the original image
    result_img = cv2.addWeighted(img, 1, newwarp, 0.5, 0)
    return result_img

def add_text_final(img, curve_rad, direction, center):
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Radius of curvature: ' + '{:04.0f}'.format(curve_rad) + ' m'
    cv2.putText(img, text, (50,100), font, 1.5, (0,255, 0), 2, cv2.LINE_AA)

    text = '{:03.2f}'.format(abs(center)) + 'm '+ direction + ' of center'
    cv2.putText(img, text, (50,175), font, 1.5, (0,255, 0), 2, cv2.LINE_AA)

    return img

def final_pipeline(img):
    bin_img = binary_pipeline(img)
    # plt.imshow(bin_img, cmap='gray')
    # plt.show()
    left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(bin_img)
    curve_rad = combine_radii(y_eval, left_fit_m, right_fit_m)
    direction, center = get_center_dist(left_fit, right_fit)

    lane_img = create_lane_image(img, bin_img, left_fit, right_fit)
    lane_img = add_text_final(lane_img, curve_rad, direction, center)
    return lane_img

##################################
# test_img_fn = 'test2.jpg'
# img = mpimg.imread('test_images/' + test_img_fn)

# out_img = final_pipeline(img)

# generating videos
fps = 24
def make_frame(t):
    return output_frames[int(t*fps)]

test_video_fn = 'harder_challenge_video.mp4'
video = mp.VideoFileClip('./test_videos/'+ test_video_fn).subclip(0, 10)
print('Video Duration (seconds): ' + str(video.duration))

output_frames = []

for frame in video.iter_frames(fps=fps):
    out_img = final_pipeline(frame)
    output_frames.append(out_img)

clip = mp.VideoClip(make_frame, duration = video.duration)
clip.fps = fps

# final_video = mp.clips_array([[,clip]])


# output_video = mp.concatenate([clips])
clip.write_videofile('./output_videos/' + test_video_fn)
