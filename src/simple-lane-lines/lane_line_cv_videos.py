import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import moviepy.editor as mp
from PIL import Image
from collections import deque

####################################################
print('Current dir: ' + os.getcwd())

####################################################
def region_of_interest_img(img):
    # apply a region of interest mask
    # select points on the image at the vertices of the region of interest
    # point = [x, y] coordinates in the image
    # image is 960 x 540 (width x height), using image coordinate system

    region_img = np.copy(img)
    left1 = [450, 300]
    left2 = [100, 539]
    top1 = [0, 300]
    top2 = [959, 300]
    right1 = [0, 0]
    right2 = [959, 539]

    # fit lines to identify a region of interest
    poly_degree = 1

    fit_left = np.polyfit((left1[0], left2[0]), (left1[1], left2[1]), poly_degree)
    fit_top = np.polyfit((top1[0], top2[0]), (top1[1], top2[1]), poly_degree)
    fit_right = np.polyfit((right1[0], right2[0]), (right1[1], right2[1]), poly_degree)

    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

    left_edge = (YY < (XX*fit_left[0] + fit_left[1]))
    top_edge = (YY < (XX*fit_top[0] + fit_top[1]))
    right_edge = (YY < (XX*fit_right[0] + fit_right[1]))

    # the highlighted region is the region of interest
    region_thresholds =  top_edge
    region_img[region_thresholds] = [255, 0, 0]
    region_thresholds = left_edge
    region_img[region_thresholds] = [255, 0, 0]
    region_thresholds = right_edge
    region_img[region_thresholds] = [255, 0, 0]

    return region_img

def grayscale_img(img):
    # convert to grayscale using opencv
    gray_img = np.copy(img)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

    return gray_img

def gaussian_blur_img(img, kernel_size = 3):
    # apply Gaussian smoothing (makes it easier to pick out sharp edges with Canny)
    blur_img = np.copy(img)
    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur_img

def canny_img(img, low_threshold = 100, high_threshold = 300):
    # apply Canny edge detection
    canny_img = np.copy(img)
    canny_img = cv2.Canny(canny_img, low_threshold, high_threshold)
    return canny_img

def hough_img(img, rho = 1, theta = np.pi/180., threshold = 50, min_line_length = 10, max_line_gap = 1):
    # apply Hough transform
    hough_img = np.copy(img)
    line_img = np.copy(image)*0

    lines = cv2.HoughLinesP(hough_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # classify Hough lines as left lane marker or right lane marker
    #line = object with two points x1, y1, x2, y2
    left_line = []
    right_line = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1) / (x2-x1)
            if slope > 0:
                right_line.append(line)
            else:
                left_line.append(line)

    # take averages of each set of lines and draw the resulting line
    num_left_lines = np.shape(left_line)[0]
    num_right_lines = np.shape(right_line)[0]

    left_line_avg = np.zeros(np.shape(left_line)[2])
    right_line_avg = np.zeros(np.shape(right_line)[2])

    for line in left_line:
        left_line_avg += line.squeeze()

    for line in right_line:
        right_line_avg += line.squeeze()

    left_line_avg = (left_line_avg/num_left_lines).astype(int)
    right_line_avg = (right_line_avg/num_right_lines).astype(int)

    # fit lines to the lane markers
    poly_degree = 1

    fit_left = np.polyfit((left_line_avg[0], left_line_avg[2]), (left_line_avg[1], left_line_avg[3]), poly_degree)
    fit_right = np.polyfit((right_line_avg[0], right_line_avg[2]), (right_line_avg[1], right_line_avg[3]), poly_degree)

    fit_left_history.append((fit_left[0], fit_left[1]))
    fit_right_history.append((fit_right[0], fit_right[1]))

    fit_left_avg = deque_avg(fit_left_history, moving_average_window)
    fit_right_avg = deque_avg(fit_right_history, moving_average_window)

    # fit linear
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    left_edge = (YY > (XX*fit_left_avg[0] + fit_left_avg[1]))
    right_edge = (YY > (XX*fit_right_avg[0] + fit_right_avg[1]))

    region_thresholds =  left_edge & right_edge
    line_img[region_thresholds] = [255, 0, 0]
    return line_img

def deque_avg(deque, num_history):
    """Find the average of each component in a list of lists

    Args:
        deque (deque): deque = [[a1, b1], [a2, b2], ...]
        a historical list of the parameters of linear fit
    """
    # f(x) = ax + b
    a_avg = 0
    b_avg = 0
    if len(deque) < num_history:
        num_history = len(deque)

    for i in range(num_history):
        a_avg += deque[i][0]
        b_avg += deque[i][1]

    return a_avg / num_history, b_avg / num_history

####################################################
# video = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')
fps = 24

#TODO get it to run longer than 6 seconds...
test_video_fn = 'solidWhiteRight.mp4'
video = mp.VideoFileClip('./test_videos/'+test_video_fn).subclip(0, 5)

# test_video_fn = 'solidYellowLeft.mp4'
# video = mp.VideoFileClip('./test_videos/'+ test_video_fn).subclip(0, 6)

print('Video Duration (seconds): ' + str(video.duration))
output_frames = []

# print('This image is :', type(image),
# 'with dimensions (height, width, color depth): ', image.shape)
current_frame = 0
moving_average_window = 120
fit_left_history = deque(maxlen = moving_average_window)
fit_right_history = deque(maxlen = moving_average_window)

for frame in video.iter_frames(fps=fps):
    image = frame
    fp = './output_images/doc/'
    # print(np.shape(image))
    ysize, xsize = image.shape[0], image.shape[1]

    img = np.copy(image)
    # plt.imshow(img)
    # fn =  fp + 'img'
    # plt.savefig(fn, dpi=300)

    img = region_of_interest_img(img)
    # plt.imshow(img)
    # fn =  fp + 'roi'
    # plt.savefig(fn, dpi=300)

    img = grayscale_img(img)
    # plt.imshow(img, cmap='gray')
    # fn =  fp + 'gray'
    # plt.savefig(fn, dpi=300)

    img = gaussian_blur_img(img)
    # plt.imshow(img, cmap='gray')
    # fn =  fp + 'blur'
    # plt.savefig(fn, dpi=300)

    img = canny_img(img)
    # plt.imshow(img, cmap='gray')
    # fn =  fp + 'canny'
    # plt.savefig(fn, dpi=300)

    img = hough_img(img)
    # plt.imshow(img, cmap='gray')
    # fn =  fp + 'hough'
    # plt.savefig(fn, dpi=300)

    plt.show()

    combined_img = cv2.addWeighted(image, 1.0, img, 0.4, 0)

    output_frames.append(combined_img)
    # plt.imshow(combined_img)
    # plt.savefig('./output_videos/cv/' + str(current_frame) + '.jpg' , dpi = 300)
    current_frame += 1

def make_frame(t):
    return output_frames[int(t*fps)]

clip = mp.VideoClip(make_frame, duration = video.duration)

clip.fps = fps
# output_video = mp.concatenate([clips])
clip.write_videofile('./output_videos/cv/' + test_video_fn)
