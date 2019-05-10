import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

############################################
print('Current dir: ' + os.getcwd())
test_img_fn = 'whiteCarLaneSwitch.jpg'
image = mpimg.imread('test_images/' + test_img_fn)

print('This image is :', type(image),
'with dimensions (height, width, color depth): ', image.shape)
ysize, xsize = image.shape[0], image.shape[1]

############################################
def region_of_interest_img(img):
    # create a 'region of interest' mask
    region_img = np.copy(img)
    not_region_img = np.copy(img)
    # select 3 points on the image at the vertices of the region of interest
    # point = [x, y] coordinates in the image
    # image is 960 x 540 (width x height), using image coordinate system
    left1 = [450, 300]
    left2 = [100, 539]
    top1 = [0, 325]
    top2 = [959, 325]
    right1 = [0, 0]
    right2 = [959, 539]

    # fit lines to identify a 3 sided region of interest
    poly_degree = 1

    fit_left = np.polyfit((left1[0], left2[0]), (left1[1], left2[1]), poly_degree)
    fit_top = np.polyfit((top1[0], top2[0]), (top1[1], top2[1]), poly_degree)
    fit_right = np.polyfit((right1[0], right2[0]), (right1[1], right2[1]), poly_degree)

    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

    left_edge = (YY < (XX*fit_left[0] + fit_left[1]))
    top_edge = (YY < (XX*fit_top[0] + fit_top[1]))
    right_edge = (YY < (XX*fit_right[0] + fit_right[1]))

    # pixels outside the region of interest are set to black
    region_thresholds =  top_edge
    region_img[region_thresholds] = [0, 0, 0]
    region_thresholds = left_edge
    region_img[region_thresholds] = [0, 0, 0]
    region_thresholds = right_edge
    region_img[region_thresholds] = [0, 0, 0]

    # pixels inside the region of interest are set to black
    region_thresholds =  top_edge
    not_region_img[~top_edge & ~left_edge & ~right_edge] = [0, 0, 0]

    return region_img, not_region_img

def color_threshold_img(img, red_threshold, green_threshold, blue_threshold):
    color_threshold_img = np.copy(img)

    # set thresholds
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # create a mask that sets pixels to red if the value is greater than any of the thresholds
    color_thresholds = (img[:, :, 0] > rgb_threshold[0]) \
    | (img[:, :, 1] > rgb_threshold[1]) \
    | (img[:, :, 2] > rgb_threshold[2])

    # color_threshold_img[color_thresholds] = [0, 0, 0]
    color_threshold_img[color_thresholds] = [255, 0, 0]
    return color_threshold_img

############################################
region_img, not_region_img = region_of_interest_img(image)
processed_img = color_threshold_img(region_img, 175, 175, 175)
plt.imshow(processed_img)
plt.show()

combined_img = cv2.addWeighted(processed_img, 1.0, not_region_img, 1.0, 0)

plt.imshow(combined_img)
# plt.show()
fn = test_img_fn
plt.savefig('./output_images/basic/' + test_img_fn, dpi = 300)


