**Advanced Lane Finding Project**

The goals / steps of this project are the following

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./camera_cal/calibration9.jpg "Original Image"
[image2]: ./output_images/cameraCalibration.jpg "Camera Calibration"
[image3]: ./output_images/undistort1.jpg "Undistort Image"
[image4]: ./output_images/threshold.jpg "Thresholded Image"
[image5]: ./output_images/perspective.jpg "Warped Image"
[image6]: ./output_images/lanes.jpg "Finding Lanes Image"
[image7]: ./output_images/formula.jpg "Formula"

## Camera Calibration
---

###  1.  Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in Ipython notebook, location: " **./CarND-Advanced-Lane-Lines/AdvancedLaneFinding.ipynb**" under camera calibration.  

**Steps**

* Prepared object points
* Found Corners using opecv method `cv2.findChessboardCorner()` in the image. 
* Drawing the corners in the image using `cv2.drawChessboardCorners()`.
* Used image points and object points to compute calibration using `cv2.calibrateCamera()` in the Image.
* Removed distortion using `cv2.undistort()` method.

**Code Snippet**
```python
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
for fname in images:
    img = mpimg.imread(fname)
    # Convert Images to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the corners in the images
    ret, corners = cv2.findChessboardCorners(img, (9,6), None)
    # Map image points and object points in the image and calibrate the camera 
    if(ret==True):
        imgpoints.append(corners)
        objpoints.append(objp)
        image = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        dst = cv2.undistort(image, mtx, dist, None, mtx)
```
**Original Image**
![alt_text][image1]

**Distortion Corrected Calibration Image**
![alt_text][image2]

### Pipeline (single images)
---
#### 1.  Provide an example of a distortion-corrected image

![alt_text][image3]

#### 2.  Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is located in the Ipython notebook, under ***Combined Threshold***  cell.

Here I used combination of color and gradient threshold. Sobel filters heliped me get derivative in x and y directions, which led me detect more edges in the image.

I used light mask on sobel output to detect lanes covered under shadows and saturation mask to detect yellow lane lines in the image.

**Code Snippet**
```python
def combined_threshold(image):
    
    s_channel, l_channel = hls(image)
    
    # Apply each of the thresholding functions
    gradx_l = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=9, thresh=(25, 100))
    gradx_s = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=9, thresh=(10, 200))
    grady_s = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=9, thresh=(10, 200))
    
    gradxy = np.zeros_like(s_channel)
    gradxy[(gradx_s == 1)&(grady_s == 1)] = 1
    
    light_mask = np.zeros_like(l_channel)
    light_mask[(s_channel >= 10) & (l_channel >= 110)] = 1

    combined = np.zeros_like(gradx_s)
    combined[((gradx_l == 1) | (gradxy == 1)) & (light_mask == 1)] = 1
    
    return combined```
    
Here's an example of output in the image:

![alt_text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is located in Ipython notebook under ***Perspective Transform***  cell.

Here I hardcoded source and destination points in the following manner and then used `cv2.getPerspectiveTransform()` function to get the warped images.

```python
    offset = 200
    src = np.float32([[570,460],[705, 460], [1140,720], [190,720]])      
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
```
**Warped Image**
![alt_text][image5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is located in Ipython notebook under ***Advanced Computer Vision*** cell.

**Steps :**

**Histogram Peaks** : I used histogram peaks algorithm to decide which pixels are part of the line and which belong to left and right line.

**Sliding Window** : The two highest peaks generated using histogram helped determining the starting point of the lane. I set up the windows and it's hyper-parameters and then iterated across the binary activations in the image to find all the pixels belonging to the line.

**Polynomial fit** : I used `np.polyfit()` function to fit polynomials on the lines.

**Search around polynomials** : The sliding window algorithm starts fresh with every fram, which is inefficient. So I just search around the previous line position.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is located in Ipython notebook under ***Draw on Image And Measure Curvature*** cell. I computed the radius of curvature of the fit with the help of the following fomulae-

![alt_text][image7]

**Code Snippet**

```python
def measure_curvature_pixels(img, left_fit, right_fit, ploty):
    
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    mid_imgx = img.shape[1]//2
        
    ## Car position with respect to the lane
    car_pos = (left_fit[-1] + right_fit[-1])/2
    
    ## Horizontal car offset 
    offset = (mid_imgx - car_pos) * xm_per_pix
    
    lane_center = (right_fit[-1] + left_fit[-1])/2

    center_offset_pixels = abs(img.shape[0]/2 - lane_center)
    offset_mtrs = xm_per_pix*center_offset_pixels
    
    return left_curverad, right_curverad, offset_mtrs
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced problems tweaking and fine tuning the gradients and color spaces. I implemented combination of sobel threshold, light and saturation mask which led me to satisfactory results.

My pipeline will fail for bigger curves, as I was unable to implement my pipeline on challenge_video and harder_challenge_video.
