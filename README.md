DicomViewerLeapMotion
=====================

DICOM viewer application for Leap Motion

## Dependencies

[Leap Motion SDK V1](https://developer.leapmotion.com/downloads) -- has not been tested with the v2 yet.

[OpenCV 2.4.8](http://opencv.org/) -- if you're using the Anaconda distribution for python and are running MacOS, here is an extremely useful [snippet](https://gist.github.com/welch/6468594) on how to build OpenCV.

## Usage

```
python dcm_gui.py your_dicom_directory
```

## How to

Waving one hand over the Leap Motion controller initiates several features:

1. __Change the brightness__ of the image by swiping your hand to the left/right side
2. __Navigate__ throughout _z_-slices by tracing a circle anti- or clockwise
3. __Translate__ the image in _x_ or _y_ axis by moving three-fingers up and down or left to right
4. __Rotate__ the image around the _z_ axis by rotating your hand
5. __Zoom in/out__ by a pinch-to-zoom gesture with two fingers

Note that your fingers have to be in the _"touch zone"_ to create one of the above events. The Leap Motion interaction space is defined as follows:

![alt text](http://www.blogcdn.com/www.engadget.com/media/2013/07/boomtouchless.jpg "interaction space")

This is the feedback that you will get when using the app:

![alt text](http://i57.tinypic.com/28tgwf6.png "example")
(green dots indicate that the fingers are in the hover zone, blue dots in the touch zone, and red dots are close to leave the hover zone).
