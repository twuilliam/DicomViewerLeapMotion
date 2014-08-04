import os
import dicom
import cv2
import sys
import numpy as np
import Leap


########################
# Extract dicom images #
########################

input_dir = sys.argv[1]

whole_img = []
for dcmfile in os.listdir(input_dir):
    fileName, fileExtension = os.path.splitext(
        os.path.join(input_dir, dcmfile)
        )

    if fileExtension == '.dcm':
        plan = dicom.read_file(os.path.join(input_dir, dcmfile))
        whole_img.append(plan.pixel_array)


def normalize_volume(v, minimum=None, maximum=None):
    if maximum is None and minimum is None:
        maximum = np.max(v)
        minimum = np.min(v)
    else:
        minimum = np.maximum(np.min(v), minimum)
        maximum = np.minimum(maximum, np.max(v))
    v = (v-minimum)/(maximum-minimum)
    cond = v > 1
    v[cond] = 1.0
    cond = v < 0
    v[cond] = 0.0
    return v

whole_img = np.float32(np.asarray(whole_img))


##########################
# Affine transformations #
##########################


class TransformDcm:
    def __init__(self, raw):
        self.raw = whole_img
        self.dimX = np.size(whole_img, 1)
        self.dimY = np.size(whole_img, 2)
        self.dimZ = np.size(whole_img, 0)

    def initparam(self):
        self.transX = 0
        self.transY = 0
        self.theta = 0
        self.scaling = 1

    def normalize(self, minimum=None, maximum=None):
        self.whole_img = normalize_volume(self.raw, minimum, maximum)
        self.transform()

    def transform(self):

        # Translation
        M = np.float32([[1, 0, self.transX],
                        [0, 1, self.transY],
                        [0, 0, 1]])

        # Rotation and Scaling
        M = np.dot(M, np.vstack((cv2.getRotationMatrix2D((int(self.dimX/2),
                                                          int(self.dimY/2)),
                                                         self.theta,
                                                         self.scaling), [0, 0, 1])))

        # Final tranformation matrix
        M = M[0:2, :]

        # Compute the image
        self.display_img = np.zeros((self.dimZ, self.dimX, self.dimY),
                                    dtype=np.float32)
        for i in xrange(self.dimZ):
            self.display_img[i] = cv2.warpAffine(self.whole_img[i], M,
                                                 (self.dimX, self.dimY))

#################
# Leap Listener #
#################


class TouchPointListener(Leap.Listener):

    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)

        self.swipeid = None
        self.previous_frame = controller.frame()

        swipe_length = 100.0
        swipe_velocity = 250
        if(controller.config.set("Gesture.Swipe.MinLength", swipe_length)
          and controller.config.set("Gesture.Swipe.MinVelocity", swipe_velocity)):
            controller.config.save()

    def add_dcm(self, TransformDcm, name_window):
        # Data
        self.dimX = TransformDcm.dimX
        self.dimY = TransformDcm.dimY
        self.dimZ = TransformDcm.dimZ
        self.dcmimage = TransformDcm
        self.name = name_window

        # Params
        self.prev_max = np.max(TransformDcm.raw)
        self.cur_max = np.max(TransformDcm.raw)
        self.prev_angle = 0
        self.prev_step = 0
        self.slide = int((self.dimZ-1)/2.)
        self.current_slide = self.slide
        self.show_info = False
        self.previous_progress = 0

        # Initial processing
        self.dcmimage.normalize()

        # 2D image initialization
        self.opencv_img = np.zeros((self.dimX, self.dimY, 3), dtype=np.uint8)
        self.set_image()

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"

    def on_frame(self, controller):

        # Current frame of the leap controller
        frame = controller.frame()

        if not frame.hands.is_empty:
            circle_id = None
            currentDir = ""
            for gesture in frame.gestures():
                if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                    circle = Leap.CircleGesture(gesture)

                    # Calculate the angle swept since the last frame
                    if circle.radius > 20 and np.size(frame.pointables) == 1 and frame.pointables[0].touch_distance < 0:

                        # Determine clock direction using the angle between the pointable and the circle normal
                        if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/4:
                            clockwiseness = 1
                        else:
                            clockwiseness = -1

                        if np.sign(circle.progress * clockwiseness) != np.sign(self.previous_progress) and self.previous_progress != 0:
                            self.slide = self.current_slide
                            print "changement de sens"

                        current_slice = self.getslice(self.slide + 2 * np.round(circle.progress) * clockwiseness)

                        if self.current_slide != current_slice:
                            self.current_slide = current_slice
                            print "changement de slide"

                        self.previous_progress = circle.progress * clockwiseness
                        circle_id = circle.id

                        print "%d %f %f %f %f" % (circle_id, self.slide,
                                                  self.current_slide,
                                                  circle.progress,
                                                  2 * np.round(circle.progress) * clockwiseness)

                    if circle.state == Leap.Gesture.STATE_STOP and circle_id == circle.id :
                        self.slide = self.current_slide

                if np.size(frame.hands) == 1:
                    hand = frame.hands[0]
                    fingers = hand.fingers
                    if np.size(fingers) >= 4 and gesture.type == Leap.Gesture.TYPE_SWIPE:
                        swipe = Leap.SwipeGesture(gesture)
                        if gesture.id != self.swipeid and self.state_string(gesture.state) == "STATE_START" and abs(swipe.direction.x) > 0.8:
                            # Get direction: -1 (Left) | +1 (Right)
                            if swipe.direction.x <= 0:
                                swipeDirection = -1
                            else:
                                swipeDirection = 1
                            # Little hack to only get one swipe instead to swipes from several fingers
                            if currentDir != swipeDirection and abs(swipe.speed) > 500:
                                print "Swipe id: %d, direction: %d, speed: %f, directionv: %s" % (
                                    gesture.id, swipeDirection, swipe.speed, swipe.direction)
                                self.cur_max = np.minimum(self.cur_max + swipeDirection * swipe.speed/5, self.prev_max)
                                self.cur_max = np.maximum(self.cur_max, 0)
                                print self.cur_max
                                if self.cur_max != self.prev_max:
                                    self.dcmimage.normalize(maximum=self.cur_max)
                            self.swipeid = gesture.id
                            currentDir = swipeDirection

        # Define the slice that will be displayed
        self.set_image()

        if not frame.hands.is_empty:
            # Get the interaction_box class
            interactionBox = frame.interaction_box

            # Iterate over the fingers detected and display them
            for i, pointable in enumerate(frame.pointables):
                normalizedPosition = interactionBox.normalize_point(pointable.tip_position)
                if(pointable.touch_distance > 0 and pointable.touch_zone != Leap.Pointable.ZONE_NONE):
                    color = (0, 255 - 255 * pointable.touch_distance, 0)
                elif(pointable.touch_distance <= 0):
                    color = (-255 * pointable.touch_distance, 0, 0)
                else:
                    color = (0, 0, 200)

                self.draw(normalizedPosition.x * self.dimX,
                          self.dimY - normalizedPosition.y * self.dimY,
                          10, color)

            # Conditions to translate the image
            cond_translation = [frame.hands[0].translation_probability(self.previous_frame) > 0.9,
                                3 <= np.size(frame.pointables) <4,
                                frame.pointables[0].touch_distance < 0,
                                frame.pointables[1].touch_distance < 0]

            # Conditions to scale the image
            cond_scaling = [frame.hands[0].scale_probability(self.previous_frame) > 0.9,
                            2 <= np.size(frame.pointables) < 3,
                            frame.pointables[0].touch_distance < 0,
                            frame.pointables[1].touch_distance < 0]

            # Conditions to rotate the image
            compteur = 0
            condition_rotation = False
            for pointable in frame.pointables:
                if pointable.touch_distance < 0:
                    compteur += 1
                    if compteur == 3:
                        condition_rotation = True
                        compteur = 0

            cond_rotation = [frame.hands[0].rotation_probability(self.previous_frame) > 0.9,
                             4 <= np.size(frame.pointables) <= 5,
                             condition_rotation]

            # Translate the image
            if all(cond_translation):
                translationX = frame.hands[0].translation(self.previous_frame)[0]
                translationY = frame.hands[0].translation(self.previous_frame)[1]
                if abs(translationX) > 10:
                    print "translation x %s %s" % (translationX,
                                                   frame.hands[0].translation_probability(self.previous_frame))
                    self.dcmimage.transX += translationX
                    self.dcmimage.transform()
                    self.previous_frame = frame
                if abs(translationY) > 10:
                    print "translation y %s %s" % (translationY,
                                                   frame.hands[0].translation_probability(self.previous_frame))
                    self.dcmimage.transY += -translationY
                    self.dcmimage.transform()
                    self.previous_frame = frame
            # Scale the image
            elif all(cond_scaling):
                scalingXY = frame.hands[0].scale_factor(self.previous_frame)
                if abs(scalingXY - 1) > 0.1:
                    print "scaling %s %s" % (scalingXY,
                                             frame.hands[0].scale_probability(self.previous_frame))
                    if scalingXY > 0:
                        self.dcmimage.scaling += scalingXY - 1
                    elif scalingXY < 0:
                        self.dcmimage.scaling += 1 - scalingXY
                    self.dcmimage.transform()
                    self.previous_frame = frame
            elif all(cond_rotation):
                step = np.rad2deg(frame.hands[0].rotation_axis(self.previous_frame).roll)
                if step>0:
                    step -= 90
                if abs(abs(self.prev_step) - abs(step)) > 5 and abs(step) > 5 and abs(step/3)<30:
                    self.prev_angle += step/3
                    print "rotation: %f %f" % (step/3, self.prev_angle)
                    self.dcmimage.theta = np.int(-self.prev_angle)
                    self.dcmimage.transform()
                    self.prev_step = step
                    self.previous_frame = frame
                    condition_rotation = False
                self.prev_step = step
            else:
                self.previous_frame = frame

        # Show final image
        cv2.imshow(self.name, self.opencv_img)

    def draw(self, x, y, radius, color):
        overlay = self.opencv_img.copy()
        cv2.circle(overlay, (int(x), int(y)), radius, color, -1)
        opacity = 0.50
        cv2.addWeighted(overlay, opacity,
                        self.opencv_img, 1-opacity, 0, self.opencv_img)

    def getslice(self, dcmslice):
        return int(np.max((np.min((self.dimZ - 1, dcmslice)), 0)))

    def set_image(self):
        tmp = self.dcmimage.display_img[self.current_slide]*255
        self.opencv_img[:, :, 0] = tmp.astype(np.uint8)
        self.opencv_img[:, :, 1] = self.opencv_img[:, :, 0]
        self.opencv_img[:, :, 2] = self.opencv_img[:, :, 0]

    def transform(self):
        pass

#############
# Main code #
#############

#load dicom images and initialize TransformDcm class
dcmimage = TransformDcm(whole_img)
dcmimage.initparam()

# Create a window
name_window = 'image'
cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)

# Leap controller
leap = Leap.Controller()
painter = TouchPointListener()
painter.add_dcm(dcmimage, name_window)

while(1):
    # keyboard action
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        # esc
        break

    leap.add_listener(painter)

leap.remove_listener(painter)
cv2.destroyAllWindows()
