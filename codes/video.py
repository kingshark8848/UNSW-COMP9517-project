import numpy as np
import cv2
import math
from collections import deque, defaultdict
import time

import imutils
# import config2 as config
import config as config


class VideoTable:
    def __init__(self, img_path, auto_find_corner=True):
        """
        :param img_path: video table image.
        :param auto_find_corner:
        """

        self.table_img_path = img_path

        # define the pool table color range
        self.hsv_color_lower = np.array(config.video_table_data["hsv_color_lower"], dtype="uint8")
        self.hsv_color_upper = np.array(config.video_table_data["hsv_color_upper"], dtype="uint8")

        self.table_corners = np.empty([4, 2], dtype="float32")  # order: Left-Down, Left-Top, Right-Top, Right-Down
        self.click_count = 0

        if auto_find_corner:
            self.auto_pool_table_detection()
        else:
            self.manual_table_corner_detection()

    def set_table_corners(self, table_corners):
        self.table_corners = table_corners

    def event_table_corner_click(self, event, x, y, flags, param):
        # print("hello")

        # if event == cv2.EVENT_LBUTTONDOWN:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.click_count < 4:
                self.table_corners[self.click_count, :] = [int(x), int(y)]
                print(x, y)
                self.click_count += 1
            else:
                print("please close image window")

    def manual_table_corner_detection(self):
        side_image = cv2.imread(self.table_img_path)

        print("Please select four corners from the pool table!")
        print("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
        cv2.imshow('Side-View', side_image)
        cv2.setMouseCallback('Side-View', self.event_table_corner_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_table_roi(self):
        table_img = cv2.imread(self.table_img_path)

        image_hsv = cv2.cvtColor(table_img, cv2.COLOR_BGR2HSV)

        # define the pool table color range
        table_lower = self.hsv_color_lower
        table_upper = self.hsv_color_upper

        mask_table_blue = cv2.inRange(image_hsv, table_lower, table_upper)
        image_roi = cv2.bitwise_and(table_img, table_img, mask=mask_table_blue)
        cv2.imshow("filtered pool table image", image_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def auto_pool_table_detection(self):
        side_image = cv2.imread(self.table_img_path)
        side_image = imutils.resize(side_image, width=800)

        side_image = cv2.medianBlur(side_image, 5)
        height, width, channels = side_image.shape

        # image = cv2.medianBlur(image,3)
        hsv_image = cv2.cvtColor(side_image, cv2.COLOR_BGR2HSV)

        # define the color range
        # lower = np.array([95, 100, 0], dtype="uint8")
        # upper = np.array([105, 255, 255], dtype="uint8")
        lower = np.array(config.video_table_data["hsv_color_lower"], dtype="uint8")
        upper = np.array(config.video_table_data["hsv_color_upper"], dtype="uint8")

        mask = cv2.inRange(hsv_image, lower, upper)

        # show the masked output
        output = cv2.bitwise_and(side_image, side_image, mask=mask)
        # cv2.imshow("maskedEffect", np.hstack([image, output]))
        cv2.imshow("maskedEffect", output)
        cv2.imshow("Mask", mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Set up the detector with default parameters.

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100
        # params.maxArea = 100

        detector = cv2.SimpleBlobDetector_create(params)

        #=================Detect blobs===================
        # Detect blobs.
        keypoints = detector.detect(output)
        # keypoints[0]: pt->center, size->radius
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(output, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #=======================/Detect blobs==============

        # draw convexHull
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        # print(contours)

        table_contours = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(table_contours)
        # test hull
        print(hull)
        print("=====hull end=====")

        cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)
        cv2.imshow("table hull", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        pure_table_contours_image = np.zeros((height, width, 1), np.uint8)
        cv2.drawContours(pure_table_contours_image, [hull], -1, 255, -1)  # thickness negative -> fill; positive means thick
        cv2.imshow("hull", pure_table_contours_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''Find Corner of Table'''
        test = cv2.approxPolyDP(hull, epsilon=40, closed=True)
        # print(test)

        test_img = np.zeros((height, width, 1), np.uint8)
        cv2.drawContours(test_img, [test], -1, 255, -1)  # thickness negative -> fill; positive means thick
        cv2.imshow("appox", test_img)

        points = test.tolist()
        # print(points)
        tmp_corner_points = []
        for p in points:
            print(p)
            x, y = p[0]
            cv2.circle(side_image, (x, y), 5, (0, 0, 255), -1)

            tmp_corner_points.append((x, y))

        print(tmp_corner_points)

        # odered from p0-3
        # determine which point is on left-up (near 0,0)
        min_distance = float("inf")
        tmp_p1 = None
        for p in tmp_corner_points:
            x, y = p
            distance = ((x - 0) ^ 2 + (y - 0) ^ 2)
            if distance < min_distance:
                min_distance = distance
                tmp_p1 = [x, y]

        self.table_corners[1, :] = tmp_p1

        for p in tmp_corner_points:
            if tmp_p1[0] == p[0] and tmp_p1[1] == p[1]:
                continue

            if p[0] - tmp_p1[0] > 200 and p[1] - tmp_p1[1] > 200:
                print("3:" + str(p))
                self.table_corners[3, :] = p
            elif p[0] - tmp_p1[0] > 200:
                print("2:" + str(p))
                self.table_corners[2, :] = p
            else:
                print("0:" + str(p))
                self.table_corners[0, :] = p

        print("auto corner:")
        print(self.table_corners)

        cv2.imshow("appox1", side_image)
        # cv2.imwrite("table_detection.png", side_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''
        # edges = cv2.Canny(table_image_gray,50,150,apertureSize = 3)
        # cv2.imshow("test", edges)
        # lines = cv2.HoughLinesP(edges, 1, np.pi/180,1)
        # print(lines)
        # hough_img = np.zeros((height,width,3), np.uint8)
        # for x1,y1,x2,y2 in lines[0]:
        # 	cv2.line(hough_img, (x1,y1), (x2,y2), (0,255,0), 100)

        # cv2.imshow("hough lines", hough_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        '''


class VideoBall:
    def __init__(self, ball_id, data=None):
        self.ball_id = ball_id
        self.hsv_color_lower = None
        self.hsv_color_lower = None

        if data:
            self.hsv_color_lower = data["hsv_color_lower"]
            self.hsv_color_upper = data["hsv_color_upper"]
            self.trajectory_color = data["trajectory_color"]


class Video:
    def __init__(self, video_file, resize=True):
        print("waiting...initialise...")

        self.video_file = video_file
        self.resize = resize
        # self.camera = cv2.VideoCapture(video_file)
        self.first_frame = None
        self.frame_count = 0

        self.balls = {}
        self.table = None
        self.tmp_ball_tracking_rec_for_trajectory = defaultdict(deque)  # limited space, tmp

        # unlimited space, for post analysis.
        # {"0":[(111,222),(111,333),None,...], "3":..., ...}
        self.ball_tracking_rec_complete = {}

        self.init_table()
        print("table detection completed")

        self.init_balls()
        print("sample ball data generated")

    def init_table(self):
        camera = cv2.VideoCapture(self.video_file)

        # use first frame to detect table
        (grabbed, self.first_frame) = camera.read()
        # cv2.imshow("first frame", self.first_frame)
        if self.resize:
            self.first_frame = imutils.resize(self.first_frame, width=800)

        cv2.imwrite(config.first_frame_save_path, self.first_frame)

        # self.table = VideoTable("test_data/check0.png")
        self.table = VideoTable(config.first_frame_save_path)

        if self.video_file == "test_data/game2/clip4.mp4":
            self.table.set_table_corners(np.array([[638, 335],[180, 336],[214, 94],[588, 95]], dtype="float32"))
        else:
            self.table.set_table_corners(np.array([[640, 455], [312, 446], [376, 11], [584, 11]], dtype="float32"))

        time.sleep(5)

        camera.release()

    def init_balls(self):
        for ball_id in config.balls_data:
            self.balls[ball_id] = VideoBall(ball_id, data=config.balls_data[ball_id])
            self.tmp_ball_tracking_rec_for_trajectory[ball_id] = deque(maxlen=64)
            self.ball_tracking_rec_complete[ball_id] = []

    def get_key_points_blob(self, frame):
        # frame = cv2.GaussianBlur(frame, (11, 11), 0)

        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        """
        Simple Blob Finding interesting area
        - use table color range, generate image that only display table area
        - apply blob detection, find key points
        - generate image that only display interesting area. (potential ball areas)
        """
        # define the pool table color range
        table_lower = self.table.hsv_color_lower
        table_upper = self.table.hsv_color_upper

        mask_table_blue = cv2.inRange(image_hsv, table_lower, table_upper)
        image_with_table_area = cv2.bitwise_and(frame, frame, mask=mask_table_blue)
        # cv2.imshow("filtered pool table image", image_with_table_area)
        # cv2.waitKey(0)

        # Setup SimpleBlobDetector parameters. Using blob to morphology detect potential pool balls
        params = cv2.SimpleBlobDetector_Params()
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2
        detector = cv2.SimpleBlobDetector_create(params)  # Initialize configured blob detector
        keypoints = detector.detect(image_with_table_area)

        # # debug
        # for p in keypoints:
        #     cv2.circle(image_with_table_area, (int(p.pt[0]), int(p.pt[1])), 2, (255,255,255),2)
        # cv2.imshow("filtered pool table image", image_with_table_area)
        # cv2.waitKey(0)

        return keypoints

    def get_img_with_roi(self, frame):
        """
        use simple blob to define ROI of an image - regions that potentially contain balls
        :param frame:
        :return: an image object with ROI
        """

        keypoints = self.get_key_points_blob(frame)

        # generate image with interesting area (areas which might have balls in it.)
        interesting_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
        for keypoint in keypoints:
            cv2.circle(interesting_mask, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size / 2),
                       1, thickness=-1)

        img_with_interesting_area = cv2.bitwise_and(frame, frame, mask=interesting_mask)
        img_with_interesting_area_hsv = cv2.cvtColor(img_with_interesting_area, cv2.COLOR_BGR2HSV)
        # cv2.imshow("img_with_interesting_area", img_with_interesting_area)
        # cv2.waitKey(0)

        # filter out table background
        table_lower = self.table.hsv_color_lower
        table_upper = self.table.hsv_color_upper
        mask_table = cv2.inRange(img_with_interesting_area_hsv, table_lower, table_upper)
        mask_table = cv2.bitwise_not(mask_table)
        img_with_interesting_area = cv2.bitwise_and(img_with_interesting_area, img_with_interesting_area,
                                                    mask=mask_table)
        # cv2.imshow("img_with_interesting_area (filter out table background)", img_with_interesting_area)
        # cv2.waitKey(0)
        return img_with_interesting_area

    def detect_one_ball_from_img_with_roi_v1(self, ball_id, img_with_roi, debug=False):
        """
        use color range to find ball
        :param debug:
        :param ball_id:
        :param img_with_roi:
        :return:
        """
        # print("ball_id", ball_id)

        """
        get previous ball status
        """
        one_ball_rec = self.tmp_ball_tracking_rec_for_trajectory[ball_id]
        prev_ball_center = None
        if len(one_ball_rec) > 0:
            prev_ball_center = one_ball_rec[0]

        """
        apply ball color range on interesting area
        """
        img_with_roi_hsv = cv2.cvtColor(img_with_roi, cv2.COLOR_BGR2HSV)

        ball_color_lower = np.array(self.balls[ball_id].hsv_color_lower, dtype="uint8")
        ball_color_upper = np.array(self.balls[ball_id].hsv_color_upper, dtype="uint8")
        mask_range = cv2.inRange(img_with_roi_hsv, ball_color_lower, ball_color_upper)
        mask_range = cv2.erode(mask_range, None, iterations=1)
        mask_range = cv2.dilate(mask_range, None, iterations=1)

        if debug:
            # only display specific ball
            image_with_ball = cv2.bitwise_and(img_with_roi, img_with_roi, mask=mask_range)
            cv2.imshow(ball_id, image_with_ball)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # # debug
        # if self.frame_count == 76:
        #     cv2.imwrite("img_with_roi76.png", img_with_roi)
        #     image_with_ball = cv2.bitwise_and(img_with_roi, img_with_roi, mask=mask_range)
        #     cv2.imwrite("image_with_ball76.png", image_with_ball)

        """
        find cnts, then ball center.
        pick up the nearest-max one to previous ball center
        """
        cnts = cv2.findContours(mask_range.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # # debug
        # if 0 < self.frame_count < 10 and ball_id == "2":
        #     if len(cnts) == 0:
        #         cv2.imshow("img_with_roi", img_with_roi)
        #         cv2.waitKey(0)
        #
        #     for c in cnts:
        #         print("c info====", cv2.minEnclosingCircle(c))
        #         ((x, y), radius) = cv2.minEnclosingCircle(c)
        #         img1 = img_with_roi.copy()
        #         cv2.circle(img1, (int(x),int(y)), 2,(0,0,255),2)
        #         cv2.imshow("debug", img1)
        #         cv2.waitKey(0)

        if len(cnts) <= 0:
            return None

        # print("ball", ball_id, "found!")
        while len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if prev_ball_center \
                    and imutils.get_distance_of_two_points(ball_center, prev_ball_center) > config.max_move_dis:
                # print("c",c)
                # cnts.remove(c)
                imutils.remove_array(cnts, c)
                continue

            # # test
            # if prev_ball_center and ball_id == "2" and imutils.get_distance_of_two_points(ball_center, prev_ball_center) > 20:
            #     print("help====", self.frame_count)

            return ball_center

        return None

    # todo
    def detect_one_ball_by_prev_status(self, ball_id, frame):
        """
        :param ball_id:
        :param frame:
        :return:
        """
        pass

    def real_time_tracking(self):
        camera = cv2.VideoCapture(self.video_file)
        fps = camera.get(cv2.CAP_PROP_FPS)
        print("Frames per second using camera.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        self.frame_count = 0

        while True:
            self.frame_count += 1

            # grab the current frame
            (grabbed, frame) = camera.read()
            # if self.frame_count < 200:
            #     continue

            # # debug
            # if self.frame_count == 366:
            #     cv2.imwrite("frame366.png", frame)

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if self.video_file and not grabbed:
                break

            if self.resize:
                # resize the frame, blur it, and convert it to the HSV color space
                frame = imutils.resize(frame, width=800)
            # frame = cv2.GaussianBlur(frame, (11, 11), 0)
            # frame = cv2.medianBlur(frame, 3)

            """
            detect ROI
            """
            img_with_roi = self.get_img_with_roi(frame)
            # cv2.imshow("img_with_interesting_area (filter out table background)", img_with_interesting_area)
            # cv2.waitKey(0)

            """
            detect each ball using ball color range
            """
            for ball_id in self.balls:

                ball_center = self.detect_one_ball_from_img_with_roi_v1(ball_id, img_with_roi)

                # if self.frame_count == 100 and ball_id == "8":
                #     print("hello!!!!!!")
                #     print(ball_center)

                if self.video_file == "test_data/game1/clip2.mp4" and ball_id=="2":
                    ball_center = None

                if ball_center:
                    self.tmp_ball_tracking_rec_for_trajectory[ball_id].appendleft(tuple(ball_center))
                    self.ball_tracking_rec_complete[ball_id].append(tuple(ball_center))
                else:
                    # # debug
                    # print("ball {} not found on frame {}!".format(ball_id, self.frame_count))

                    self.ball_tracking_rec_complete[ball_id].append(None)

                    # not_found_file_name = ball_id+"_not_found_frame_"+str(frame_count)+".png"
                    # cv2.imwrite(not_found_file_name, frame)

                """
                draw balls and trajectory
                """
                if ball_center:
                    # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 2)
                    cv2.circle(frame, ball_center, config.video_ball_radius, self.balls[ball_id].trajectory_color, 2)

                    # trajectory
                    tmp_ball_record = self.tmp_ball_tracking_rec_for_trajectory[ball_id]
                    # print(tmp_ball_record)
                    for i in range(1, len(tmp_ball_record)):
                        if tmp_ball_record[i - 1] is None or tmp_ball_record[i] is None:
                            # print("Doesn't draw the line for current frame!")
                            continue

                        # thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        thickness = int(np.sqrt(64 / float(i + 1)) * 1.2)

                        # print(thickness)
                        # print(tmp_ball_record[i - 1])
                        # print(tmp_ball_record[i])

                        cv2.line(frame, tmp_ball_record[i - 1], tmp_ball_record[i], self.balls[ball_id].trajectory_color, thickness)

            """
            after check every ball, show processed frame
            """
            # print("tracking record:", self.ball_tracking_rec_for_real_time)

            # show the frame to our screen
            cv2.imshow("Pool ball tracking frame", frame)
            # print(self.ball_tracking_rec_for_real_time)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                cv2.destroyAllWindows()

                # cv2.imwrite("video_tracking.png", frame)

                break

        camera.release()

    def draw_simple_trajectory(self):
        img = self.first_frame

        for ball_id in self.ball_tracking_rec_complete:
            one_ball_rec = self.ball_tracking_rec_complete[ball_id]
            one_ball_rec = [x for x in one_ball_rec if x is not None]

            for i in range(len(one_ball_rec) - 1):
                # print(one_ball_rec[i], one_ball_rec[i+1])

                cv2.circle(img, one_ball_rec[i], 3, self.balls[ball_id].trajectory_color, -1)
                cv2.line(img, one_ball_rec[i], one_ball_rec[i + 1], self.balls[ball_id].trajectory_color, 1)

        # print("hello")
        cv2.imshow("draw_simple_trajectory", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _calc_euclidean_dist(bgr_value, benchmark_rgb_value):
        dist = 0.0
        for i in range(3):
            dist += math.pow(bgr_value[i] - benchmark_rgb_value[i], 2)
        return dist

    # ==========================meanshift test========================
    def test_mean_shift(self):
        camera = cv2.VideoCapture(self.video_file)

        # fps = camera.get(cv2.CAP_PROP_FPS)
        # print ("Frames per second using camera.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        self.frame_count = 0

        # first frame to determine the original ball (ball 0) and it's tracking window
        (grabbed, frame) = camera.read()
        if self.resize:
            frame = imutils.resize(frame, width=800)

        img_with_roi = self.get_img_with_roi(frame)
        ball_center = self.detect_one_ball_from_img_with_roi_v1("0", img_with_roi)
        print(ball_center)
        c, r, w, h = ball_center[0] - 5, ball_center[1] - 5, 10, 10  # rectangle of ball area
        track_window = (c, r, w, h)

        cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)
        cv2.imshow('test', frame)
        cv2.waitKey(0)

        # Create mask and normalized histogram
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ball_color_lower = np.array(self.balls["0"].hsv_color_lower, dtype="uint8")
        # ball_color_upper = np.array(self.balls["0"].hsv_color_upper, dtype="uint8")
        # mask = cv2.inRange(hsv_roi, ball_color_lower, ball_color_upper)

        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1)

        # process each frame
        while True:
            ret, frame = camera.read()
            if self.resize:
                frame = imutils.resize(frame, width=800)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    # ==========================give up===============================
    # todo
    def is_table_background_color(self, hsv_val):
        h, s, v = hsv_val
        h_lower, h_upper = self.table.hsv_color_lower[0], self.table.hsv_color_upper[0]
        s_lower, s_upper = self.table.hsv_color_lower[1], self.table.hsv_color_upper[1]
        v_lower, v_upper = self.table.hsv_color_lower[2], self.table.hsv_color_upper[2]

        if (h in range(h_lower, h_upper)
            and s in range(s_lower, s_upper)
            and v in range(v_lower, v_upper)):
            return True
        else:
            return False

    # todo
    def analyze_ball_region_color(self, frame, x_center, y_center):
        # hsv img
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # create ball roi window
        # use ball_radius
        ball_radius = config.video_ball_radius
        # c->column->x, r->row->y
        c, r, w, h = x_center - ball_radius, y_center - ball_radius, ball_radius, ball_radius

        # traverse each point in roi get:
        # - hue color histogram. (exclude: white, black)
        # - white count
        pixel_count = 0
        hue_val_sum = 0
        hue_count = 0
        hue_color_hist = [0] * 180
        white_pixel_count = 0
        black_pixel_count = 0
        for y in range(r, r + h):
            for x in range(c, c + w):
                # outside of circle, should not count as ball region pixel
                if imutils.get_distance_of_two_points((x, y), (x_center, y_center)) > ball_radius:
                    continue

                pixel_hsv = frame_hsv[y, x]
                pixel_count += 1

                # check if it's table background pixel
                if self.is_table_background_color(pixel_hsv):
                    continue

                # check if it's black pixel
                if pixel_hsv[2] <= config.black_v_max:
                    black_pixel_count += 1
                    continue

                # check if it's white pixel
                if pixel_hsv[1] <= config.white_s_max:
                    white_pixel_count += 1
                    continue

                # count as color pixel.
                hue_color_hist[pixel_hsv[0]] += 1
                hue_val_sum += pixel_hsv[0]
                hue_count += 1

        print("=====")
        print("center", (x_center, y_center))
        print("pixel_count", pixel_count)
        print("white_pixel_count:", white_pixel_count)
        print("black_pixel_count:", black_pixel_count)
        print("color_hist:", hue_color_hist)
        print("hue_count:", hue_count)

        # return {"white_pixel_count:": white_pixel_count,
        #         "black_pixel_count:": black_pixel_count,
        #         "color_hist:": hue_color_hist}

        # calc primary color count
        primary_pixel_count = 0
        if hue_count > 0:
            mean_hue = int(hue_val_sum / hue_count)
            print("mean_hue:", mean_hue)
            mean_hue_variance = config.mean_hue_variance
            for hue_val in range(mean_hue - mean_hue_variance, mean_hue + mean_hue_variance):
                primary_pixel_count += hue_color_hist[hue_val]

        # If the number of black pixels is greater than the number of primary colored pixels,
        # the number of black pixels is used as the number of primary pixels.
        if black_pixel_count > primary_pixel_count:
            primary_pixel_count = black_pixel_count

        ball_pb = (primary_pixel_count + white_pixel_count) / float(pixel_count)
        print("ball_pb:", ball_pb)

        return ball_pb

    # todo
    def find_ball_candinates(self, frame):
        ball_candinates_dic = {}

        keypoints = self.get_key_points_blob(frame=frame)
        # for each ROI
        for key_point in keypoints:
            x_center, y_center, radius = int(key_point.pt[0]), int(key_point.pt[1]), int(key_point.size / 2)

            # ROI window
            # c->column->x, r->row->y
            c, r, w, h = x_center - radius, y_center - radius, radius, radius
            for y in range(r, r + h):
                for x in range(c, c + w):
                    ball_pb = self.analyze_ball_region_color(frame, x, y)
                    ball_candinates_dic[(x, y)] = ball_pb
                    # cv2.circle(frame, (x, y), config.video_ball_radius, (255, 0, 0), 2)

        ball_candinates_ordered_list = sorted(ball_candinates_dic.items(), key=lambda d: d[1], reverse=True)
        print(ball_candinates_ordered_list)

        # cv2.imshow("test", frame)

        return ball_candinates_ordered_list

    # todo
    def select_balls_from_candinates(self, ball_candinates_ordered_list):
        final_ball_list = []
        while len(ball_candinates_ordered_list) != 0 and len(final_ball_list) < 16:
            leader_ball = ball_candinates_ordered_list.pop(0)[0]

            if imutils.is_point_within_radius_of_point_list(leader_ball, final_ball_list, config.video_ball_radius):
                continue

            final_ball_list.append(leader_ball)

        return final_ball_list


if __name__ == '__main__':
    # video_table = VideoTable("test_data/check0.png")
    # print(video_table.field_corners)



    # # test1
    # img = cv2.imread("test_data/game1/4.png")
    # ball_list = myvideo1.find_ball_candinates(img)
    # final_ball_list = myvideo1.select_balls_from_candinates(ball_list)
    # print("final_ball_list", final_ball_list)
    #
    # for p in final_ball_list:
    #     cv2.circle(img, p, config.video_ball_radius, (255,0,0), 2)
    #
    # cv2.imshow("effect", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # test2
    # video_file = "test_data/game1/video/1.mp4"
    # myvideo1 = Video(video_file)
    # img = cv2.imread("test_data/game1/frame100.png")
    # img = imutils.resize(img, width=800)
    # # img = cv2.imread("test_data/game1/balls/all_balls.png")
    # # img = cv2.imread("img_with_roi76.png")
    # roi = myvideo1.get_img_with_roi(img)
    # cv2.imshow("roi", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # img_with_roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #
    # ball_color_lower = np.array(myvideo1.balls["6"].hsv_color_lower, dtype="uint8")
    # ball_color_upper = np.array(myvideo1.balls["6"].hsv_color_upper, dtype="uint8")
    # mask_range = cv2.inRange(img_with_roi_hsv, ball_color_lower, ball_color_upper)
    #
    # mask_range = cv2.erode(mask_range, None, iterations=1)
    # mask_range = cv2.dilate(mask_range, None, iterations=1)
    #
    # image_with_ball = cv2.bitwise_and(roi, roi, mask=mask_range)
    # cv2.imshow("6", image_with_ball)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cnts = cv2.findContours(mask_range.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # print(len(cnts))

    # test3
    video_file = "test_data/game2/clip5.mp4"
    myvideo2 = Video(video_file)
    myvideo2.table.show_table_roi()

    img = cv2.imread("test_data/game2/frame366.png")
    img = imutils.resize(img, width=800)
    # img = cv2.imread("test_data/game1/balls/all_balls.png")
    # img = cv2.imread("img_with_roi76.png")
    roi = myvideo2.get_img_with_roi(img)
    cv2.imshow("roi", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_with_roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    ball_color_lower = np.array(myvideo2.balls["4"].hsv_color_lower, dtype="uint8")
    ball_color_upper = np.array(myvideo2.balls["4"].hsv_color_upper, dtype="uint8")
    mask_range = cv2.inRange(img_with_roi_hsv, ball_color_lower, ball_color_upper)

    mask_range = cv2.erode(mask_range, None, iterations=1)
    mask_range = cv2.dilate(mask_range, None, iterations=1)

    image_with_ball = cv2.bitwise_and(roi, roi, mask=mask_range)
    cv2.imshow("4", image_with_ball)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cnts = cv2.findContours(mask_range.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    print(len(cnts))