#!/usr/bin/env python

"""UI for vision python binding. Opens an image and draws the vision results on it.

  Usage: ./vision_ui.py IMG_FILE_OR_FOLDER
"""

import os
import sys
print __file__
sys.path.append(os.path.join(os.path.dirname(__file__), "../python_binding/build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pyagent"))

import numpy as np
import cv2
import cPickle as pickle

import base64
import pyexiv2

import vision

from monitor.camera import CameraBase, AbstractEventHandler
from utils.camera_info import CameraInfo
from utils.camera_matrix import frame_robot_to_image_pixel

from yuv422 import convert_ycbcr_to_bgr


class VisionBindingUI():
    def __init__(self, window_name="VisionBindingUI"):
#         super(VisionBindingUI, self).__init__(None, window_name=window_name)
        self.vision = None
        self.processed_filename = None

    def process_file(self, filename):
#         print "processing file", filename
        self.bgr_img = cv2.imread(filename)
        h, w = self.bgr_img.shape[:2]
#         #show img array
#         print self.bgr_img

        if h < 240 or w < 320:
            print 'ignoring small image', (h, w), filename
            return

        self.ycrcb_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2YCR_CB)

        self.yuv422 = vision.YUV422(w, h)
        self.yuv422.loadFromNumpyYCrCbImage(self.ycrcb_img)
        

        if self.vision is None or self.vision.getHeight() != h or self.vision.getWidth() != w:
            # create vision if necessary
#             print 'w: ', w
#             print 'h: ', h
            self.vision = vision.Vision(w, h)
            
#             print 'getWidth(): ', self.vision.getWidth()
            
            self.vision.setConfigurationValue('scanlineRaster', 8)
        camera_matrix_utils = self.vision.getCameraMatrixUtils()

        metadata = pyexiv2.ImageMetadata(filename)
        metadata.read()


        self.vision.process(self.yuv422)
        self.processed_filename = filename
#         print '_visualize: ', self._draw_balls()
        ball_list = self._draw_balls()
#         print 'ball_list in process_file: ', ball_list
#         print 'processed vision'
        return ball_list

    def get_png_files_from_dir(self, source_dir):
        files = [os.path.join(source_dir, f) for f in os.listdir(source_dir)
                 if os.path.isfile(os.path.join(source_dir, f)) and f.endswith(".png")]
        files.sort()
        return files

    def process_folder(self, source_dir):
        png_files = self.get_png_files_from_dir(source_dir)
        n_png_files = len(png_files)
        print 'len(png_files): ', len(png_files)
        
#         for i in range(len(png_files)):
#             self.process_file(i)

        def text_save(filename, data):
            file = open(filename,'w')
            for i in range(len(data)):
                s = str(data[i]).replace('[','').replace(']','') #remove []
                s = s.replace("'",'').replace(',','|') +'\n'   #add '\n' for each line
                file.write(s)
            file.close()
            print("successful saved data") 

        def process_i_file(i):
            f = png_files[i]
#             print 'png_files name: ', png_files[i]
            ball_instance = self.process_file(f)[0]
            ball_instance.insert(0, png_files[i])
            return [ball_instance]
        ball_list_all = []
        for i in range(len(png_files)):
            ball_list = process_i_file(i)
#             print 'ball_list in range: ', ball_list
            ball_list_all += ball_list
#         print 'ball_list_all: ', ball_list_all
        text_save('test_prediction.txt', ball_list_all)
        


        
            
            
            
#         class EventHandler(AbstractEventHandler):
#             i = 0
#             def on_key(evt_h, key): #@NoSelf
#                 if chr(key) in [' '] or key == 83:  # RIGHT_ARROW
#                     evt_h.i += 1
#                     evt_h.i %= n_png_files
#                     process_i_file(evt_h.i)
#                 elif key == 81:  # LEFT_ARROW
#                     evt_h.i -= 1
#                     evt_h.i %= n_png_files
#                     process_i_file(evt_h.i)
#         self.add_event_handler(EventHandler())
        
#         process_i_file(0)
        


    def process_file_or_folder(self, pathname):
        if os.path.isfile(pathname):
            print "processing single image file:", pathname
            self.process_file(pathname)
        elif os.path.isdir(pathname):
            print "processing folder:", pathname
            self.process_folder(pathname)

    def get_image(self):
        return None, self.bgr_img

    
#     def _text_save(self, filename, data):
#         file = open(filename,'w')
#         for i in range(len(data)):
#             s = str(data[i]).replace('[','').replace(']','') #remove []
#             s = s.replace("'",'').replace(',','|') +'\n'   #add '\n' for each line
#             file.write(s)
#         file.close()
#         print("successful saved data") 

    
    def _draw_balls(self):
        balls = self.vision.getBallCalc().getBalls()
        ball_list = []
        for b in balls:
            c = b.getCenter()
            r = b.getRadius()
#             print 'b.getCenter(): ', c
#             print 'b.getRadius(): ', b.getRadius()
#             print 'b.getConfidence(): ', b.getConfidence()
            ball_list += [[c[0], c[1], r, b.getConfidence()]]            
        if len(ball_list) == 0:
            ball_list = [[0,0,0,0]]
        if len(ball_list) > 1 :
            ball_list = np.array(ball_list)
            index = np.argmax(ball_list[:, -1])
            ball_list = list(ball_list[index, :])
            ball_list = [ball_list]
#         print 'length of ball_list:', len(ball_list)
#         print 'ball_list: ', ball_list
        return ball_list
            
        
            #getConfidence probability whether a ball is
#             ball = (c[0], c[1], b.getRadius(), b.getConfidence() * 100, 0, 0, 0, 0)
#             self.draw_ball(ball)    
    
    def _draw_horizon(self):
        horizon = self.vision.getHorizonAndBodyContourCalc().getHorizon()
        if len(horizon) == 2:
            left, right = horizon
            cv2.line(self.img, (left[0], left[1]), (right[0], right[1]), (255, 0, 0), 2)

    def _draw_body_contours(self):
        body_contours = self.vision.getHorizonAndBodyContourCalc().getBodyContoursInImageSpace()
        for i, contour in enumerate(body_contours):
            colors = ((255, 0, 0), (255, 255, 0), (0, 122, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255))
            if len(contour) > 2:
                contour = np.array(np.int32([contour]))
                c = colors[i % len(colors)]
                cv2.polylines(self.img, contour, True, c, 2)

    def _draw_point_test(self):
        horizonAndBodyContourCalc = self.vision.getHorizonAndBodyContourCalc()
        scan_raster = 16
        for y in range(scan_raster // 2, self.h, scan_raster):
            for x in range(scan_raster // 2, self.w, scan_raster):
                point = np.array([x, y], dtype=np.int32)
                if horizonAndBodyContourCalc.isPointUnderHorizon(point) and horizonAndBodyContourCalc.isPointOutsideOfBody(point):
                    cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)

    def _draw_segments(self):
        for segment in self.vision.getEdgeSegmentCalc().getYColorSegments():
            start = segment.getStart()
            end = segment.getEnd()
            color = segment.getMedianColor()
            if segment.getType() == vision.SegmentType.UNKNOWN_TYPE:
                cv2.line(self.img, (start[0], start[1]), (end[0], end[1]), (0, 0, 0), 2)

            if segment.getType() == vision.SegmentType.LINE_TYPE:
                cv2.line(self.img, (start[0], start[1]), (end[0], end[1]), convert_ycbcr_to_bgr((color.getY(), color.getCb(), color.getCr())), 1)
                cv2.circle(self.img, (start[0], start[1]), 2, (0, 0, 255), -1)
                cv2.circle(self.img, (end[0], end[1]), 2, (0, 0, 255), -1)

                if segment.getType() == vision.SegmentType.LINE_TYPE:
                    start_gradient = segment.getStartGradient()
                    end_gradient = segment.getEndGradient()
                    scaleFactor = 10.0
                    cv2.line(self.img, (start[0], start[1]), (start[0] + int(round(start_gradient[0] / scaleFactor)), start[1] + int(round(start_gradient[1] / scaleFactor))), (255, 0, 0), 1)
                    cv2.line(self.img, (end[0], end[1]), (end[0] + int(round(end_gradient[0] / scaleFactor)), end[1] + int(round(end_gradient[1] / scaleFactor))), (255, 0, 0), 1)

    def _draw_field_border_candidate_points(self):
        for point in self.vision.getEdgeSegmentCalc().getFieldBorderCandidatePoints():
            cv2.circle(self.img, (point[0], point[1]), 2, (0, 255, 255), -1)

    def _draw_border(self):
        for point in self.vision.getFieldBorderCalc().getBorderPoints():
            cv2.circle(self.img, (point[0], point[1]), 3, (0, 0, 255), -1)
        border_polyline = self.vision.getFieldBorderCalc().getBorderPolyline()
        cv2.polylines(self.img, np.array([border_polyline]), False, (0, 0, 255), 1)

    def _draw_center_circle(self):
        center_circle = self.vision.getCenterCircleCalc().getCenterCircle()
        if center_circle is not None and self.camera_matrix is not None:
            p = [center_circle.getCenter()[0], center_circle.getCenter()[1], 0]
            p = frame_robot_to_image_pixel(self.camera_matrix, CameraInfo(), p)
            cv2.circle(self.img, (p[0], p[1]), 10, (0, 225, 0), -1)

    def _draw_lines(self):
        lines = self.vision.getLineCalc().getLines()
        for line in lines:
            cv2.line(self.img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 0), 2)

    def _draw_obstacles(self):
        obstacles = self.vision.getObstacleCalc().getObstacles()
        for obstacle in obstacles:
            p1 = obstacle.getPoint()
            p2 = p1 + obstacle.getSize()
            cv2.rectangle(self.img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2)

    def _draw_ball_candidate_regions(self):
        ball_candidate_regions = self.vision.getBallCalc().getBallCandidateRegions()
        for rect in ball_candidate_regions:
            p1 = rect.getPoint()
            p2 = p1 + rect.getSize()
            cv2.rectangle(self.img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 0), 2)
            cv2.rectangle(self.img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 0), 1)



    def _visualize(self):
        ret = self.draw_all(self.vision_perception)
        self._draw_horizon()
        self._draw_body_contours()
        #self._draw_point_test()
        self._draw_segments()
        self._draw_field_border_candidate_points()
        self._draw_border()
        self._draw_center_circle()
        self._draw_lines()
        self._draw_obstacles()
        self._draw_ball_candidate_regions()
        self._draw_balls()
        return ret

if __name__ == '__main__':
    if len(sys.argv) == 2:
#         print 'hello guys'
#         print 'sys.argv[1]: ', sys.argv[1]
        vision_ui = VisionBindingUI()
        vision_ui.process_file_or_folder(sys.argv[1])
#         vision_ui.start()
    else:
        print __doc__
        print "wrong number/type of arguments."
        sys.exit(1)
