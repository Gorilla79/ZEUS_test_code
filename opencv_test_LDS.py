import random
import os
import cv2
import time
import darknet
import argparse  
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import queue 
import Queue
import socket
import subprocess
import math

#################
'''
path = 'D:/background/seria_prj/zeus_detect_multibox'
#path = 'D:/background/seria_prj/qr_measure'
img_name = '02.jpg'
full_path = path + '/' +img_name
 
img_array = np.fromfile(full_path, np.uint8)
'''
# webcam = True
# #webcam = False
# #path = '01.jpg'
# cap = cv2.VideoCapture(0)
# cap.set(10, 160)
# cap.set(3, 1920)
# cap.set(4, 1080)
# scale = 3
# wP = 297 * scale
# hP = 210 * scale

'''
kernel = np.ones((5, 5), np.uint8)
dev_SN = []
dist_to_object = 0
color_cnt = 0
box_cnt = 0
detect_stack = bool
x_m, y_m = 0, 0
xmin, ymin = 0, 0
clicked_points = []
clone = None
'''

def send_socket_xandy(soc_x, soc_y):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("192.168.0.23",9000))
    client_socket.sendall(str(soc_x).encode())
    client_socket.sendall(str(soc_y).encode())

    client_socket.close()

'''
def send_socket_y(soc_y):
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("192.168.0.23",9000))
    client_socket.sendall(str(soc_y).encode())

    client_socket.close()
'''

def send_socket_z(soc_z):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(("192.168.0.23", 9000))
        client_socket.sendall(str(soc_z).encode())
    except ConnectionRefusedError as e:
        print("Connection refused:", e)
    finally:
        client_socket.close()

def recive_socket():
    global recive_socket_data
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #server_socket.bind(("192.168.1.21",9999))
    server_socket.bind(("192.168.0.42",9999))
    server_socket.listen(1)
    conn, addr = server_socket.accept()
    
    recive_socket_data = conn.recv(1024)
    recive_socket_data = recive_socket_data.decode()
    server_socket.close()
    
    return recive_socket_data

#'''
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="C:/Users/SUBIN/Desktop/yolov3/darknet/temp_build/Debug/box_1_2_3_data/backup/box_1_2_3_last.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action="store_true",
                        help="window inference display. For headless systems")
    parser.add_argument("--ext_output", action="store_true",
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="C:/Users/SUBIN/Desktop/yolov3/darknet/temp_build/Debug/cfg/box_1_2_3.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="C:/Users/SUBIN/Desktop/yolov3/darknet/temp_build/Debug/box_1_2_3_data/box_1_2_3.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()
'''
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="C:/Users/pooom/Desktop/yolov3/darknet/temp_data/Debug/box_1_2_3_data/backup/box_1_2_3_last.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action="store_true",
                        help="window inference display. For headless systems")
    parser.add_argument("--ext_output", action="store_true",
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="C:/Users/pooom/Desktop/yolov3/darknet/temp_data/Debug/cfg/box_1_2_3.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="C:/Users/pooom/Desktop/yolov3/darknet/temp_data/Debug/box_1_2_3_data/box_1_2_3.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()
'''
'''
def MouseLeftClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN :
        print('왼쪽 마우스 클릭 했을 때 좌표 : ', x, y)
        clicked_points.append((x, y))

    # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
    for point in clicked_points:
        cv2.circle(image, (point[0], point[1]), 2, (0, 255, 255), -1)
'''


def MouseLeftClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        print('왼쪽 마우스 클릭 했을 때 좌표 : ', x, y)



def get_device_sequence():
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A"]

    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    devs = []
    for dev in devices:
        time.sleep(1)
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                dev_SN_int = dev.get_info(rs.camera_info.serial_number)                
                dev_SN_int = int(dev_SN_int)
                dev_SN.append(dev_SN_int)
                print(dev.get_info(rs.camera_info.serial_number))
    return dev_SN

def nothing(x):
    pass
'''
def convex(PATH):
    global angle, rotated_direction
    img = PATH
    img1 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(imgray, alpa, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnt = max(contours, key=cv2.contourArea)
    cnt = None
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        
        # 컨투어가 유효한지 확인
        if len(cnt) >= 3:  # 최소한 3개의 점이 필요
            rect = cv2.minAreaRect(cnt)
            # 나머지 작업 수행
        else:
            print("Invalid contour detected")
    else:
        print("No contours found")
        
    x, y, w, h = cv2.boundingRect(cnt)


    roi_image = img[int(y):int(y + h), int(x):int(x + w)]
    cv2.imshow("#2 roi_image", roi_image)
    #cv2.imshow("#3 imgray", imgray)
    # print('rect',rect)
    angle = rect[2:3]
    angle = angle[0] 
    
    # print('angle :',angle)
    # c mean center
    (hh, ww) = img.shape[:2]
    # print('ww',ww)
    # print('hh',hh)

    (cX, cY) = (ww / 2, hh / 2)
    #cv2.putText(roi_image, 'angle' + str(angle), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if abs(angle) > 45:
        M = cv2.getRotationMatrix2D((cX, cY), 90 + angle, 1.0)
        #print('rotated_angle : ', 90 + angle)
    else:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        #print('rotated_angle : ', angle)

    #rotated = cv2.warpAffine(img1, M, (x + ww, y + hh))
    #cv2.imshow("#2 Rotated ", rotated)

    box = cv2.boxPoints(rect)
    box = np.intp(box)          # box = np.int0(box) [numpy 버전 1.24 이하 일 때]
    # print('box',box)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
    # print('cnt',cnt)
    check = cv2.isContourConvex(cnt)
    # print('check',check)
    
    # # 회전 방향
    # if (box[0][0] > box[2][0]):
    #     rotated_direction = CCW
    # else:
    #     rotated_direction = CW

    if not check:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img1, [hull], 0, (0, 255, 0), 3)
        cv2.imshow('#2 convexhull', img1)
    
    cv2.imshow('#2 contour', img)

    return angle #, rotated_direction
'''
def Colorsearch(img_hsv, lower_color, upper_color):
    global color_cnt, angle, distant_to_object, write_stack

    img_mask = cv2.inRange(img_hsv, lower_color, upper_color)

    erosion_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel) # 침식

    ############
    # contours, _ = cv2.findContours(alpa, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if detection:
        contours, _ = cv2.findContours(erosion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print('box',box)
        cv2.drawContours(erosion_mask, [box], 0, (0, 255, 0), 3)

    ############

    cv2.imshow('#2 erosion_mask', erosion_mask)
    '''
    img_result_color = cv2.bitwise_and(color_image, color_image, mask=erosion_mask)

    if erosion_mask[x_m, y_m] == 0:
        color_cnt = 0

    else:
        color_cnt = color_cnt + 1

    #############################################################################################
    ################################             roi set            #############################

    ROI_frame_color = img_result_color
    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 2500 and area < 307200:
            if color_cnt == 100:
                distant_to_object = aligned_depth_frame.get_distance(x, y) 
                distant_to_object = round(distant_to_object * 100, 3)
                
                cv2.circle(color_image, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(color_image, str(distant_to_object), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if color_cnt > 100:
                distant_to_object = aligned_depth_frame.get_distance(x, y) 
                distant_to_object = round(distant_to_object * 100, 3)
                cv2.circle(color_image, (centerX, centerY), 10, (216, 168, 74), 10)
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (216, 168, 74))
                
                #cv2.circle(color_image, (400, 300), 3, (0, 0, 255), -1)
                cv2.putText(color_image, str(distant_to_object), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
'''
    return 0
    ################################             roi set            #############################
    #############################################################################################


# trackbar setting
cv2.namedWindow('threshold Controller')
cv2.createTrackbar('value threshold', 'threshold Controller', 0, 255, nothing)
cv2.setTrackbarPos('value threshold', 'threshold Controller', 170)

cv2.namedWindow('Color - Depth Controller')
cv2.createTrackbar('alpha_value', 'Color - Depth Controller', 0, 255, nothing)
cv2.setTrackbarPos('alpha_value', 'Color - Depth Controller', 117)

def main():
    global detection
    time.sleep(1)
    
    # get_device_sequence()
    #dev_SN.sort()
    #first_device = dev_SN
    #print('first_device', first_device)
    
    # Create a pipeline
    print('set #1 camera')
    pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
    config = rs.config()
## camera serial number
    #first_device = str(first_device)
    #config.enable_device(first_device)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
    profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
    #depth_sensor = profile.get_device().first_depth_sensor()
    #depth_scale = depth_sensor.get_depth_scale()

#print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    print('clear #1 camera set')
    
    time.sleep(0.5)
    
    parser()
    args = parser()
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1)
    
    
    width_yolo = 640
    height_yolo = 480 
    darknet_image = darknet.make_image(width_yolo, height_yolo, 3)
    
    

    try:
        while True:
            alpa = cv2.getTrackbarPos('value threshold', 'threshold Controller')
            betta = cv2.getTrackbarPos('alpha_value', 'Color - Depth Controller')
            frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
            #print("frame out")
            aligned_frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            width_depth = depth_frame.get_width()
            height_depth = depth_frame.get_height()

        # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            colorizer = rs.colorizer()
            hole_filling = rs.hole_filling_filter()
            
            filled_depth = hole_filling.process(aligned_depth_frame)
            colorizer_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
        # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=betta / 255), cv2.COLORMAP_JET)
            img_hsv = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2HSV)
            
        # Yolo Size
            frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            frame_resized = cv2.resize(frame_rgb, (width_yolo, height_yolo), interpolation=cv2.INTER_LINEAR)
            depth_colormap_resized = cv2.resize(depth_colormap, (width_yolo, height_yolo), interpolation=cv2.INTER_LINEAR)
            
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.7)

            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            '''
            if detections:
                detect_stack = True
                name = []
                name = detections[0]
                for detection in name:

                    class_name = name[0]
            '''

            if detections:
                detect_stack = True

                for detection in detections:
                    class_name = detection[0]


                    if class_name == 'BOX':
                        box_cnt = box_cnt + 1
                        bbox = detection[2]
                        xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
                        x_m = (xmin + xmax) / 2
                        y_m = (ymin + ymax) / 2

                        x_diff = (x_m - 309)
                        y_diff = (y_m - 391)
                        x_cm = x_diff * 0.0987
                        y_cm = y_diff * 0.0987

                        send_socket_xandy(x_cm, y_cm)




                        #print('')
                        #deep_roi_x_min = int(object_boxes_x)
                        #deep_roi_x_max = int(object_boxes_x + object_boxes_width)
                        #deep_roi_y_min = int(object_boxes_y)
                        #deep_roi_y_max = int(object_boxes_y + object_boxes_height)

                        deep_ROI_frame_color = image[ymin: ymax, xmin: xmax]
                        #deep_ROI_hsv_color = img_hsv_resized[ymin: ymax, xmin: xmax]
                        deep_ROI_depth_color = depth_colormap_resized[ymin: ymax, xmin: xmax]

                        if deep_ROI_frame_color is None :
                            pass

                        else :
                            #print('')
                            ####cv2.imshow("deep_ROI_frame_color", deep_ROI_frame_color)
                            #cv2.imshow("deep_ROI_hsv_color", deep_ROI_hsv_color)
                            #cv2.imshow("deep_ROI_depth_color", deep_ROI_depth_color)
                            fps = int(1/(time.time() - prev_time))
                            #ROI_depth = deep_ROI_depth_color.get_depth_frame()

                        dist_to_object = aligned_depth_frame.get_distance(int(x_m), int(y_m))
                        dist_to_object = round(dist_to_object * 100, 3)
                        #move_to_x = int(width_yolo/2) - (int((xmax - xmin)/2) + xmin)

                        print('###################################################')
                        print('')
                        print('class_name : ', class_name, 'accuracy : ', detection[1])
                        print('object_boxes_xmin : ', xmin, ', object_boxes_xmax : ', xmax)
                        print('object_boxes_ymin : ', ymin, ', object_boxes_ymax : ', ymax)
                        print("FPS: {}".format(fps))

                        # if abs(angle) > 45:
                        #     print('rotated_angle : ', 90 + angle, ', rotated_direction : ', rotated_direction)
                        # else :
                        #     print('rotated_angle : ', angle, ', rotated_direction : ', rotated_direction)

                        print('')
                        print('###################################################')
                        ## check point ##
                        cv2.circle(image, (int(xmin), int(ymin)), 5, (0, 255, 255), -1)  
                        cv2.circle(image, (int(xmax), int(ymax)), 5, (0, 255, 255), -1)  
                        #cv2.circle(image, (int(x_m), int(y_m)), 3, (0, 0, 255), -1)  
                        #cv2.circle(image, (int(x_m), int(y_m)), 3, (0, 0, 255), -1)  
                        ###############
                        cv2.circle(image, (int(x_m), int(y_m)), 3, (0, 0, 255), -1)      
                        cv2.putText(image, (str(dist_to_object)), (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(image, (str(x_cm) + str(y_cm)), (xmin, ymin - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        #########################  x 좌표 그대로, y 좌표 변환 -> 좌하단 좌표계 변환  #########################
                        

                    '''
                    try:
                        if box_cnt == 10 :
                            dist_to_object = aligned_depth_frame.get_distance(int(x_m + xmin), int(y_m + ymin))
                            dist_to_object = round(dist_to_object, 3)
                            #move_to_x = int(width_yolo/2) - (int((xmax - xmin)/2) + xmin)

                            print('###################################################')
                            print('')
                            print('class_name : ', class_name, 'accuracy : ', name[1])
                            print('object_boxes_xmin : ', xmin, ', object_boxes_xmax : ', xmax)
                            print('object_boxes_ymin : ', ymin, ', object_boxes_ymax : ', ymax)
                            print("FPS: {}".format(fps))

                            # if abs(angle) > 45:
                            #     print('rotated_angle : ', 90 + angle, ', rotated_direction : ', rotated_direction)
                            # else :
                            #     print('rotated_angle : ', angle, ', rotated_direction : ', rotated_direction)

                            print('')
                            print('###################################################')

                            cv2.circle(image, (int(x_m), int(y_m)), 3, (0, 0, 255), 2)      #  
                            # cv2.line(image, (int(x_m + (x_diff/10)), int(y_m + (y_diff/10))), (int(x_m + (x_diff/2)), int(y_m + (y_diff/2))), (255, 0, 0), 2)
                            # cv2.line(image, (int(x_m - (x_diff/10)), int(y_m - (y_diff/10))), (int(x_m - (x_diff/2)), int(y_m - (y_diff/2))), (255, 0, 0), 2)
                            #cv2.circle(img2, [obj[1]], True, (0, 255, 55), 2)   # 
                            #cv2.circle(img2, [obj[2]], True, (255, 0, 0), 2)    # 
                            cv2.putText(image, (str(dist_to_object)), (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                            #cv2.circle(color_image, (centerX, centerY), 10, (216, 168, 74), 10)
                            #cv2.rectangle(color_image, (x, y), (x + width, y + height), (216, 168, 74))
                            
                            #cv2.circle(color_image, (400, 300), 3, (0, 0, 255), -1)
                            #cv2.putText(color_image, str(distant_to_object), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                            box_cnt = 0
                        pass
                        
                    except RuntimeError as err:
                        print(err)
                        pass
                    '''
            else :
                detect_stack = False
                pass

            

            #cv2.imshow('#2 color_image', color_image)
            cv2.imshow('#2 depth_colormap', depth_colormap)
            
            ROI_depth_colormap = img_hsv[300, 400]
            one_pixel = np.uint8([ROI_depth_colormap])
            hsv = one_pixel[0][0]
            
            hsv_value = hsv / 10
            hsv_value = int(hsv_value) * 10

            if hsv_value < 10:
                lower_color = np.array([hsv_value - 10 + 180, 30, 30])
                upper_color = np.array([180, 255, 255])

            elif hsv_value > 170:
                lower_color = np.array([hsv_value, 30, 30])
                upper_color = np.array([180, 255, 255])
            else:
                lower_color = np.array([hsv_value, 30, 30])
                upper_color = np.array([hsv_value + 10, 255, 255])
            
            img_mask = cv2.inRange(img_hsv, lower_color, upper_color)

            erosion_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel) # 침식



            ############
            # contours, _ = cv2.findContours(alpa, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
            contours, _ = cv2.findContours(erosion_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:  # 윤곽선이 존재하는 경우
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #print(box)
                #cv2.drawContours(erosion_mask, [box], -1, (0, 255, 0), 3)
                cv2.polylines(image, [box], True, (0, 0, 255), 3)

                x0 = box[0, 0]
                x1 = box[1, 0]
                x2 = box[2, 0]
                y0 = box[0, 1]
                y1 = box[1, 1]
                y2 = box[2, 1]
                awidth_1 = x1 - x0
                aheight_1 = y1 - y0
                awidth_2 = x2 - x1
                aheight_2 = y2 - y1
                len_a = math.sqrt((awidth_1 ** 2) + (aheight_1 ** 2))
                len_b = math.sqrt((awidth_2 ** 2) + (aheight_2 ** 2))

                if len_a > len_b:
                    ang_degree = math.atan((aheight_2) / (awidth_2)) * 180 / 3.141592

                else:
                    ang_degree = math.atan((aheight_1) / (awidth_1)) * 180 / 3.141592
                ang_degree = round(ang_degree, 2)
                cv2.putText(image, 'angle =' + str(ang_degree), (xmin + 40, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            else:
                print("No contours found in erosion_mask.")

            cv2.imshow('#2 erosion_mask', erosion_mask)

            #angle = convex(image)
            #dist_to_object_z = Colorsearch(img_hsv, lower_color, upper_color)
        # Display the resulting frame
            cv2.imshow('frame', image)
            
            cv2.setMouseCallback('frame', MouseLeftClick)
            
            cv2.imshow('colorizer_depth', colorizer_depth)
            
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            if count >= 3:
                count = 0
                recive_socket_data = recive_socket()
                if recive_socket_data == 'h':
                    recive_socket_data = recive_socket()
                    if recive_socket_data == 'g':
                        break
    finally:
        pipeline.stop()
        
    # if webcam:success, img = cap.read()
    # else: img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
   
    
    '''
    img, conts = utlis_ga.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utlis_ga.warpImg(img, biggest, wP, hP)
        img2, conts2 = utlis_ga.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw = False)
        
        #print('conts2', conts2)
        
        # points = np.array(conts2[2])
        # x1 = points[0][0][0]
        # x3 = points[2][0][0]
        # print('x1', x1)
        # print('x3', x3)
        
        #first_point = biggest[0][0]
        #third_point = biggest[2][0]
        #cv2.rectangle(img, first_point, third_point, (0,0,255), 3)
        #img_filter = utlis_ga.dstContours(img) 
        #img_filter = cv2.resize(img, (0, 0), None, 0.2, 0.2)
        #cv2.imshow('filter', img_filter)
        
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(img2, [obj[2]], True, (0, 255, 0), 2)
                #print(obj[0])
                points = np.array(obj[2])
                x1 = int(points[0][0][0])
                x2 = int(points[1][0][0])
                x3 = int(points[2][0][0])
                x_m = (x3 + x1) / 2
                y1 = int(points[0][0][1])
                y2 = int(points[1][0][1])
                y3 = int(points[2][0][1])
                y_m = (y3 + y1) / 2
                
                x_diff_1 = x2 - x1
                x_diff_2 = x3 - x2
                y_diff_1 = y2 - y1
                y_diff_2 = y3 - y2
                len_a = math.sqrt(abs(x_diff_1)**2 + abs(y_diff_1)**2) 
                len_b = math.sqrt(abs(x_diff_2)**2 + abs(y_diff_2)**2)
                
                if len_a > len_b:
                    x_diff = x_diff_2
                    y_diff = y_diff_2
                    if x_diff == 0 or y_diff == 0:
                        # x_diff와 y_diff가 모두 0인 경우
                        angle = 0  # 또는 다른 적절한 값으로 설정
                    else:
                        # x_diff와 y_diff가 모두 0이 아닌 경우, 각도 계산
                        angle = round(math.atan(y_diff / x_diff) * 180 / math.pi, 2)
                    #print('angle = ', angle)
                    text = ('angle = ')
                    cv2.putText(img2, (text + str(angle)), (700, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    x_diff = x_diff_1
                    y_diff = y_diff_1
                    if x_diff == 0 or y_diff == 0:
                        # x_diff와 y_diff가 모두 0인 경우
                        angle = 0  # 또는 다른 적절한 값으로 설정
                    else:
                        # x_diff와 y_diff가 모두 0이 아닌 경우, 각도 계산
                        angle = round(math.atan(y_diff / x_diff) * 180 / math.pi, 2)    
                    #print('angle = ', angle)   
                    text = ('angle = ')
                    cv2.putText(img2, (text + str(angle)), (700, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                
                
                
                #print('y1', y1)
                #print('y3', y3)
                
                
                cv2.circle(img2, (int(x_m), int(y_m)), 3, (0, 0, 255), 2)      #  
                cv2.line(img2, (int(x_m + (x_diff/10)), int(y_m + (y_diff/10))), (int(x_m + (x_diff/2)), int(y_m + (y_diff/2))), (255, 0, 0), 2)
                cv2.line(img2, (int(x_m - (x_diff/10)), int(y_m - (y_diff/10))), (int(x_m - (x_diff/2)), int(y_m - (y_diff/2))), (255, 0, 0), 2)
                #cv2.circle(img2, [obj[1]], True, (0, 255, 55), 2)   # 
                #cv2.circle(img2, [obj[2]], True, (255, 0, 0), 2)    # 
                cv2.putText(img2, (text + str(angle)), (700, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('A4', img2)
        '''
    
            
        
    #img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    #img_filter = utlis.
    # cv2.imshow('Original', img)
    # cv2.waitKey(1)

if __name__ == '__main__':
    count = 0
    while count == 0:
        kernel = np.ones((5, 5), np.uint8)
        dev_SN = []
        dist_to_object = 0
        color_cnt = 0
        box_cnt = 0
        detect_stack = bool
        x_m, y_m = 0, 0
        xmin, ymin = 0, 0
        clicked_points = []
        clone = None

        main()