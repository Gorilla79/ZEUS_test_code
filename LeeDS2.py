from ctypes import *
import random
import os
import cv2
import time
#import pynvml # 추가한 부분
import darknet
import argparse
import pyrealsense2.pyrealsense2 as rs
import numpy as np
from queue import Queue
import socket
import subprocess


def send_socket_y(soc_y):
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("192.168.0.23",9000))
    client_socket.sendall(str(soc_y).encode())

    client_socket.close()

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

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov3_custom.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov3_custom.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/yolov3_custom.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def nothing(x):
    pass

def convex(PATH):
    global angle, rotated_direction
    img = cv2.imread(PATH)
    img1 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray, 10, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)

    roi_image = img[int(y):int(y + h), int(x):int(x + w)]
    ####cv2.imshow("#2 roi_image", roi_image)

    # print('rect',rect)
    angle = rect[2:3]
    angle = angle[0] 
    # print('angle :',angle)
    # c mean center
    (hh, ww) = img.shape[:2]
    # print('ww',ww)
    # print('hh',hh)

    (cX, cY) = (ww / 2, hh / 2)

    if abs(angle) > 45:
        M = cv2.getRotationMatrix2D((cX, cY), 90 + angle, 1.0)
        #print('rotated_angle : ', 90 + angle)
    else:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        #print('rotated_angle : ', angle)

    rotated = cv2.warpAffine(img1, M, (x + ww, y + hh))
    
    cv2.namedWindow('#2 Rotated', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('#2 Rotated', 500, 200)
    cv2.moveWindow('#2 Rotated', 1280, 400) # Top-right corner 2
    cv2.imshow('#2 Rotated', rotated)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print('box',box)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
    # print('cnt',cnt)
    check = cv2.isContourConvex(cnt)
    # print('check',check)

    # 회전 방향
    if (box[0][0] > box[2][0]):
        rotated_direction = CCW
    else:
        rotated_direction = CW

    if not check:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img1, [hull], 0, (0, 255, 0), 3)
        cv2.namedWindow('#2 convexhull', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('#2 convexhull', 500, 200)
        cv2.moveWindow('#2 convexhull', 1280, 600) # Middle-right middle 2
        cv2.imshow('#2 convexhull', img1)
    
    cv2.namedWindow('#2 contour', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('#2 contour', 500, 200)
    cv2.moveWindow('#2 contour', 1280, 800) # Bottom-right corner 2
    cv2.imshow('#2 contour', img)

    return angle, rotated_direction

def Colorsearch(img_hsv2, lower_color, upper_color):
    global color_cnt, angle, rotated_direction, dist_to_object_z, write_stack,count

    img_mask2 = cv2.inRange(img_hsv2, lower_color, upper_color)

    erosion_mask2 = cv2.morphologyEx(img_mask2, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('#2 erosion_mask', erosion_mask2)

    cv2.namedWindow('#2 erosion_mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('#2 erosion_mask', 640, 400)
    cv2.moveWindow('#2 erosion_mask', 640, 600)  # Bottom-right corner
    cv2.imshow('#2 erosion_mask', erosion_mask2)

    img_result_color = cv2.bitwise_and(color_image2, color_image2, mask=erosion_mask2)

    if erosion_mask2[300, 400] == 0:
        color_cnt = 0

    else:
        color_cnt = color_cnt + 1

    #############################################################################################
    ################################             roi set            #############################

    ROI_frame_color = img_result_color
    numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(img_mask2)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])


        if area > 2500 and area < 307200:
            if color_cnt == 100:
                dist_to_object_z = aligned_depth_frame2.get_distance(400, 300) 
                dist_to_object_z = round(dist_to_object_z, 3)
                count = count + 1

                print('get image', dist_to_object_z)

                print('1')
                PATH = "image/saveimg.jpg"
                print('2')
                cv2.imwrite(PATH, ROI_frame_color)
                print('3')
                angle, rotated_direction = convex(PATH)
                print('4')
                send_socket_z(dist_to_object_z)
                print('5')
                #print('wait recive_socket')
                #recive_socket_data = recive_socket()
                #print('get recive_socket', recive_socket_data)
                '''
                if recive_socket_data == 'g':
                    if write_stack == True:
                        serial_write()
                        print('send_serial_Clear')
                        write_stack = False
                    else :
                        pass

                else : 
                    pass
                '''
            if color_cnt > 100:
                cv2.circle(color_image, (centerX, centerY), 10, (216, 168, 74), 10)
                cv2.rectangle(color_image, (x, y), (x + width, y + height), (216, 168, 74))

    return angle, rotated_direction, dist_to_object_z
    ################################             roi set            #############################
    #############################################################################################

# trackbar setting
cv2.namedWindow('Color - Depth Controller', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Color - Depth Controller', 640, 120)
cv2.createTrackbar('alpha_value', 'Color - Depth Controller', 0, 255, nothing)
cv2.setTrackbarPos('alpha_value', 'Color - Depth Controller', 117)
cv2.moveWindow('Color - Depth Controller', 0, 0)  # Top-left corner

def main():
    #gpu 사용 여부 추가
    #check_gpu_usage()
    global aligned_depth_frame2 , color_image, color_image2, recs_cup_cnt, regobox_cnt, regobox_socket,count
    regobox_socket = True  # regobox_socket 초기화 추가
    write_stack = True
    detect_stack = False
    recive_socket_data = []
    print('clear end parameter')

    time.sleep(1)

    get_device_sequence()
    dev_SN.sort()
    first_device, second_device = dev_SN
    print('first_device', first_device)
    print('second_device', second_device)
#############################################################################################
#######################             camera #1 setting                 #######################
# Create a pipeline
    print('set #1 camera')
    pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
    config = rs.config()
## camera serial number
    first_device = str(first_device)
    config.enable_device(first_device)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
    profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

#print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    print('clear #1 camera set')
#######################             camera #1 setting                 #######################
#############################################################################################
    time.sleep(0.5)
#############################################################################################
#######################             camera #2 setting                 #######################
# Create a pipeline
    print('set #2 camera')
    pipeline2 = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
    config2 = rs.config()
## camera serial number
    second_device = str(second_device)
    config2.enable_device(second_device)
# 831612071989
    config2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
    profile2 = pipeline2.start(config2)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor2 = profile2.get_device().first_depth_sensor()
    depth_scale2 = depth_sensor2.get_depth_scale()

#print("#2 Depth Scale is: ", depth_scale2)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
    align_to2 = rs.stream.color
    align2 = rs.align(align_to2)
    print('clear #2 camera set')

#######################             camera #2 setting                 #######################
#############################################################################################
#print("multi_realsemse_camera_setting clear")
    parser()
    args = parser()
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )

#    width_yolo = darknet.network_width(network)
#    height_yolo = darknet.network_height(network)
    width_yolo = 640
    height_yolo = 480
    darknet_image = darknet.make_image(width_yolo, height_yolo, 3)

    try:
        while True:
        ################################################################################################################
        ##########################                    #1 camera hsv color search                  ######################
            alpa = cv2.getTrackbarPos('alpha_value', 'Color - Depth Controller')

        # Get frameset of color and depth
            frames = pipeline.wait_for_frames()


        # Align the depth frame to color frame
       #     print("frame out1")
            aligned_frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            width_depth = depth_frame.get_width()
            height_depth = depth_frame.get_height()

        # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        # Render images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=alpa / 255), cv2.COLORMAP_JET)
        # Yolo Size
            frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        #img_hsv = cv2.cvtColor(depth_colormap_resized, cv2.COLOR_BGR2HSV)

            frame_resized = cv2.resize(frame_rgb, (width_yolo, height_yolo), interpolation=cv2.INTER_LINEAR)
            depth_colormap_resized = cv2.resize(depth_colormap, (width_yolo, height_yolo), interpolation=cv2.INTER_LINEAR)
        #img_hsv_resized = cv2.resize(img_hsv, (width_yolo, height_yolo), interpolation=cv2.INTER_LINEAR)

            #cv2.imshow("aligned_depth_frame", aligned_depth_frame)
            #cv2.imshow('#1 depth_colormap', depth_colormap)
            #cv2.imshow("depth_image", depth_image)
        ##########################                    #1 camera hsv color search                  ######################
        ################################################################################################################

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.7)

            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            '''
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 640, 400)
            cv2.moveWindow('frame', 0, 170)
            cv2.imshow('frame', image)
            '''

            if detections:
                detect_stack = True
                name = []
                name = detections[0]

                class_name = name[0]
                if class_name == 'recs_cup':
                    recs_cup_cnt = recs_cup_cnt + 1
                    bbox = name[2]
                    xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)

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
                        ####cv2.imshow("deep_ROI_depth_color", deep_ROI_depth_color)
                        fps = int(1/(time.time() - prev_time))
                        #ROI_depth = deep_ROI_depth_color.get_depth_frame()jp

                    try:
                        if recs_cup_cnt == 10 :
                            dist_to_object = aligned_depth_frame.get_distance(int(((xmax - xmin) / 2) + xmin), int(((ymax - ymin) / 2) + ymin))
                            dist_to_object = round(dist_to_object, 3)
                            move_to_x = int(width_yolo/2) - (int((xmax - xmin)/2) + xmin)

                            print('###################################################')
                            print('')
                            print('class_name : ', class_name, 'accuracy : ', name[1])
                            print('object_boxes_xmin : ', xmin, ', object_boxes_xmax : ', xmax)
                            print('object_boxes_ymin : ', ymin, ', object_boxes_ymax : ', ymax)
                            print('Get ROI y_vector and x_vector !!')
                            print("FPS: {}".format(fps), ', x_vector', move_to_x, ', y_vector', dist_to_object,'m')

                            if abs(angle) > 45:
                                print('rotated_angle : ', 90 + angle, ', rotated_direction : ', rotated_direction)
                            else :
                                print('rotated_angle : ', angle, ', rotated_direction : ', rotated_direction)

                            print('')
                            print('###################################################')

                            recs_cup_cnt = 0
                        pass

                    except RuntimeError as err:
                        print(err)
                        pass

                elif class_name == 'regobox':
                    regobox_cnt = regobox_cnt + 1
                    bbox = name[2]
                    xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)

                    deep_ROI_frame_color = image[ymin: ymax, xmin: xmax]
                    deep_ROI_depth_color = depth_colormap_resized[ymin: ymax, xmin: xmax]

                    if deep_ROI_frame_color is None :
                        pass

                    else :
                        fps = int(1/(time.time() - prev_time))

                    try:
                        if regobox_cnt == 10 : # origin 10, 나중에 수정해보기

                            dist_to_object = aligned_depth_frame.get_distance(int(((xmax - xmin) / 2) + xmin), int(((ymax - ymin) / 2) + ymin))
                            dist_to_object = round(dist_to_object, 3)
                            move_to_x = int(width_yolo/2) - (int((xmax - xmin)/2) + xmin)

                            print('###################################################')
                            print('')
                            print('class_name : ', class_name, 'accuracy : ', name[1])
                            print('object_boxes_xmin : ', xmin, ', object_boxes_xmax : ', xmax)
                            print('object_boxes_ymin : ', ymin, ', object_boxes_ymax : ', ymax)
                            print('Get ROI y_vector and x_vector !!')
                            print("FPS: {}".format(fps), ', x_vector', move_to_x, ', y_vector', dist_to_object,'m')

                            if abs(angle) > 45:
                                print('rotated_angle : ', 90 + angle, ', rotated_direction : ', rotated_direction)
                            else :
                                print('rotated_angle : ', angle, ', rotated_direction : ', rotated_direction)

                            print('')
                            print('###################################################')

                            if regobox_socket == True:
                                send_socket_y(dist_to_object)              ## mani socket test
                                print('socket send finish', dist_to_object)
                                #time.sleep(2)
                                #print('time.sleep(2) clear')
                                regobox_socket = False
                                regobox_cnt = 0

                            else :
                                regobox_cnt = 0
                        pass

                    except RuntimeError as err:
                        print(err)
                        pass
                else :
                    #print('No Detect')
                    pass

            else :
                detect_stack = False
                pass


            ################################################################################################################
            ##########################                    #2 camera hsv color search                  ######################

            frames2 = pipeline2.wait_for_frames()
            aligned_frames2 = align2.process(frames2)
            aligned_depth_frame2 = aligned_frames2.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame2 = aligned_frames2.get_color_frame()

            if not aligned_depth_frame2 or not color_frame2:
                continue

            depth_image2 = np.asanyarray(aligned_depth_frame2.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())

            '''
            cv2.namedWindow('#2 color_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('#2 color_image', 640, 400)
            cv2.moveWindow('#2 color_image', 640, 170)  # Top-right corner
            cv2.imshow('#2 color_image', color_image2)
            '''

            depth_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image2, alpha=alpa / 255), cv2.COLORMAP_JET)
            img_hsv2 = cv2.cvtColor(depth_colormap2, cv2.COLOR_BGR2HSV)

            #cv2.imshow('#2 color_image', color_image2)dist_to_object_zsocek
            #cv2.imshow('#2 depth_colormap', depth_colormap2)

            # Move windows to different screen corners
            '''
            cv2.namedWindow('#2 depth_colormap', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('#2 depth_colormap', 640, 400)
            cv2.moveWindow('#2 depth_colormap', 0, 600)  # Bottom-left corner
            cv2.imshow('#2 depth_colormap', depth_colormap2)
            '''

            # hus value detect
            ROI_depth_colormap = img_hsv2[300, 400]
            one_pixel = np.uint8([ROI_depth_colormap])
            hsv = one_pixel[0][0]
            #print('hsv  : ', hsv)
            #print('')
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

            angle, rotated_direction, dist_to_object_z = Colorsearch(img_hsv2, lower_color, upper_color)

            ##########################                    #2 camera hsv color search                     ###################
            ################################################################################################################

            # Display the resulting frame
            #cv2.imshow('frame', image)


            
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 640, 400)
            cv2.moveWindow('frame', 0, 170)
            cv2.imshow('frame', image)
            
            cv2.namedWindow('#2 color_image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('#2 color_image', 640, 400)
            cv2.moveWindow('#2 color_image', 640, 170)  # Top-right corner
            cv2.imshow('#2 color_image', color_image2)
            
            cv2.namedWindow('#2 depth_colormap', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('#2 depth_colormap', 640, 400)
            cv2.moveWindow('#2 depth_colormap', 0, 600)  # Bottom-left corner
            cv2.imshow('#2 depth_colormap', depth_colormap2)
            
            print(count)
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
        pipeline2.stop()

if __name__ == '__main__':
    count = 0
    while count == 0:
          #############################################################################################
           ################################            value set           #############################
        kernel = np.ones((5, 5), np.uint8)
        color_cnt = 0
        max_x = 0
        max_y = 0
        max_w = 0
        max_h = 0
        cnt_fisrt_roi_catch = 0
        rotated_direction = 0
        CCW = 5
        CW = 4
        angle = 0
        count = 0
        

        number_of_devices = 2          #realsense_device_number_setting
        dev_SN = []
        detect_stack = bool
        recs_cup_cnt = 0
        regobox_cnt = 0
        starting_pap_0 = []
        recive_socket_data = []
        regobox_socket = True
        write_stack = True
        dist_to_object_z = 0

        main()
        
        