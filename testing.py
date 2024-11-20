#!/usr/bin/python
# -*- coding: utf-8 -*-
## 1. 초기 설정① #######################################
# 라이브러리 가져오기
## 1．초기 설정 ①　모듈 가져오기 ######################
from i611_MCS import *
from i611_extend import *
from i611_io import *

from socket import *
import time


def DHA16_Close():                                                                                                       
    dout(48,'001') 
    rb.sleep(0.5)
    dout(48,'000')                                                                                                      
                                                                                                                         
def DHA16_Open():                                                                                                        
    dout(48,'100') 
    rb.sleep(0.5) 
    dout(48,'000')

# 데이터 스케일링 함수 (cm에서 mm로 변환)
def scale_data(raw_data_cm):
    # 받은 데이터를 적절한 스케일로 변환 (cm에서 mm로 변환)
    scaled_data_mm = raw_data_cm * 10  # cm를 mm로 변환
    return scaled_data_mm


def socket_server():
    global port
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSock.bind(('192.168.0.23', port))
    serverSock.listen(1)

    print('%d port connect waiting...'%port)

    connectionSock, addr = serverSock.accept()

    print(str(addr), 'connect_success!')

    print('socket close')
    serverSock.close

    return connectionSock

def main():
    rb.open()
    IOinit(rb)
    DHA16_Open()

    print('hand to eye')

    global port
    port = 9999

    p1 = Position(92.42, -155.40, 732.34, 0.03, 1.33, 137.01)
    j1 = Joint(-0.245, -0.328, 24.647, -0.850, 112.681, 88.564)
    rb.move(j1)

    while True:
        try:
            r_Data = socket_server()
            r_Data = scale_data(float(r_Data))  # 스케일링 (cm에서 mm로 변환)
            if r_Data is not None:
                print('Received data (in mm):', r_Data)
                p2 = p1.offset(dx=r_Data)
                rb.move(p2)
                rb.sleep(10)
                p3 = p2.offset(dx=0, dy=r_Data)
                rb.move(p3)
                rb.sleep(10)
                rb.home()
        except Exception as e:
            print("Error:", e)

if __name__ == '__main__':
    rb = i611Robot()
    _BASE = Base()
    print('start')
    count_max = int(input('retry number: '))
    count = 0
    while count < count_max:
        count += 1
        print('retry:', count)
        main()
