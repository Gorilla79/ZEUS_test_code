#!/usr/bin/python
# -*- coding: utf-8 -*-
## 1. 초기 설정① #######################################
# 라이브러리 가져오기
## 1．초기 설정 ①　모듈 가져오기 ######################
from i611_MCS import *
from i611_extend import *
from i611_io import *

from socket import *
import threading
import time


def DHA16_Close():                                                                                                       
    dout(48,'001') 
    rb.sleep(0.5)
    dout(48,'000')                                                                                                      
                                                                                                                         
def DHA16_Open():                                                                                                        
    dout(48,'100') 
    rb.sleep(0.5) 
    dout(48,'000')

'''
def send(sock):
    while True:
        sendData = raw_input('>>>')
        sock.send(sendData)
        sock.send(str(scaled_data).encode('utf-8'))


def receive(sock):
    while True:
        try:
            recvData = sock.recv(1024)
            if not recvData:
                raise Exception("Connection closed")
            
            # 데이터를 스케일링하여 로봇 제어에 맞는 값으로 변환
            receiveData = float(recvData.decode('utf-8'))  # 문자열을 실수로 변환
            scaled_data = scale_data(receiveData)  # 데이터 스케일링 함수 호출
            print('113')
            print('Received data:', scaled_data)
            print('112')
            

            return scaled_data
            
        except Exception as e:
            print("Error:", e)
            break
            '''


def socket_server():
    global port
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSock.bind(('192.168.0.23', port))
    serverSock.listen(1)

    print('%d port connect waiting...'%port)

    connectionSock, addr = serverSock.accept()

    print(str(addr), 'connect_succcess!')

    serverSock.close()
    print('socket close')

    return connectionSock

# 데이터 스케일링 함수
def scale_data(raw_data):
    # 받은 데이터를 적절한 스케일로 변환
    # 예: 받은 데이터가 -400이면 로봇에 맞는 스케일로 변환
    scaled_data = raw_data * 1  # 예: -400을 -4로 스케일링
    print('111')
    return scaled_data


def main():
    rb.open()
    IOinit( rb )
    DHA16_Open()

    print('hand to eye')

    global port
    port = 9999

    p1 = Position(92.42, -155.40, 732.34, 0.03, 1.33, 137.01)
    j1 = Joint(-0.245, -0.328, 24.647, -0.850, 112.681, 88.564)
    rb.move(j1)

    
    '''
    sender = threading.Thread(target=send, args=(connectionSock,))
    receiver = threading.Thread(target=receive, args=(connectionSock,))
    '''
    
    print('1')

    r_Data = socket_server()
    print('131')
    print(r_Data)
    r_Data = float(r_Data)*1000
    if r_Data is not None:
        print('Received data:', r_Data)
        print('3')
        # 스케일링된 데이터를 이용하여 로봇 제어
        p2 = p1.offset(dx=r_Data)
        rb.move(p2)
        rb.sleep(10)
        p3 = p2.offset(dx=0, dy=r_Data)
        rb.move(p3)
        rb.sleep(10)
        rb.home()   

    while True:
        time.sleep(1)
        pass



if __name__ == '__main__':     
    rb = i611Robot()
    _BASE = Base()                                                                                          
    print('start')   
    count_max = int(input('retry number :'))
    count = 0 
    while count < count_max:
        count += 1
        print('retry : ',count)
        main()     