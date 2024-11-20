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


def send(sock):
    while True:
        sendData = raw_input('>>>')
        sock.send(sendData)


def receive(sock):
    while True:
        recvData = sock.recv(1024)
        receiveData = recvData.decode('utf-8')
        print('other people :', receiveData )
        Data = float(recvData.decode('utf-8'))
        return Data
        


port = 9999

def main():
    rb.open()
    IOinit( rb )
    DHA16_Open()

    print('hand to eye')

    p1 = Position(92.42, -155.40, 732.34, 0.03, 1.33, 137.01)
    #j1 = Joint(-0.245, -0.328, 24.647, -0.850, 112.681, 88.564)
    rb.move(p1)

    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.bind(('192.168.0.23', port))
    serverSock.listen(1)

    print('%d port connect waiting...'%port)

    connectionSock, addr = serverSock.accept()

    print(str(addr), 'connect_succcess!')

    sender = threading.Thread(target=send, args=(connectionSock,))
    receiver = threading.Thread(target=receive, args=(connectionSock,))

    sender.start()
    receiver.start()

    r_Data = receiver.start()
    print(r_Data)

    p2 = p1.offset( dx = r_Data)
    rb.sleep(10)

    p3 = p2.offset( dx = 0, dy = r_Data)
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