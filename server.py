#!/usr/bin/python
# -*- coding: utf-8 -*-
import socket 
import time 

print("1")

server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#server_socket.bind(("192.168.1.21",9999))
server_socket1.bind(("192.168.0.42",9999))
server_socket1.listen(1) #소켓 서버의 클라이언트의 접속을 기다린다.
print("2")
client_socket, addr = server_socket1.accept() #요청 수신되면 요청을 받아들여 데이터 통신을 위한 소켓 생성

x_cm = 12.914
y_cm = -30.325

while True: #데이터 송수신
    try:

        data = client_socket.recv(65535) #data 인스턴스 생성 및 수신
        data = data.decode()  #수신된 byte code를 문자열로 변환
        print (data) #변환된 문자열을 출력 

        server_socket1.close()

        #setdata = '{x_cm:.3f},{y_cm:.3f}'.format(x_cm, y_cm)

        client1_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # socket() 소켓서버 생성
        client1_socket1.bind(('192.168.0.23',9000)) #서버가 사용할 IP주소와 포트번호를 생성한 소켓에 결합
        client1_socket1.sendall(str(x_cm).encode())
        print('x_cm')
        client1_socket1.close()

        time.sleep(10)

        client2_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # socket() 소켓서버 생성
        client2_socket.bind(('192.168.0.23',9000)) #서버가 사용할 IP주소와 포트번호를 생성한 소켓에 결합
        client2_socket.sendall(str(y_cm).encode())
        print('y_cm')
        client2_socket.close()


        server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #server_socket.bind(("192.168.1.21",9999))
        server_socket2.bind(("192.168.0.42",9999))
        server_socket2.listen(1) #소켓 서버의 클라이언트의 접속을 기다린다.
        client_socket2, addr = server_socket2.accept() #요청 수신되면 요청을 받아들여 데이터 통신을 위한 소켓 생성

        data2 = client_socket2.recv(65535) #data 인스턴스 생성 및 수신
        data2 = data2.decode()  #수신된 byte code를 문자열로 변환
        print (data2) #변환된 문자열을 출력


    except :
        server_socket2.close()
    finally:
        server_socket2.close()