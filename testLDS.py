#!/usr/bin/python                                                                                                        
# -*- coding: utf-8 -*-                                                                                                  
## 1.초기 설정① #######################################                                                            
# �                  *                                                                                                  
from i611_MCS import *                                                                                                   
from i611_extend import *                                                                                                
from i611_io import * 
                                                                                                                         
from socket import *                                                                                                     
                                                                                                                         
def DHA16_Close():                                                                                                       
    dout(48,'001') 
    rb.sleep(0.5)
    dout(48,'000')                                                                                                      
                                                                                                                         
def DHA16_Open():                                                                                                        
    dout(48,'100') 
    rb.sleep(0.5) 
    dout(48,'000')      
                                                                                            
                                                                                                                       
def socket_server_run_hte_x():
    s = socket(AF_INET, SOCK_STREAM)                                                                                     
    s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    s.bind(("192.168.0.23", 9000))                                                                                       
    s.listen(1)
    conn, addr = s.accept()
    move_y = conn.recv(1024)
    print(move_y)
    #print 'handtoeye Received Data : ' + recv_data                                                                                 
                                                                                                                         
    #move_x, move_y = recv_data.split(',')                                                                        
                                                                                                                 
    s.close()
    print('socket1 close')                                                                                                         
                                                                                                                         
    return move_y                                                                                   

def socket_server_run_hte_y():
    s2 = socket(AF_INET, SOCK_STREAM)
    s2.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    s2.bind(("192.168.0.23",9000))
    s2.listen(1)
    conn2, addr2 = s2.accept()
    move_y = conn2.recv(1024)
    print('handineye Received Data : ' + move_y)
    #move_z, rotated_angle, rotated_direction = recv_data2.split(',')

    s2.close()
    print('socket2 close')

    return move_z

def socket_client_run():
    s3 = socket(AF_INET, SOCK_STREAM) 
    s3.connect(("192.168.0.42", 9999))
    s3.sendall('g'.encode())  # 데이터를 바이트로 인코딩하여 전송
    print('g send')
    s3.close()  # 소켓 닫기

def socket_client_run2():
    s3 = socket(AF_INET, SOCK_STREAM) 
    s3.connect(("192.168.0.42", 9999))
    s3.sendall('h'.encode())  # 데이터를 바이트로 인코딩하여 전송
    print('h send')
    s3.close()  # 소켓 닫기

def main():
    rb.open()
    IOinit( rb )
    DHA16_Open()

    #handtoeye
    print('hand to eye')
    socket_client_run2()
    block_dis_x = socket_server_run_hte_x()
    print('block_dis', block_dis_x)
    #move_x = 0 #float(move_x)*0
    #print('block_dis', block_dis)

    block_dis_x = float(block_dis_x)*1000
    block_dis_x = block_dis_x
    print('block_dis_x:', block_dis_x)
                                                    # block_dis - x-(첫째값 - 둘째값)
    rb.home()
    p2 = Position(0, -300, 400, 176.66 , -0.15 , 179.51)
    m = MotionParam( jnt_speed=10, lin_speed=20, overlap = 30 )                                                                        
    rb.motionparam(m)
    rb.move(p2)
    print('p3 move start')
    p3 = p2.offset(dx=-block_dis_x) 
    rb.move(p3)
    print('p3 move finish')

    #handineye 
    print('hand in eye') 

    while True:
        move_y =  socket_server_run_hte_y()
        #socket_client_run()
        if not move_y:
            print('Client disconnected. Exiting...')
            break
         
        move_y = float(move_y)*1000 - 55 #camera and greeper error length
        print(move_y) 
        p4 = p2.offset(dx= 0, dy= move_y)
        rb.move(p4)

        #box pick
        DHA16_Close()
        rb.sleep(1.0)
        
        
        rb.home()
        ##socket_server_run()
        print('home_end')
        print('home_end1')
        
        print('home_end2')
        rb.close    
        print('home_end3')
        socket_client_run()
        print('home_end4')                                    
        break        


                                                                                                                 
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






