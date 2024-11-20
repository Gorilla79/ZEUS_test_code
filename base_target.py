#!/usr/bin/python
# -*- coding: utf-8 -*-

## 1. 초기 설정① #######################################
# 라이브러리 가져오기
## 1．초기 설정 ①　모듈 가져오기 ######################
from i611_MCS import *
from teachdata import *
from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import * 


def main():
    
    ## 2. 초기 설정② ####################################
    # i611 로봇 생성자
    rb = i611Robot()
    # 좌표계의 정의
    _BASE = Base()
    # 로봇과 연결 시작 초기화
    rb.open()
    # I/O 입출력 기능의 초기화 
    IOinit( rb )
    # 교시 데이터 파일 읽기
    data = Teachdata( "teach_data" )
    ## 1. 교시 포인트 설정 ######################
    
    j1 = Joint( -13.6, 33.5, 99.7, -0.3, 45.2, 29.0 )
    p1 = rb.Joint2Position(j1)
    #p7 = Position(486, -500, 219.3, 47.6, 1.51, 179.42)
    #j6 = rb.Position2Joint(p6)

    j2 = Joint( 17.6, 42.2, 88.2, -0.3, 47.8, 29.0 )
    p2 = rb.Joint2Position(j2)

    j3 = Joint( 38.4, 43.6, 80.0, -1.4, 55.7, 62.5 )
    p3 = rb.Joint2Position(j3)

    j4 = Joint( -33.1, 43.6, 80.0, -1.4, 55.7, 13.6 )
    p4 = rb.Joint2Position(j4)

    j5 = Joint( -39.3, 72.4, 12.3, -1.4, 91.4, -35.4 )
    p5 = rb.Joint2Position(j5)

    p6 = p5.offset(dx = 300)
    p7 = p6.offset(dx = 300)
    p8 = p7.offset(dx = 200)
    ## 2. 동작 조건 설정 ######################## 
    m = MotionParam( jnt_speed=20, lin_speed=200, pose_speed=100, overlap = 30 )
    #MotionParam 형으로 동작 조건 설정
    rb.motionparam( m )
    
    ## 3. 로봇 동작을 정의 ##############################
    # 작업 시작
    # Home 위치 이동
    rb.home()
    
    # Move 위치로 p1 dz :100 offset만큼 이동
    rb.move(p1)
    #rb.sleep(3)

    pos = rb.getpos()       
    print('pos', pos)
    pos_value = pos.position()                     
    print('pos_value : ', pos_value)
    pos_str = [str(round(x,2)) for x in pos_value[0:6]]
    print('pos_str : ', pos_str)
    
    rb.home()
    #rb.sleep(5)

    rb.move(p2)
    #rb.sleep(3)

    pos = rb.getpos()       
    print('pos', pos)
    pos_value = pos.position()                     
    print('pos_value : ', pos_value)
    pos_str = [str(round(x,2)) for x in pos_value[0:6]]
    print('pos_str : ', pos_str)

    rb.home()
    #rb.sleep(5)
        
    rb.move(p3)
    #rb.sleep(3)

    pos = rb.getpos()       
    print('pos', pos)
    pos_value = pos.position()                     
    print('pos_value : ', pos_value)
    pos_str = [str(round(x,2)) for x in pos_value[0:6]]
    print('pos_str : ', pos_str)

    rb.home()
    #rb.sleep(5)

    rb.move(p4)
    #rb.sleep(3)

    pos = rb.getpos()       
    print('pos', pos)
    pos_value = pos.position()                     
    print('pos_value : ', pos_value)
    pos_str = [str(round(x,2)) for x in pos_value[0:6]]
    print('pos_str : ', pos_str)

    rb.home()
    rb.sleep(5)

    rb.move(p5)
    rb.sleep(3)

    pos = rb.getpos()       
    print('pos', pos)
    pos_value = pos.position()                     
    print('pos_value : ', pos_value)
    pos_str = [str(round(x,2)) for x in pos_value[0:6]]
    print('pos_str : ', pos_str)

    rb.line(p6)
    rb.line(p7)
    rb.line(p8)
    rb.sleep(2)
    rb.home()


if __name__ == '__main__':
    main()