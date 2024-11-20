import cv2
import pyzbar.pyzbar as pz

def AddressReader(frame, used_codes):
    PointAddress = 0
    for code in pz.decode(frame):
        my_code = code.data.decode('utf-8')
        home = my_code.split(',', maxsplit=1)
        if len(home) >= 2:

            ho = home[1]
            adho = int(home[1])
                        

            if my_code not in used_codes:
                print('┌─────────────────────┐')
                #print('│        ', end="")
                print('         ',adho)
                print('└─────────────────────┘')
                used_codes.append(my_code)
                hcnt = True
                if adho == 103 :
                    PointAddress = 'a'
                elif adho == 601 :
                    PointAddress = 'b'
                elif adho == 1113 :
                    PointAddress = 'c'
                elif adho == 1703 :
                    PointAddress = 'd'
            else:
                '''
                ┌──────────────────────────────────────────────┐
                │                 aready exist                 │
                └──────────────────────────────────────────────┘
                '''
                #print("already exist")
        else:
            print("바코드 오류")

    return str(PointAddress)


used_codes = []


while True:
    barcode_reader = cv2.VideoCapture(0)
    barcode_reader.set(3, 640)
    barcode_reader.set(4, 480)
    acc = input('acc = '  )
    while acc:
        success, frame = barcode_reader.read()
        if not success:
            break  # 프레임을 읽을 수 없으면 루프를 벗어남
        address = AddressReader(frame, used_codes)
        if address != '0':
            ###send###
            acc = False  # 특정 조건을 충족하면 acc를 False로 변경하여 루프를 벗어남
        cv2.imshow('QRcode Barcode Scan', frame)
        key = cv2.waitKey(1)
    
    if not acc:
        barcode_reader.release()
        cv2.destroyAllWindows()
        # 다시 카메라를 켜도록 설정
    





