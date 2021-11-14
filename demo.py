import cv2
import os
import predict_test

def get_test_data(num):
    if num == '1':
        name = get_local()
    elif num== '0':
        class_name = input("请输入拍摄存储目录：")
        name = get_camera(class_name)
    return name

def get_camera(class_name):
    print("=============================================")
    print("=  热键(请在摄像头的窗口使用)：             =")
    print("=  z: 更改存储目录                          =")
    print("=  x: 拍摄图片                              =")
    print("=  q: 退出                                  =")
    print("=============================================")
    while os.path.exists(class_name):
        class_name = print("目录已存在！")
    os.mkdir(class_name)

    index = 1
    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    w = 360
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    crop_w_start = (width-w)//2
    crop_h_start = (height-w)//2

    print(width, height)

    while True:
        # get a frame
        ret, frame = cap.read()
        # show a frame
        frame = frame[crop_h_start:crop_h_start+w, crop_w_start:crop_w_start+w]
        frame = cv2.flip(frame,1,dst=None)
        cv2.imshow("capture", frame)

        input = cv2.waitKey(1) & 0xFF

        if input == ord('z'):
            class_name = input("请输入拍摄存储目录：")
            while os.path.exists(class_name):
                class_name = print("目录已存在！")
            os.mkdir(class_name)
        elif input == ord('x'):
            cv2.imwrite("%s/%d.jpeg" % (class_name, index),
            cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
            print("%s: %d 张图片" % (class_name, index))
            index += 1
        if input == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()
    return class_name

def get_local():
    while True:
        class_name = input('请输入存储目录：')
        if os.path.exists(class_name):
            print("目录已存在！")
            break
        else:
            print("目录不存在，请重新输入")
    return class_name

if __name__=='__main__':
    print("=============================================")
    num = input('请输出0或1,0代表使用摄像头拍照，1代表手动传输图片:')
    print(get_test_data(num))