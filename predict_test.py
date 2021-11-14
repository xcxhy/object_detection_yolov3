import os
import json
import time
import shutil
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box



def main(dir_name,img_name):
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "weights/yolov3spp-29.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    dir_name = dir_name
    img_path = os.path.join(dir_name,img_name)
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        img_o = cv2.imread(img_path)  # BGR
        assert img_o is not None, "Image Not Found " + img_path

        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]  # only get inference result
        t2 = torch_utils.time_synchronized()
        print(t2 - t1)

        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        print(t3 - t2)

        if pred is None:
            print("No target detected.")
            old_path = os.path.join(dir_name,img_name)
            new_path = os.path.join('test_result',img_name)
            shutil.copyfile(old_path,new_path)
            return 
            # exit(0)
            
        # process detections
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        print(pred.shape)

        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

        img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
        # plt.imshow(img_o)
        # plt.show()

        img_o.save(os.path.join("test_result",img_name))

def get_test_data(num):
    if num == '1':
        name = get_local()
    elif num == '0':
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
    if os.path.exists(class_name):
        print("目录已存在！")
    else:
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
        #cv2.imshow("capture", frame)

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

if __name__ == "__main__":
    print("=============================================")
    num = input('请输出0或1,0代表使用摄像头拍照，1代表手动传输图片:')
    dir_name = get_test_data(num)
    files = os.listdir(dir_name)
    # print(files,dir_name)
    # print(os.path.join(dir_name,files[0]))
    # print(os.path.exists(os.path.join(dir_name,files[0])))
    for i in range(len(files)):
        print(dir_name,files[i])
        main(dir_name,files[i])
