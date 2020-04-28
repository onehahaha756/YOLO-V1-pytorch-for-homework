# -*- coding: utf-8 -*-


CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']

import xml.etree.ElementTree as ET
import os
import cv2

def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(DATASET_PATH,image_id):
    """把图像image_id的xml文件转换为目标检测的label文件(txt)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化"""
    in_file = open(DATASET_PATH + 'Annotations/%s' % (image_id))
    image_id = image_id.split('.')[0]
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)
        with open(DATASET_PATH + 'labels/%s.txt' % (image_id), 'w+') as out_file:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    in_file.close()
    
    

def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    DATASET_PATH = 'VOCdevkit/VOC2007/'
    filenames = os.listdir(DATASET_PATH + 'Annotations')
    for file in filenames:
        convert_annotation(DATASET_PATH,file)

make_label_txt()

def show_labels_img(dataset_path,imgname):
    """imgname是输入图像的名称，无下标"""
    img = cv2.imread(dataset_path + 'JPEGImages/' + imgname + '.jpg')
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open(dataset_path+"labels/"+imgname+".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    cv2.imshow("img",img)
    cv2.waitKey(0)

DATASET_PATH = "VOCdevkit/VOC2007/"
show_labels_img(DATASET_PATH,"000009")

'''
为了方便做数据划分train & val
import random

file1 = DATASET_PATH+"labels/name.txt"
file2 = DATASET_PATH+"ImageSets/Main/train.txt"
file3 = DATASET_PATH+"ImageSets/Main/val.txt"

f = open(file2,'a+')
F = open(file3,'a+')
filename = open(file1,'r')
for line in filename:
    if random.random()<0.5:
        f.write(line.split('.')[0]+'\n')
    else:
        F.write(line.split('.')[0]+'\n')
        
f.close()
F.close() 
filename.close()
    
'''