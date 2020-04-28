# -*- coding: utf-8 -*-


#为了加快计算，只用了7类，
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
#CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as tvmodel
from torch.utils.data import Dataset, DataLoader, TensorDataset

DATASET_PATH = 'VOCdevkit/VOC2007/'
NUM_BBOX = 2

def convert_bbox2labels(bbox):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0/7
    labels = np.zeros((7,7,5*NUM_BBOX+len(CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    return labels

class VOC2007(Dataset):
    def __init__(self,is_train=True,is_aug=True):
        """
        :param is_train: 调用的是训练集(True)，还是验证集(False)
        :param is_aug:  是否进行数据增广
        """
        self.filenames = []  # 储存数据集的文件名称
        if is_train:
            with open(DATASET_PATH + "ImageSets/Main/train.txt", 'r') as f: # 调用包含训练集图像名称的txt文件
                self.filenames = [x.strip() for x in f]
        else:
            with open(DATASET_PATH + "ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        self.imgpath = DATASET_PATH + "JPEGImages/"  # 原始图像所在的路径
        self.labelpath = DATASET_PATH + "labels/"  # 图像对应的label文件(.txt文件)的路径
        self.is_aug = is_aug

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = cv2.imread(self.imgpath+self.filenames[item]+".jpg")  # 读取原始图像
        h,w = img.shape[0:2]
        input_size = 448  # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
        if h>w:
            padw = (h - w) // 2
            img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)
        elif w>h:
            padh = (w - h) // 2
            img = np.pad(img,((padh,padh),(0,0),(0,0)), 'constant', constant_values=0)
        img = cv2.resize(img,(input_size,input_size))
        # 图像增广部分，这里不做过多处理，因为改变bbox信息还蛮麻烦的
        if self.is_aug:
            aug = transforms.Compose([transforms.ToTensor()])
            img = aug(img)

        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(self.labelpath+self.filenames[item]+".txt") as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox)%5!=0:
            raise ValueError("File:"+self.labelpath+self.filenames[item]+".txt"+"——bbox Extraction Error!")

        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        for i in range(len(bbox)//5):
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
            # 此处可以写代码验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验

        labels = convert_bbox2labels(bbox)  # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        # 此处可以写代码验证一下，经过convert_bbox2labels函数后得到的labels变量中储存的数据是否正确
        labels = transforms.ToTensor()(labels)
        return img,labels



def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0



class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1,self).__init__()

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """
        num_gridx, num_gridy = labels.size()[-2:]  # 划分网格数量
        num_b = 2  # 每个网格的bbox数量
        num_cls = np.size(CLASSES)  # 类别数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n = labels.size()[0]  # batchsize的大小

        # 可以考虑用矩阵运算进行优化，提高速度，为了准确起见，这里还是用循环
        for i in range(n):  # batchsize循环
            for m in range(7):  # x方向网格循环
                for n in range(7):  # y方向网格循环
                    if labels[i,4,m,n]==1:# 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((pred[i,0,m,n]+m)/num_gridx - pred[i,2,m,n]/2,(pred[i,1,m,n]+n)/num_gridy - pred[i,3,m,n]/2,
                                           (pred[i,0,m,n]+m)/num_gridx + pred[i,2,m,n]/2,(pred[i,1,m,n]+n)/num_gridy + pred[i,3,m,n]/2)
                        bbox2_pred_xyxy = ((pred[i,5,m,n]+m)/num_gridx - pred[i,7,m,n]/2,(pred[i,6,m,n]+n)/num_gridy - pred[i,8,m,n]/2,
                                           (pred[i,5,m,n]+m)/num_gridx + pred[i,7,m,n]/2,(pred[i,6,m,n]+n)/num_gridy + pred[i,8,m,n]/2)
                        bbox_gt_xyxy = ((labels[i,0,m,n]+m)/num_gridx - labels[i,2,m,n]/2,(labels[i,1,m,n]+n)/num_gridy - labels[i,3,m,n]/2,
                                        (labels[i,0,m,n]+m)/num_gridx + labels[i,2,m,n]/2,(labels[i,1,m,n]+n)/num_gridy + labels[i,3,m,n]/2)
                        iou1 = calculate_iou(bbox1_pred_xyxy,bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy,bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,0:2,m,n] - labels[i,0:2,m,n])**2) \
                                        + torch.sum((pred[i,2:4,m,n].sqrt()-labels[i,2:4,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,4,m,n] - iou1)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i,9,m,n]-iou2)**2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,5:7,m,n] - labels[i,5:7,m,n])**2) \
                                        + torch.sum((pred[i,7:9,m,n].sqrt()-labels[i,7:9,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,9,m,n] - iou2)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n]-iou1) ** 2)
                        class_loss = class_loss + torch.sum((pred[i,10:,m,n] - labels[i,10:,m,n])**2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i,[4,9],m,n]**2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss/n





class YOLOv1_resnet(nn.Module):
    def __init__(self):
        super(YOLOv1_resnet,self).__init__()
        
        # 调用torchvision里的resnet34预训练模型
        # 链接太慢，直接下载调用了
        # resnet = tvmodel.resnet34(pretrained=True)  
        
        #预先下载
        resnet = tvmodel.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./models_pkl/resnet34-333f7ec4.pth'),False)

        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel,1024,3,padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*(5*NUM_BBOX+len(CLASSES))),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)
        input = self.Conn_layers(input)
        return input.reshape(-1, (5*NUM_BBOX+len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据





if __name__ == '__main__':
	 #设置使用的gpu
    # os.environ['CUDA_VISIBLE_DEVICES']='0'

    epoch = 50
    batchsize = 5
    lr = 0.005

    train_data = VOC2007()
    train_dataloader = DataLoader(VOC2007(is_train=True),batch_size=batchsize,shuffle=True)
    
    #用cpu训练
    # model = YOLOv1_resnet()
    
    #用gpu训练
    model = YOLOv1_resnet().cuda()
    
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

    """
    is_vis = False  # 是否进行可视化，如果没有visdom可以将其设置为false
    if is_vis:
        vis = visdom.Visdom()
        viswin1 = vis.line(np.array([0.]),np.array([0.]),opts=dict(title="Loss/Step",xlabel="100*step",ylabel="Loss"))
    """
    
    for e in range(epoch):
        model.train()
        
        # 如果要可视化，下面这行要取消注释
        # yl = torch.Tensor([0]).cuda()
        
        for i,(inputs,labels) in enumerate(train_dataloader):

            # 用gpu
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            
            #用 cpu
            # inputs = inputs
            # labels = labels.float()

            pred = model(inputs)

            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e,epoch,i,len(train_data)//batchsize,loss))

            """
            # 如果要可视化，下面这段要取消注释
            yl = yl + loss
            if is_vis and (i+1)%100==0:
                vis.line(np.array([yl.cpu().item()/(i+1)]),np.array([i+e*len(train_data)//batchsize]),win=viswin1,update='append')
            """
            
        if (e+1)%10==0:
            torch.save(model,"./models_pkl/YOLOv1_epoch"+str(e+1)+".pkl")
            # compute_val_map(model)

