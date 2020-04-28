# YOLO-V1-pytorch for homework
## YOLO-V1 for homework
* 大家有找到好的代码可以以自己的名字建一个目录，往上面传一下。
* 或者大家对目录结构有啥建议的也可以说说

# xushanbo 的branch
* xmlToLabels.py是用来创建从xml类型的标注文件到 VOCdevkit\VOC2007\labels 的单图像.txt的标注文件。（这样程序里直接读的是.txt的标注文件）
* train.py现在是gpu的版本，如果需要用cpu运行，需要把 .cuda 都删除。
* 另外 train.py 里加载了预训练的resnet, 需要下载并放在 models_pkl 的目录下。<br/>
下载地址：<br/>
 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',<br/>
 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',<br/>
 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',<br/>
 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',<br/>
 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',<br/>

