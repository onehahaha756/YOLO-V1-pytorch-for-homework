# YOLO-V1-pytorch for homework
## YOLO-V1 for homework
* 大家有找到好的代码可以以自己的名字建一个目录，往上面传一下。
* 或者大家对目录结构有啥建议的也可以说说

# xushanbo 的branch
* xmlToLabels.py是用来创建从xml类型的标注文件到 VOCdevkit\VOC2007\labels 的单图像.txt的标注文件。（这样程序里直接读的是.txt的标注文件）
* train.py现在是gpu的版本，如果需要用cpu运行，需要把 .cuda 都删除。
** 另外 train.py 里加载了预训练的resnet, 需要下载并放在 models_pkl 的目录下。
