1. 训练集和测试集都来自PASCAL VOC2007
2. 为了加快训练速度，现在的训练和测试都用的是动物这7类的图片。
3. train应该可以直接训练
4. test还有一点问题。我电脑的显卡没办法跑，不知道在gpu上会不会报错。
5. 如果之后要加数据，只要改数据文件，并把 .py 里的CLASSES改为对应的类别就可以。并且要用xmlToLabels.py把xml的标注文件变为VOCdevkit\VOC2007\labels 里的对应txt文件。同时，重新对数据进行划分，放在VOCdevkit\VOC2007\ImageSets\Main里的train.txt和val.txt。