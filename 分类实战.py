import os
import zipfile
import random
import json
import sys
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from PIL import Image, ImageEnhance

# 参数配置
train_parameters = {
    "input_size": [3, 224, 224],                                #输入图片的shape，模型需要的输入维度
    "class_dim": -1,                                          #分类数，后面会初始化
    "src_path":"data/data35095/train.zip",                   #原始数据集路径
    "target_path":"/home/aistudio/data/",                     #要解压的路径
    "train_list_path": "/home/aistudio/train.txt",            #train.txt路径train.txt
    "eval_list_path": "/home/aistudio/eval.txt",              #eval.txt路径
    "readme_path": "/home/aistudio/readme.json",              #readme.json路径
    "label_dict":{},                                          #标签字典
    "train_batch_size": 128                                    #训练时每个批次的大小
}

#解压数据集函数
def unzip_data(src_path,target_path):
    # 解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    if(not os.path.isdir(target_path + "train")):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


# 生成数据函数，生成训练集和测试集文件
def get_data_list(target_path, train_list_path, eval_list_path):
    '''
    生成数据列表
    '''
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + "train/"  # "target_path":"/home/aistudio/data/"
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":  # mac系统会自动产生这个文件，忽略掉
            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + '/' + img_path  # 每张图片的路径
                if class_sum % 8 == 0:  # 每8张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum测试数据的数目
                class_sum += 1  # 每类图片的数目
                all_class_images += 1  # 所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir  # 类别名称
            class_detail_list['class_label'] = class_label  # 类别标签
            class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
            class_detail.append(class_detail_list)
            # 初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1

            # 初始化分类数
    train_parameters['class_dim'] = class_dim

    # 乱序，写入文件
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

            # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')


# 执行上面创建的函数
'''
参数初始化
'''
src_path = train_parameters['src_path']
target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']
batch_size = train_parameters['train_batch_size']

'''
解压原始数据到指定路径
'''
unzip_data(src_path, target_path)

'''
划分训练集与验证集，乱序，生成数据列表
'''
# 每次生成数据列表前，首先清空train.txt和eval.txt，防止出现错误
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()  # 从0开始截断，即清空文件
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(target_path, train_list_path, eval_list_path)

class Cutout(paddle.vision.transforms.BaseTransform):
    def __init__(self, n_holes=1, length=112, prob=0.5, keys=None):
        super(Cutout, self).__init__(keys)
        self.prob = prob
        self.n_holes = n_holes
        self.length = length

    def _get_params(self, inputs):
        image = inputs[self.keys.index('image')]
        params = {}
        params['cutout'] = np.random.random() < self.prob
        # params['size'] = _get_image_size(image)
        return params

    def _apply_image(self, img):
        """ cutout_image """
        if self.params['cutout']:
            h, w = img.shape[:2] ## input image, with shape of (H, W, C)
            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                img[y1:y2, x1:x2] = 0
            return img
        else:
            return img


# 构造数据集读取类
class RabbishDataset(paddle.io.Dataset):

    def __init__(self, mode='train'):
        """
        初始化函数
        """
        self.data = []
        with open('{}.txt'.format(mode)) as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])
        self.transforms = T.Compose([
            T.Resize((224, 224)),  # 图片缩放
            T.ToTensor(),  # 数据的格式转换和标准化、 HWC => CHW
            Cutout(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transforms(image)
        return image, np.array(label, dtype='int64')

    def __len__(self):
        """
        获取样本总数
        """
        return len(self.data)

# 原始数据的训练集测试集
train_dataset = RabbishDataset(mode='train')
eval_dataset = RabbishDataset(mode='eval')
print(train_dataset.__len__())
print(eval_dataset.__len__())

# 数据展示
import matplotlib.pyplot as plt
print("图像的尺寸是",train_dataset[0][0][0].shape)
plt.imshow(train_dataset[0][0][0])

# 数据增强
# 数据集中没有标注文件
def generateLabelFile(all_file_dir, shuffle=True):
    # 生成图片路径列表和标签列表
    img_list = []
    label_list = []

    label_id = 0

    class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir, c))]

    for class_dir in class_list:
        img_path_pre = os.path.join(all_file_dir, class_dir)

        for img in os.listdir(img_path_pre):
            img_list.append(os.path.join(img_path_pre, img))
            label_list.append(label)
        label_id += 1

    # 生成标注的Dataframe
    img_df = pd.DateFrame(img_list)
    label_df = pd.DataFrame(label_list)

    img_df.columns = ['images']
    label_df.columns = ['labels']

    df = pd.concat([img_df, label_df], axis=1)

    if shuffle:
        df = df.reindex(np.random.permutation(df.index))

    df.to_csv("data.csv", index=0)


#################开始数据增强#####################
class DataAugmentation():
    def __init__(self, path):
        self.path = path
        df = pd.read_csv(path, names=['imgs', 'labels'], sep="	")

        self.imgs = df['imgs'].values
        self.labels = df['labels'].values

        self.img_list = []
        self.label_list = []

    def start(self, save_path):
        count = 0
        for i in range(len(self.imgs)):
            img = Image.open(self.imgs[i])
            img = img.convert("RGB")
            label = self.labels[i]

            # 上下颠倒
            out = img.transpose(Image.FLIP_TOP_BOTTOM)
            # out.convert("RGB")
            out.save(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.img_list.append(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.label_list.append(label)
            count += 1

            # 左右颠倒
            out = img.transpose(Image.FLIP_LEFT_RIGHT)
            out.convert("RGB")
            out.save(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.img_list.append(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.label_list.append(label)
            count += 1

            # 旋转90、180、270、360度
            for j in range(4):
                img = img.transpose(Image.ROTATE_90)
                img.save(os.path.join(save_path, "{:0>6}.jpg".format(count)))
                self.img_list.append(os.path.join(save_path, "{:0>6}.jpg".format(count)))
                self.label_list.append(label)
                count += 1

            # 调整亮度
            bright_enhancer = ImageEnhance.Brightness(img)

            bright_img = bright_enhancer.enhance(1.5)
            dark_img = bright_enhancer.enhance(0.5)

            bright_img.save(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.img_list.append(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.label_list.append(label)
            count += 1

            dark_img.save(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.img_list.append(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.label_list.append(label)
            count += 1

            # 对比度调整
            contrast_enhancer = ImageEnhance.Contrast(img)
            contrast_img = contrast_enhancer.enhance(1.2)
            contrast_img.save(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.img_list.append(os.path.join(save_path, "{:0>6}.jpg".format(count)))
            self.label_list.append(label)
            count += 1

    def writeToFile(self):
        img_df = pd.DataFrame(self.img_list)
        label_df = pd.DataFrame(self.label_list)

        img_df.columns = ['imgs']
        label_df.columns = ['labels']

        df = pd.concat([img_df, label_df], axis=1)

        df.to_csv("/home/aistudio/data.csv")

! mkdir new_rabbish_set
dg = DataAugmentation("/home/aistudio/train.txt")
dg.start("new_rabbish_set")
print("数据增强数据集已完成")
dg.writeToFile()

data = pd.read_csv('data.csv', encoding='utf-8')
with open('news_data.txt','a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[1])+'\t'+str(line[2])+'\n'))

# 导入增强后的数据
train_dataset = RabbishDataset(mode='news_data')
eval_dataset = RabbishDataset(mode='eval')
print(train_dataset.__len__())
print(eval_dataset.__len__())

MyCNN = paddle.nn.Sequential(
    nn.Conv2D(in_channels=3, out_channels=128, kernel_size=2, stride=1),
    #nn.MaxPool2D(kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Conv2D(in_channels=128, out_channels=196, kernel_size=1, stride=1),
    nn.Conv2D(in_channels=196, out_channels=256, kernel_size=7, stride=1),
    nn.Conv2D(in_channels=256, out_channels=318, kernel_size=1, stride=1),
    nn.Conv2D(in_channels=318, out_channels=618, kernel_size=3, stride=1),
    #nn.MaxPool2D(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=618*55*55,out_features=40)
)
MyAlexNet = paddle.nn.Sequential(
    nn.Conv2D(in_channels=3, out_channels=96, kernel_size=10, stride=1),
    nn.ReLU(),
    # nn.Conv2D(in_channels=3, out_channels=48, kernel_size=6, stride=1),
    # nn.ReLU(),
    # nn.Conv2D(in_channels=48, out_channels=96, kernel_size=5, stride=1),
    # nn.ReLU(),
    nn.MaxPool2D(kernel_size=3,stride=2),

    nn.Conv2D(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=3,stride=2),

    nn.Conv2D(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),

    nn.Conv2D(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),

    nn.Conv2D(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2D(kernel_size=3,stride=2),

    nn.Flatten(),
    nn.Linear(256*6*6, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(4096,4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096,40)
)
MyMixModel = paddle.nn.Sequential(
    nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3 ,stride=1, padding=1),
    nn.BatchNorm2D(64),
    nn.ReLU(),

    paddle.vision.models.vgg16().features[5:9],
    paddle.vision.models.resnet18().layer3,
    paddle.vision.models.resnet18().avgpool,
    nn.Flatten(),
    nn.Linear(in_features=256, out_features=40)
)

model1 = paddle.Model(MyCNN)
model2 = paddle.Model(MyAlexNet)
model3 = paddle.Model(MyMixModel)
model3.summary((10,3, 64, 64))
model4 = paddle.Model(paddle.vision.resnet50())

model4.prepare(paddle.optimizer.Adam(learning_rate=0.0001, parameters=model4.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 开始模型训练
model4.fit(train_dataset, epochs=1, batch_size=128, verbose=1)
model4.save("model_yes_1227")

# 模型评估，根据prepare接口配置的loss和metric进行返回
result = model4.evaluate(eval_dataset, verbose=1)
print(result)

#解压数据集函数
def unzip_data(src_path,target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至target_path目录下
    '''
    if(not os.path.isdir(target_path + "test")):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()

unzip_data("data/data38817/test.zip",target_path)


# 读取预测图像，进行预测
class predictDataset(paddle.io.Dataset):
    def __init__(self, img_path=None):
        """
        初始化函数，根据文件名分别读取训练集和测试集文件，获得图像文件的路径和标签
        """
        super().__init__()
        if img_path:
            self.img_paths = [img_path]
        else:
            raise Exception("请指定预测图片的路径")

    def __getitem__(self, index):
        """
        根据索引获取单个图像样本，并对图像数据进行预处理，返回图片数据和标签
        """
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1))  # 读出来的图像是rgb,rgb,rbg..., 转置为 rrr...,ggg...,bbb...
        img = img / 255.0
        return img

    def __len__(self):  # 获取数据的大小
        return len(self.img_paths)

## 遍历进行预测结果提取
result_all = []
data_list_path = target_path + "test/"
# "target_path":"/home/aistudio/data/"
for p in os.listdir(data_list_path):
    p = "/home/aistudio/data/test/" + p

    infer_data = predictDataset(p)  # 读取图片数据
    result = model4.predict(infer_data)  # 预测并返回结果列表
    result_all.append(np.argmax(result))

with open('model_result.txt','w') as fp:
    [fp.write(str(item)+'\n') for  item in result_all]
    fp.close()
