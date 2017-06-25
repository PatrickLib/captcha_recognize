介绍
====
### 其他语言： [English](https://github.com/PatrickLib/captcha_recognize/blob/master/README.md) [中文](https://github.com/PatrickLib/captcha_recognize/blob/master/README-zhcn.md)

基于TensorFlow的验证码识别，不需要对图片进行切割，运行环境Ubuntu 16.04，Python 2.7

![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/CMQVA_num717_1.png)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/CMQZJ_num908_1.png)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/CRGEU_num339_1.png)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/CZHBN_num989_1.png)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/DZPEW_num388_1.png)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/CZWED_num21_1.png)

使用captcha_eval.py评估的准确率为99.7%，训练集大小为50000，20000轮训练，验证码的生成代码见项目：https://github.com/lepture/captcha

![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1ab2s_num286.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1ezx8_num398.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1iv22_num346.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1kxw2_num940.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/3mtj9_num765.jpg)![captcha](https://raw.githubusercontent.com/PatrickLib/captcha_recognition/master/data/test_data/1vuy5_num17.jpg)

使用captcha_eval.py评估的准确率为52.1%，训练集大小为100000，200000轮训练，验证码的生成代码见项目：https://github.com/Gregwar/CaptchaBundle
 
依赖环境
=======
### python 2.7
### Anaconda2 4.3.1
https://www.continuum.io/downloads#linux
### TensorFlow 1.1
https://github.com/tensorflow/tensorflow
### captcha
https://pypi.python.org/pypi/captcha/0.1.1

使用步骤
=======
## 1.准备验证码图片
将验证码图片分别放在 **<工作目录>/data/train_data/** 用于模型训练，**<工作目录>/data/valid_data/** 用于模型效果评估， **<工作目录>/data/test_data/** 用于验证码识别测试，图片命名样式是 **验证码内容_\*.jpg** 或者 **验证码内容_\*.png** ，图片大小最好为 **128x48** . 可以执行默认的验证码生成:
```
python captcha_gen_default.py
```

## 2.将验证码图片转换为tfrecords格式
生成的结果为 **<工作目录>/data/train.tfrecord** 和 **<工作目录>/data/valid.tfrecord** ，执行：
```
python captcha_records.py
```

## 3.模型训练
可以在CPU或者一个GPU上进行模型训练，执行：
```
python captcha_train.py
```
也可以在多个GPU上进行模型训练，执行：
```
python captcha_multi_gpu_train.py
```

## 4.模型评估
用于评估训练的效果，执行：
```
python captcha_eval.py
```

## 5.验证码识别
训练好模型后，可以对 **<工作目录>/data/test_data/** 目录下的原始图片进行识别，执行：
```
python captcha_recognize.py
```
结果如下
```
...
image WFPMX_num552.png recognize ----> 'WFPMX'
image QUDKM_num468.png recognize ----> 'QUDKM'
```

