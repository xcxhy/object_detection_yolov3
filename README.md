# object_detection_yolov3
基于yolov3的目标检测项目，你可以直接下载代码，或直接利用docker下载环境直接运行。详情见README
-------------------------------------------------------------------------------------------------
update:2021.11.16
这是第一次更新，优化了程序的一些问题，还有docker的自动化问题做了些优化。

我最新最新添加了一个flask1.py文件，基于flask设置了一个简单的前端。可以在网页端直接选择图片，并返回结果。
在运行flask1.py文件的时候，记得确保文件夹中不包含flask.py文件，否则会报错，显示flask库没安装。

修改：
更新了docker版本至xcxhy/ob_detector:1.1版本
重新更新了dockerfile文件，下载完毕之后，可以可以直接运行 

docker run -p  3333:5000 -it xcxhy/ob_detector:1.1

进入就会自动运行flask1.py文件。我们只需要在网页端输入IP:3333/detector就可以进行测试。
我把环境重新配在了base环境下，把CV环境删了。解决了之前不能直接进入CV环境的问题。
需要pull 新版本。

---------------------------------------------------------------------------------------------------------------

这里首先给出测试的代码
1.利用DOCKER下载镜像，直接测试
首先需要在Liunx或者Windows下安装docker
安装Docker请参考https://yeasy.gitbook.io/docker_practice/install
安装自己对应的系统的DOCKER
因为docker hub下载速度较慢，需要自己去配置对应系统的镜像加速器加速器
这里可以在阿里云官网搜索到方法。https://www.aliyun.com/
docker pull xcxhy/ob_detector:1.0
因为镜像里包括了pytorch，cuda等环境和部分训练数据，所以镜像比较大，后续版本会给出专门测试的版本。

下载完成后，可以利用docker images 查看镜像的信息。

docker run -it -v /home/test:/home/yolov3/test xcxhy/ob_detector:1.0 /bin/bash
这样就进入了测试的conda环境中。

激活我们的pytorch环境
conda activate CV

因为我们是利用了挂载技术所以test文件夹是与我们本地的文件夹数据共享的。
我们需要把测试图片放进test文件夹中就可以了。

利用python predict_test.py命令进行测试，测试的结果会报存在test_result文件夹中。
如果需要把结果报存到本地，输入docker cp 容器Id:/home/test_result /home/yolov3/test_result (容器ID需要CTRL+Q+P退出，利用docker ps查看ID号，再利用docker attach ID号进入)

2.配置环境，进行测试
需要配置的环境在requirement.txt文件中可见。
本目录下的代码，不包含训练集，以及测试参数，需要自己配置训练。
这里利用的是COCO数据集，所以必须安装pycocotools
不过clone的代码中没有数据，需要手动下载coco数据集。构建一个VOCdevkit文件夹，把下载的VOC2012文件夹放进去。

基础代码是利用https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp的代码构建做了些许修改
网络的具体讲解，也可以查看上述链接的讲解（推荐）

如果只想测试的话，我把我训练的数据放在百度云上链接：https://pan.baidu.com/s/18l_9ErlFUEIaSYQdyRQRWw    提取码：b7fs
新建一个weights目录，把下载的数据放进去，直接运行就行predict_test.py文件


