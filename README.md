# Deep-learning-project-Combat-1
这是一个简单的深度学习实战项目，你甚至可以零基础去使用它  
最近也在学习深度学习有关方面的课程，然后就做了这样一个小项目，感觉比很多的项目要简单易懂，可以分享一下  
之前一直不能入门也感觉很烦恼，希望可以给同样经历的人一些帮助。

代码环境：pytorch
文件格式：首先我们新建一个文件夹可以命名为 work1作为项目文件夹  
接着新建四个文件夹，分别命名为dataset imgs logs_train model_weight  
将核心代码放在与这几个文件夹同一目录下。  
dataset：用于存放数据集，这里是采用cifar-10作为样例，因为很小方便使用，不需要去下载，调用train.py时会自动下载数据集，只需要建立好文件夹名称对应就好。
model_weight：当我们训练好一个网络，最重要的当然就是保留好参数啦，将他们放在这个文件夹下，对应的分别是每轮的结果，以.pth格式保存
imgs:包含测试图片，用于测试网络效果。
logs_train：是一个tensorboard保存的events文件，方便在tensorboard上进行每次结果查看，这块不懂可以看度娘。。可以建个文件夹，不使用它  
train_gpu_1.py和train_gpu_2.py是一样的，只是写法不同。。  

使用方法：先运行train.py，或者train_gpu_1，train_gpu_2，接着根据使用test.py查看结果。


