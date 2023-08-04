from __future__ import print_function
import argparse #这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import os
import numpy#一个用于科学计算的基础包
import torch#一个科学计算框架，广泛支持机器学习算法
import torch.nn as nn#专门为神经网络设计的模块化接口
import torch.nn.functional as F#一个很常用的模块，nn中的大多数layer在functional中都有一个与之对应的函数
import torch.optim as optim#torch.optim：一个实现了各种优化算法的库
from torchvision.datasets import ImageFolder#ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名
from PIL import Image#PIL(Python Image Library)是python的第三方图像处理库
import numpy as np
from torchvision import datasets, transforms#datasets : 常用数据集的dataset实现  #transforms :常用的图像预处理方法，例如裁剪、旋转等
from torch.autograd import Variable#构建神经网络的计算图时，需用orch.autograd.Variable将Tensor包装起来，形成计算图中的节点。
                                   #Variable是torch.autograd中很重要的类。它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example') #创建解析器--使用 argparse 的第一步是创建一个 ArgumentParser 对象。
                                                                      #ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
#19~44：添加参数--通过调用 add_argument() 方法完成
parser.add_argument('--dataroot', default="/input/" ,help='path to dataset')#--dataroot：名字；default：不指定参数时的默认值；help：参数的帮助信息
parser.add_argument('--evalf', default="/eval/" ,help='path to evaluate sample')
parser.add_argument('--outf', default='models',
                    help='folder to output images and model checkpoints')
parser.add_argument('--ckpf', default='',
                    help="path to model checkpoint file (to continue training)")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train', action='store_false',
                    help='training a ConvNet model on MNIST dataset')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate a [pre]trained model')


#属性给与args实例： 把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中，使用即可。
args = parser.parse_args()
# use CUDA?
use_cuda = not args.no_cuda and torch.cuda.is_available()#如果arg.no_cuda为False并且当前设备支持CUDA，则将args.cuda设置为True，否则将其设置为False
device = torch.device("cuda" if use_cuda else "cpu")#将device命名为device，有cuda用cuda，没cuda就用cpu

# Is there the outf?是否存在局部变量
try:                            # os.makedirs() 方法用于递归创建目录。                           
    os.makedirs(args.outf)      # 如果子目录创建失败或者已经存在，会抛出一个 OSError 的异常 
except OSError:
    pass#如果出现OSError直接中断

torch.manual_seed(args.seed)# 为CPU设置种子用于生成随机数，以使得结果是确定的
if use_cuda:
    torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子

# From MNIST to Tensor从数据集到张量                                #kwargs：keyword arguments表示关键字参数
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}#num_workers：使用多进程加载的进程数
                                                                   #pin_memory：是否将数据保存在pin memory区
# Load MNIST only if training只在训练时载入mnist
if args.train:#加载训练集和测试集
    train_loader = torch.utils.data.DataLoader(#读取数据，主要是对数据进行batch的划分
        datasets.MNIST(root=args.dataroot, train=True, download=True,#导入数据集，读入的数据作为训练集，根目录下没有时自动下载
                       transform=transforms.Compose([#Compose()类：主要作用是串联多个图片变换的操作。
                           transforms.ToTensor(),#转换一个PIL库的图片或者numpy的数组为tensor张量类型
                           transforms.Normalize((0.1307,), (0.3081,))#通过平均值（0.1307）和标准差（0.3081）来标准化一个tensor图像
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)#batch_size::即一次训练所抓取的数据样本数量;shuffle=True用于打乱数据集，每次都会以不同的顺序返回
                                                           #**kwargs允许你将不定长度的 【键值对 key-value 】，作为参数传递给一个函数    
    #测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=args.dataroot, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

#构建模型并送到GPU加速
class Net(nn.Module):#子类继承nn.Module
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    #子模块创建
    def __init__(self):
        super(Net, self).__init__()# 使用父类的初始化方法来初始化子类
        self.conv1 = nn.Conv2d(1, 20, 5, 1)#二维卷积输入和输出通道数分别为1和20，卷积核大小为5，步长为1#conv卷积层
        self.conv2 = nn.Conv2d(20, 50, 5, 1)#输入和输出通道数分别为20和50
        self.fc1 = nn.Linear(4*4*50, 500)#输入张量的形状为4*4*50，输出张量的形状为500#fc神经网络层
        self.fc2 = nn.Linear(500, 10)#输入张量形状为500*500*500，输出张量的形状为10
    #子模块拼接。将数据送到模型、模型如何处理
    def forward(self, x):#forward函数用于前向传播计算，输入数据x会被送入模型中进行计算，并最终得到输出结果
        x = F.relu(self.conv1(x))# 用一下卷积层conv1，然后做一个激活。
        x = F.max_pool2d(x, 2, 2)#用x作为一个池化层，池化的窗口大小为2*2，激活窗口为2*2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)# 最大池化 + 激活函数 = 下采样
        x = x.view(-1, 4*4*50)#view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变。4*4*50大小的张量
        x = F.relu(self.fc1(x))#用一下神经网络层fc1，然后激活
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)#在softmax的结果上再做多一次log运算# dim：归一化方式，0是对列做归一化，1是对行做归一化


model = Net().to(device)#将模型加载到相应的设备中

# Load checkpoint模型保存
if args.ckpf != '':
    if use_cuda:
        model.load_state_dict(torch.load(args.ckpf))#直接加载模型
    else:
        # Load GPU model on CPU  #将cpu训练好的模型参数load到gpu上
        model.load_state_dict(torch.load(args.ckpf, map_location=lambda storage, loc: storage))

#1创建一个optimizer对象：能保存当前的参数状态并且基于计算梯度更新参数。
#2建构一个优化器，优化方法时SGD，给它一个包含参数（第一个）进行优化，然后可以指定optimizer的参数选项：learning rate（学习率）、momentum（动量）
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#训练过程，接收配置 args, 模型model, GPU device, 训练数据train_loader，优化器optimizer和当前训练周期epoch
def train(args, model, device, train_loader, optimizer, epoch):
    """Training"""
    model.train()#模型进入训练模式
    for batch_idx, (data, target) in enumerate(train_loader):#加载训练数据集合
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print('{{"metric": "Train - NLL Loss", "value": {}}}'.format(
        loss.item()))

#测试模型
def test(args, model, device, test_loader, epoch):
    """Testing"""
    model.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('{{"metric": "Eval - NLL Loss", "value": {}, "epoch": {}}}'.format(
        test_loss, epoch))
    print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
        100. * correct / len(test_loader.dataset), epoch))


def test_image():
    """从args中获取图像。评估过程是否符合MNIST标准，并使用MNIST ConvNet模型进行分类"""
    def get_images_name(folder):
        """Create a generator to list images name at evaluation time"""
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for f in onlyfiles:
            yield f

    def pil_loader(path):
        """Load images from /eval/ subfolder, convert to greyscale and resized it as squared"""
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                sqrWidth = np.ceil(np.sqrt(img.size[0]*img.size[1])).astype(int)
                return img.convert('L').resize((sqrWidth, sqrWidth))

    eval_loader = torch.utils.data.DataLoader(ImageFolder(root=args.evalf, transform=transforms.Compose([
                       transforms.Resize(28),
                       transforms.CenterCrop(28),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), loader=pil_loader), batch_size=1, **kwargs)

    # Name generator
    names = get_images_name(os.path.join(args.evalf, "images"))
    model.eval()
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            label = output.argmax(dim=1, keepdim=True).item()
            print ("Images: " + next(names) + ", Classified as: " + str(label))

# Train?
if args.train:
    # Train + Test per epoch
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    # Do checkpointing - Is saved in outf
    torch.save(model.state_dict(), '%s/mnist_convnet_model_epoch_%d.pth' % (args.outf, args.epochs))

# Evaluate?
if args.evaluate:
    test_image()
