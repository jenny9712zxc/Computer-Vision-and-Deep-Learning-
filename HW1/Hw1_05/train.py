import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tensorboardX import SummaryWriter

from nn_module_vgg19 import VGG19
from torch.autograd import Variable
import matplotlib.pyplot as plt

#參數設置
parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--lr', default=1e-2,help='learning rate')
parser.add_argument('--batch_size',default=256,help='batch size')
parser.add_argument('--epoch',default=40,help='time for ergodic')
parser.add_argument('--pre_epoch',default=0,help='begin epoch')
parser.add_argument('--outf', default='./model', help='folder to output images and model checkpoints') #輸出結果保存路徑
parser.add_argument('--pre_model', default=False,help='use pre-model')#恢復訓練時的模型路徑
args = parser.parse_args()

#使用gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#數據預處理
#圖像預處理和增強
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #先四周填充0，再把圖像隨機裁剪成32*32
    transforms.RandomHorizontalFlip(),  #圖像一半的概率翻轉，一半的概率不翻轉
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

#Cifar-10的標籤
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#模型定義 VGG19
net = VGG19().to(device)

# 定義損失函數和優化方式
criterion = nn.CrossEntropyLoss() #損失函數爲交叉熵，多用於多分類問題
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #優化方式爲mini-batch momentum-SGD，並採用L2正則化（權重衰減）

#使用預訓練模型
if args.pre_model:
    print("Resume from checkpoint...")
    assert os.path.isdir('checkpoint'),'Error: no checkpoint directory found'
    state = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(state['state_dict'])
    best_test_acc = state['acc']
    pre_epoch = state['epoch']
else:
    #定義最優的測試準確率
    best_test_acc = 0
    pre_epoch = args.pre_epoch

#訓練
if __name__ == "__main__":
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        #"val_loss": [],
        "val_acc": []
    }

    writer = SummaryWriter(log_dir='./log')
    print("Start Training, VGG-19...")
    with open("acc.txt","w") as acc_f:
        with open("log.txt","w") as log_f:
            for epoch in range(pre_epoch, args.epoch):
                print('\nEpoch: %d' % (epoch + 1))
                #開始訓練
                net.train()
                print(net)
                #總損失
                sum_loss = 0.0
                #準確率
                accuracy = 0.0
                total = 0.0

                for i, data in enumerate(trainloader):
                    #準備數據
                    length = len(trainloader) #數據大小
                    inputs, labels = data #取出數據
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad() #梯度初始化爲零（因爲一個batch的loss關於weight的導數是所有sample的loss關於weight的導數的累加和）
                    inputs, labels = Variable(inputs), Variable(labels)
                    #forward + backward + optimize
                    outputs = net(inputs) #前向傳播求出預測值
                    loss = criterion(outputs, labels) #求loss
                    loss.backward() #反向傳播求梯度
                    optimizer.step() #更新參數

                    # 每一個batch輸出對應的損失loss和準確率accuracy
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)#返回每一行中最大值的那個元素，且返回其索引
                    total += labels.size(0)
                    accuracy += predicted.eq(labels.data).cpu().sum() #預測值和真實值進行比較，將數據放到cpu上並且求和

                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * accuracy / total))

                    #寫入日誌
                    log_f.write('[epoch:%d, iter:%d] |Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * accuracy / total))
                    log_f.write('\n')
                    log_f.flush()

                #寫入tensorboard
                writer.add_scalar('loss/train',sum_loss / (i + 1), epoch)
                writer.add_scalar('accuracy/train',100. * accuracy / total, epoch)

                H["train_loss"].append(sum_loss / (i + 1))
                #H["train_acc"].append(100. * accuracy / total)
                H["train_acc"].append(100. * accuracy / total)


                #每一個訓練epoch完成測試準確率
                print("Waiting for test...")
                #在上下文環境中切斷梯度計算，在此模式下，每一步的計算結果中requires_grad都是False，即使input設置爲requires_grad=True
                with torch.no_grad():
                    accuracy = 0
                    total = 0
                    for data in testloader:
                        #開始測試
                        net.eval()

                        images, labels = data
                        images, labels = images.to(device), labels.to(device)

                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)#返回每一行中最大值的那個元素，且返回其索引(得分高的那一類)
                        total += labels.size(0)
                        accuracy += (predicted == labels).sum()

                    #輸出測試準確率
                    print('測試準確率爲: %.3f%%' % (100 * accuracy / total))
                    acc = 100. * accuracy / total
                    
                    #寫入tensorboard
                    writer.add_scalar('accuracy/test', acc, epoch)
                    H["val_acc"].append(acc.item())
                    
                    #將測試結果寫入文件
                    print('Saving model...')
                    torch.save(net.state_dict(), '%s/net_%3d.pth' % (args.outf, epoch + 1))
                    acc_f.write("epoch = %03d, accuracy = %.3f%%" % (epoch + 1, acc))
                    acc_f.write('\n')
                    acc_f.flush()

                    #記錄最佳的測試準確率
                    if acc > best_test_acc:
                        torch.save(net.state_dict(), "pretrainedModel.pth")

                        print('Saving Best Model...')
                        #存儲狀態
                        state = {
                            'state_dict': net.state_dict(),
                            'acc': acc,
                            'epoch': epoch + 1,
                        }
                        #沒有就創建checkpoint文件夾
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        #best_acc_f = open("best_acc.txt","w")
                        #best_acc_f.write("epoch = %03d, accuracy = %.3f%%" % (epoch + 1, acc))
                        #best_acc_f.close()
                        torch.save(state, './checkpoint/ckpt.t7')
                        best_test_acc = acc
                        #寫入tensorboard
                        writer.add_scalar('best_accuracy/test', best_test_acc,epoch)
            
            #訓練結束
            print("Training Finished, Total Epoch = %d" % epoch)
            writer.close()

            #print(H["train_loss"])#[1.593853461265564, 1.0795880400538445, 0.8638074445128441]
            #print(H["train_acc"])#[tensor(40.4760), tensor(62.2480), tensor(70.6760)]


            fig, axes = plt.subplots(2)          

            axes[0].plot(H["train_acc"], color='b', label='train')
            axes[0].plot(H["val_acc"], color='r', label='test')    
            axes[1].plot(H["train_loss"],color='b')

            axes[0].set_title("accurancy")
            #axes[1].set_title("epoch")

            axes[0].legend(loc='upper right', shadow=True) 
            #axes[0].set_xlabel("accurancy")
            axes[0].set_ylabel("%")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("loss")

            plt.savefig('plot.png')
            plt.show()



