import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os 
dir = "C:\\Users\\pengz\\Desktop\\taskthree\\task2\\"
os.chdir(dir )

from torch.utils.data import Dataset


# 导入数据集的类
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.lines = open(csv_file).readlines()
        
    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].split(',')
        cur_line1 = self.lines[index].split(',')

        sin_input = np.float32(np.vstack((cur_line[0:15],cur_line1[0:15])))
        cos_output = np.float32(np.vstack((cur_line[15:21],cur_line1[15:21])))

        return sin_input, cos_output

    def __len__(self):
        return len(self.lines)  # MyDataSet的行数

from torch.utils.data import Dataset
time_step = 8
# 导入数据集的类
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.lines = open(csv_file).readlines()
        
    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].split(',')
        sin_input=np.float32(cur_line[0:15])
        cos_output = np.float32(cur_line[15:21])
        for i in range(time_step-1):
          cur_line = self.lines[index+1+i].split(',')
          sin_input = np.vstack((sin_input,cur_line[0:15]))
          cos_output = np.vstack((cos_output,cur_line[15:21]))

        return np.float32(sin_input),  np.float32(cos_output)

    def __len__(self):
        return int(len(self.lines)/time_step)  # MyDataSet的行数

from torch import nn


class Rnn(nn.Module):
    def __init__(self, input_num=15, hidden_num=32, layer_num=3, output_num=6, seq_len=1000):
        super(Rnn, self).__init__()
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.seq_len = seq_len  # 序列长度

        self.rnn = nn.RNN(
            input_size=input_num,
            hidden_size=hidden_num,
            num_layers=layer_num,
            nonlinearity='relu',
            batch_first=True  # 输入(batch, seq, feature)
        )

        self.Out = nn.Linear(hidden_num, output_num)

    def forward(self, u, h_state):
        """
        :param u: input输入
        :param h_state: 循环神经网络状态量
        :return:
        """
        # print(u.shape)
        r_out, h_state_next = self.rnn(u, h_state)
        # print(r_out.shape)
        # r_out_reshaped = r_out.view(-1,2, self.hidden_num)  # to 2D data
        # print(r_out_reshaped.shape)
        outs = self.Out(r_out)
        # print(outs.shape)
        outs = outs.view(-1, self.seq_len, self.output_num)  # to 3D data
        # print(outs.shape)
        return outs, h_state_next

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# batch size
batch_size_train = 100
# total epoch(总共训练多少轮)
total_epoch = 50

# 1. 导入训练数据
filename1 =dir + 'alldata.csv'

dataset_train = MyDataset(filename1)
train_size = int(len(dataset_train)*0.8)
test_size = len(dataset_train) - train_size
train_dataset,  test_dataset = torch.utils.data.random_split(dataset_train, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)

# 2. 构建模型，优化器
rnn = Rnn(seq_len=batch_size_train).to(device)
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01, momentum=0.8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()  # mean square error
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 1  # 网络训练过程中最大的loss


# 3. 模型训练
def train_rnn(epoch):
    hidden_state = None  # 隐藏状态初始化
    global total_loss
    mode = True
    rnn.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和

    for idx, (sin_input, cos_output) in enumerate(train_loader):
        sin_input_np = sin_input.numpy()  # 1D
        cos_output_np = cos_output.numpy()  # 1D
        # print(sin_input_np.shape)
        sin_input_torch = Variable(torch.from_numpy(sin_input_np))  # 3D
        cos_output_torch = Variable(torch.from_numpy(cos_output_np))  # 3D
        prediction, hidden_state = rnn(sin_input_torch.to(device), hidden_state)

        # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错!!!
        hidden_state = Variable(hidden_state.data).to(device)
        # print(prediction.transpose(1,0).shape,cos_output_torch.shape)
        loss = criterion(prediction.transpose(1,0), cos_output_torch.to(device))  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients

        loss_epoch += loss.item()  # 将每个batch的loss累加，直到所有数据都计算完毕
        if idx == len(train_loader) - 1:
            print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
            train_loss_list.append(loss_epoch)
            if loss_epoch < total_loss:
                total_loss = loss_epoch
                torch.save(rnn, '..\\model\\rnn_model.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    print("Start Training...")
    for i in range(total_epoch):  # 模型训练1000轮
        train_rnn(i)
    torch.save(rnn, dir + '/rnn_model3.pkl')
    print("Stop Training!")

plt.plot(train_loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(dir +'/trainloss.jpg')

batch_size_test = 100

# 导入数据
filename1 = dir + 'alldata.csv'

test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=True)

criterion = nn.MSELoss()  # mean square error

import matplotlib.pyplot as plt


# rnn 测试
def test_rnn():
    net_test = Rnn(seq_len=batch_size_test).to(device)
    net_test = torch.load(dir + 'rnn_model3.pkl')  # load model
    hidden_state = None
    test_loss = 0
    net_test.eval()
    with torch.no_grad():
        for idx, (sin_input, cos_output) in enumerate(test_loader):
            sin_input_np = sin_input.numpy()  # 1D
            cos_output_np = cos_output.numpy()  # 1D
            sin_input_torch = Variable(torch.from_numpy(sin_input_np))  # 3D
            cos_output_torch = Variable(torch.from_numpy(cos_output_np))  # 3D
            prediction, hidden_state = net_test(sin_input_torch.to(device), hidden_state)
            # print(prediction.shape)
            prediction = prediction.transpose(1,0)
            if idx == 0:
                predict_value = prediction.squeeze()
                real_value = cos_output_torch.squeeze()
            else:
                predict_value = torch.cat([predict_value, prediction.squeeze()], dim=0)
                real_value = torch.cat([real_value, cos_output_torch.squeeze()], dim=0)
            loss = criterion(prediction,cos_output_torch.to(device))
            test_loss += loss.item()

    print('Test set: Avg. loss: {:.9f}'.format(test_loss))
    return predict_value, real_value


if __name__ == '__main__':
    # 模型测试
    print("testing...")
    p_v, r_v = test_rnn()
    print(r_v.shape)
    # 对比图
    data = p_v[350:400,1,:].reshape(-1,6)
    data1 = r_v[350:400,1,:].reshape(-1,6)
    plt.plot(data[:,4].cpu(), c='green') 

    plt.plot(data1[:,4].cpu(), c='orange')
    plt.ylabel('prediction')
    plt.xlabel('data')
    plt.yticks(np.arange(0,1,0.05))

    plt.savefig(dir + '训练50次第O3气体气体50时间检测结果.jpg')

    plt.show()
    print("stop testing!")







