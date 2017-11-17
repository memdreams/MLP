import torch
import torch.nn as nn
from torch.nn.init import xavier_normal
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
from scipy import io
import matplotlib.pyplot as plt


def weights_init(m):

    if isinstance(m, nn.Linear):
        xavier_normal(m.weight.data)


def splitData(rawData):
    class1 = rawData.get('class_1').T  # label set to 0
    class2 = rawData.get('class_2').T  # label set to 1
    # print(class1.shape)
    # add bias feature colomn
    # class1 = np.hstack((np.ones((class1.shape[0], 1)), class1))
    # class2 = np.hstack((np.ones((class2.shape[0], 1)), class2))

    train_num = int(0.8 * class1.shape[0])
    test_num = int(0.2 * class1.shape[0])
    # Combine class1 data and class2 data
    x_train = np.vstack((class1[0:train_num], class2[0:train_num]))
    y_train = np.vstack((np.zeros((train_num, 1)), np.ones((train_num, 1))))
    x_test = np.vstack((class1[-test_num:], class2[-test_num:]))
    y_test = np.vstack((np.zeros((test_num, 1)), np.ones((test_num, 1))))

    # # Min-Max Normalization
    # x_train -= x_train.min()
    # x_train /= x_train.max()
    # x_test -= x_test.min()
    # x_test /= x_test.max()
    return x_train, y_train, x_test, y_test

# accuracy
def AccuracyTest(predic, label):
    predic = predic.data.numpy()
    label = label.data.numpy().astype(int)
    # print(predic.shape,label.shape)
    # print('predict:',predic)
    apply = np.vectorize(mapping)
    predict_after_mapping = apply(predic)
    # print('predict:',predic,predict_after_mapping)
    test_np = (predict_after_mapping == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)
def mapping(x):
    if x >= 0.5:
        return 1
    else:
        return 0
class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input = input.view(-1, input.size(0))
        out = self.relu(self.hidden(input))
        out = self.sigmoid(self.predict(out))
        return out

model = MLP(4, 32, 1)
weights_init(model)
# if torch.cuda.is_available():
#     model = MLP.cuda()
print(model)

#method 2
mlp = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
print(mlp)

# loss func and optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)

lossfunc = nn.BCELoss()
torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)


# # test accuarcy
# print(AccuracyTest(
#     np.array([[1,10,6],[0,2,5]],dtype=np.float32),
#     np.array([[1,2,8],[1,2,5]],dtype=np.float32)))
mat = io.loadmat("./observed/classify_d4_k3_saved2.mat")
# iteration = 10000
# learning_rate = 0.05
X_train, y_train, X_test, y_test = splitData(mat)
m, n = X_train.shape

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
# x_train, y_train = Variable(X_train), Variable(y_train)
# x_test, y_test = Variable(X_test), Variable(y_test)
print('training_size:',X_train.size(),y_train.size())
torch_dataset = Data.TensorDataset(data_tensor=X_train, target_tensor=y_train)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

for epoch in range(8):   # 训练所有数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...
        

        inputs = torch.autograd.Variable(batch_x)
        labels = torch.autograd.Variable(batch_y)

        outputs = model.forward(inputs)
        # print('debug:',outputs.size(),labels.size(),type(batch_y))
        loss = lossfunc(outputs,labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(step,":",AccuracyTest(outputs,labels),'loss:',loss.data[0])
        # 打出来一些数据
        # print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

# x, y = Variable(X_train), Variable(y_train)
# x, y = Variable(X_test), Variable(y_test)


# n_data = torch.ones(100,2)
# x0= torch.normal(2*n_data,1)
# y0= torch.zeros(100)
# x1= torch.normal(-2*n_data,1)
# y1= torch.zeros(100)
# x = torch.cat((x0,x1), 0).type(torch.FloatTensor)
# y = torch.cat((y0,y1),).type(torch.LongTensor)
# x, y = Variable(x), Variable(y)


# for t in range(1000):
#     out = model(x)
#     out = mlp(x)
#     loss = lossfunc(out, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if t % 100 == 0:
#         print(t,":", AccuracyTest(out, y))
#         plt.cla()
#         pred = torch.max(nn.functional.softmax(out), 1)[1]
#         pred_y = pred.data.numpy().squeeze()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y, s=100, lw=0)
#         accuracy = sum(pred_y == target_y)/y.data.shape[0]
#         plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
#         plt.pause(0.1)

# plt.ioff()
# plt.show()















