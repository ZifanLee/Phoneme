import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm

VAL_RATIO = 0.2  # 训练集中用于测试的比例
BATCH_SIZE = 64
fix_seed = 0
EPOCH = 10
learning_rate = 0.0001
model_path = './model.pkl'

print('Loading data ...')

data_root = './timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


class TIMITDataset(Dataset):
    def __init__(self, x, y=None):
        self.data = torch.from_numpy(x).float()
        if y is not None:
            y = y.astype(np.int64)
            # astype将numpy数组的元素转换为指定类型元素
            # np已经废弃了int，只能用int64
            # type(x)打印x的数据类型
            self.label = torch.LongTensor(y)
            # torch.LongTensor使用参数创建torch.int64类型的张量数据
        else:
            self.label = None

    def __getitem__(self, item):
        if self.label is not None:
            return self.data[item], self.label[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)


index_for_train = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y = train[:index_for_train], train_label[:index_for_train]
val_x, val_y = train[index_for_train:], train_label[index_for_train:]

print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)

        return x


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


same_seed(fix_seed)
device = get_device()

model = Classifier().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
for epoch in tqdm.tqdm(range(EPOCH)):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for index, data in enumerate(train_loader):
        input, label = data
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, label)
        pred_y = torch.max(output, 1)[1]
        loss.backward()
        optimizer.step()

        train_acc += (pred_y.cpu() == label.cpu()).numpy().astype(np.int64).sum().item()
        # 累加最终得到训练集中预测正确的item数量，除len(tran_set)得到一个epoch中平均准确度
        train_loss += loss.item()
        # 累加得到训练集所有batch的loss和，除len(train_loader)得到一个epoch中平均loss

    # 这个判断仅仅是为了保证验证集非空，避免执行无效代码
    # len(data_set)返回样本集大小
    # len(data_loader)返回batch数量，等于len(data_set)/batch_size
    # 每次epoch训练结束之后进行，使用验证集进行评估
    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                input, label = data
                input, label = input.to(device), label.to(device)
                output = model(input)
                loss = loss_func(output, label)
                pred_y = torch.max(output, 1)[1]

                val_acc += (pred_y.cpu() == label.cpu()).numpy().astype(np.int64).sum().item()
                # 验证集中本次batch中预测正确的item数量，并做累加，最后除len(val_set)得到验证集中预测准确度
                val_loss += loss.item()
                # 累加每个batch的loss，最后除len(val_loader),得到不同batch平均loss_

        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, EPOCH, train_acc / len(train_set), train_loss / len(train_loader), val_acc / len(val_set),
            val_loss / len(val_loader)
        ))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, EPOCH, train_acc / len(train_set), train_loss / len(train_loader)
        ))
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

test_set = TIMITDataset(test, None)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

predict = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        input = data
        input = input.to(device)
        output = model(input)
        predict_y = torch.max(output, 1)[1]

        for y in predict_y.cpu().numpy():
            predict.append(y)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
