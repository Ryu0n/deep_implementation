"""
두 번째 단어를 입력으로 세 번째 단어가 무엇이 나올지 예측
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sentences = ['i like dog',
             'i love coffee',
             'i hate milk',
             'you like cat',
             'you love milk',
             'you hate coffee']
dtype = torch.float

word_list = list(set(' '.join(sentences).split()))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}

# 모든 단어의 종류 수 : 9
n_class = len(word_dict)

# 문장의 수 (샘플의 수) : 6
batch_size = len(sentences)

# 은닉층 사이즈
n_hidden = 5


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sentence in sentences:
        words = sentence.split()
        input_ = [word_dict[word] for word in words[:-1]]
        target_ = word_dict[words[-1]]

        """
        np.eye(n_class)[[7, 0]] -> [[0. 0. 0. 0. 0. 0. 0. 1. 0.] 
                                    [1. 0. 0. 0. 0. 0. 0. 0. 0.]]
        """
        input_batch.append(np.eye(n_class)[input_])  # One-Hot Encoding
        target_batch.append(target_)

    return input_batch, target_batch


input_batch, target_batch = make_batch(sentences)
input_batch = torch.tensor(input_batch, dtype=torch.float32, requires_grad=True)
target_batch = torch.tensor(target_batch, dtype=torch.int64)

print(input_batch.shape)  # N x T x D : (6, 2, 9)
print(target_batch.shape)  # (6,)


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.3)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(dtype))  # (5, 9)
        self.b = nn.Parameter(torch.randn([n_class]).type(dtype))  # (9,)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, hidden):
        # switch dim 0 and 1
        X = X.transpose(0, 1)                  # (6, 2, 9) -> (2, 6, 9)
        outputs, hidden = self.rnn(X, hidden)  # X : (2, 6, 9), hidden : (1, 6, 5)
        # outputs : (2, 6, 5) (output_node x batch_size x n_hidden)
        outputs = outputs[-1]                  # outputs : (6, 5) (맨 마지막 노드의 출력)
        model = torch.mm(outputs, self.W) + self.b
        return model


model = TextRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)  # 양방향일시 zeros(2,
    output = model(input_batch, hidden)  # input_batch : (6, 2, 9), hidden : (1, 6, 5)
    loss = criterion(output, target_batch)  # output : (6, 9), target_batch : (6,)

    if (epoch + 1) % 100 == 0:
        print('Epoch : ', '%04d' % (epoch + 1), 'Cost : ', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hidden = torch.zeros(1, batch_size, n_hidden, requires_grad=True)
predict = model(input_batch, hidden).data  # (6, 9)
predict = predict.max(axis=1, keepdim=True)[1]  # (6, 1)
print([number_dict[n.item()] for n in predict.squeeze()])

a = torch.randn(3, 4)
print(a)
"""
tensor([[ 0.7745, -1.2034,  0.7053,  0.0947],
        [ 1.3365, -0.0998, -0.0091,  0.1973],
        [-0.1648,  0.9201,  0.8369,  0.2715]])
"""
print(torch.max(a, 0))
"""
torch.return_types.max(
values=tensor([1.3365, 0.9201, 0.8369, 0.2715]),
indices=tensor([1, 2, 2, 2]))
"""
