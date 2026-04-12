import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os


class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.hardtanh(self.lin2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] #without Badge Dimension
        num = 1
        for s in size:
            num *= s
        return num



netz = MeinNetz()
#netz = netz.cuda()
#print(netz)


if os.path.isfile('meinNetz.pt'):
    netz.load_state_dict(torch.load('meinNetz.pt'))
    netz.eval()
    print("Netz wurde geladen")
else :
    print('wak')
for i in range(100):

    x = [1,0,0,0,1,0,0,0,1,1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    #input = input.cuda()
     #Main Training Loop

    out = netz(input)

    x = [0,1,1,1,0,1,1,1,0,0]
    target = Variable(torch.Tensor([x for _ in range(10)]))
    #target = target.cuda()

    criterion = nn.MSELoss() #um austauschbar zu machen
    loss = criterion(out, target)
    print(loss)

    netz.zero_grad()
    loss.backward()

    optimizer = optim.SGD(netz.parameters(), lr=0.10) #stochastic gradient descent / lr - lernrate von 0.01
    optimizer.step()


torch.save(netz.state_dict(), 'meinNetz.pt')

