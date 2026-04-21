import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import os

from tests.test import optimizer  # Make sure this import is correct and needed

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs
)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, **kwargs
)

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=1280, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(x)
        x = x.view(-1, 1280)

        # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
print(torch.cuda.is_available())

model = Netz()


if os.path.isfile('../ZahlenNetz.pt'):
    model.load_state_dict(torch.load('../ZahlenNetz.pt'))
    model.eval()
    print("Netz wurde geladen")
else :
    print('wak')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = data.to(device)
        target = target.to(device)
        # No need for Variable() in modern PyTorch
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}, [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id + 1, len(train_data), 100. * batch_id / len(train_data), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # size average = false , damit loss selber ausgerechnet wird -> data[0], weil aufsummieren
            prediction = output.max(1, keepdim=True)[1]
            correct+= prediction.eq(target.view_as(prediction)).cpu().sum().item() #cdamit prediction potenziell auf gpu ist und das andere auf der cpu -> bei mir noch nicht der fall
        test_loss /= len(test_data.dataset)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        print('Genauigkeit: ', 100.*correct/len(test_data.dataset))
for epoch in range(1, 8):
    train(epoch)
    test()

torch.save(model.state_dict(), '../ZahlenNetz.pt')