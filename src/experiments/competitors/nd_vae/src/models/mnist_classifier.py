'''Drawn from the following tutorial https://nextjournal.com/gkoehler/pytorch-mnist'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt


#TODO CONVERT TO MODEL(S) FROM PAPER

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1,padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2,padding=0)
        self.conv2_drop = nn.Dropout2d(p=.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

def train_nn_mnist(EPOCHS=1):
  

  n_epochs = EPOCHS
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10

  random_seed = 1
  torch.backends.cudnn.enabled = True
  torch.manual_seed(random_seed)


  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                              ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                              ])),
    batch_size=batch_size_test, shuffle=True)


  examples = enumerate(test_loader)
  batch_idx, (example_data, example_targets) = next(examples)


  network = Net()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
  criterion = nn.CrossEntropyLoss()


  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]      


  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      #loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), '../../model_parameters/mnist_model.pth')
        torch.save(optimizer.state_dict(), '../../model_parameters/mnist_optimizer.pth')
    torch.save(network.state_dict(), '../../model_parameters/fashion_mnist_model.pth')
    torch.save(optimizer.state_dict(), '../../model_parameters/fashion_mnist_optimizer.pth')


  def test():
    network.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = network(data)
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += criterion(output,target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))



  test()
  for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()



  fig = plt.figure()
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.savefig('../../results/mnist_loss')


  with torch.no_grad():
    output = network(example_data)
  fig = plt.figure()
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
      output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
  plt.savefig('../../results/mnist_example_preds.jpg')
  plt.clf()


###############################################################################
###################     Fashion MNIST Version     #############################
###############################################################################


def train_nn_fashion_mnist(EPOCHS=1):
  '''Drawn from the following tutorial https://nextjournal.com/gkoehler/pytorch-mnist'''

  n_epochs = EPOCHS
  batch_size_train = 64
  batch_size_test = 1000
  learning_rate = 0.01
  momentum = 0.5
  log_interval = 10

  random_seed = 1
  torch.backends.cudnn.enabled = True
  torch.manual_seed(random_seed)

  classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')

  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('/files/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                              ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('/files/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                              ])),
    batch_size=batch_size_test, shuffle=True)


  examples = enumerate(test_loader)
  batch_idx, (example_data, example_targets) = next(examples)


  network = Net()
  optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)
  criterion = nn.CrossEntropyLoss()


  train_losses = []
  train_counter = []
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]      


  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data)
      #loss = F.nll_loss(output, target)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), '../../model_parameters/fashion_mnist_model.pth')
        torch.save(optimizer.state_dict(), '../../model_parameters/fashion_mnist_optimizer.pth')

    torch.save(network.state_dict(), '../../model_parameters/fashion_mnist_model.pth')
    torch.save(optimizer.state_dict(), '../../model_parameters/fashion_mnist_optimizer.pth')

  def test():
    network.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = network(data)
        #test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += criterion(output,target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))



  test()
  for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()



  fig = plt.figure()
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.savefig('../../results/fashion_mnist_loss')


  with torch.no_grad():
    output = network(example_data)
  fig = plt.figure()
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
      classes[output.data.max(1, keepdim=True)[1][i].item()]))
    plt.xticks([])
    plt.yticks([])
  plt.savefig('../../results/fashion_mnist_example_preds.jpg')
  plt.clf()



#Un comment to run training through this script
#train_nn_mnist()
#train_nn_fashion_mnist()
