import sys
import torch 
from torch import nn
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

model = nn.Sequential(
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
       # nn.LogSoftmax(dim=1)
    ) #implementing a model using a Sequential structure

optimiser = optim.SGD(model.parameters(), lr=0.003) # implementing optimizer with learning rate = 0.003, SGD stands for stochastic gradient descent

criterion = nn.CrossEntropyLoss() # define a loss

data = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
train, test = random_split(data, [48000, 12000]) #spliting the data on training data and test data in 80/20 proportion

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)

epochs = int(input("Please provide iteration number: "))

training_array_accuracy = []
testing_array_accuracy = []
training_array_loss = []
testing_array_loss = []
repeat_array = []

# training and testing
if(isinstance(epochs, int)):
    for e in range(epochs):
        losses = list()
        accuracy = list()
        for batch in train_loader:
            image, label = batch
            # Flatten MNIST images into a 784 long vector cause 28*28 = 784
            batch_size = image.shape[0]
            image = image.view(batch_size, -1)

            l = model.forward(image) #result of the neural net for one image
            loss = criterion(l, label) # "distance" between performance and the correct label

            # cleaning the gradients
            model.zero_grad()

            # compute the new gradient
            loss.backward()

            # in the opposite direction
            optimiser.step()

            accuracy.append(label.eq(l.detach().argmax(dim=1)).float().mean()) # value on exit of nn is exactly like label
            losses.append(loss.item())
        
        mean_losses_value = torch.tensor(losses).mean()
        mean_accuracy_value = float(torch.tensor(accuracy).mean()) * 100

        print(f'Epoch {e+1}, training loss is: {mean_losses_value:.2f}')
        print(f'Epoch {e+1}, training accuracy is: {round(mean_accuracy_value, 2)}%')
        training_array_accuracy.append(round(float(mean_accuracy_value), 2))
        training_array_loss.append(mean_losses_value)

        #testing
        losses = list()
        accuracy = list()
        for batch in test_loader:
            image, label = batch
            # Flatten MNIST images into a 784 long vector cause 28*28 = 784
            batch_size = image.shape[0]
            image = image.view(batch_size, -1)

            with torch.no_grad():
                l = model.forward(image)
            
            loss = criterion(l, label) # "distance" between performance and the correct label
            
            losses.append(loss.item())
            accuracy.append(label.eq(l.detach().argmax(dim=1)).float().mean())
        
        mean_losses_value = torch.tensor(losses).mean()
        mean_accuracy_value = float(torch.tensor(accuracy).mean()) * 100
    
        print(f'Epoch {e+1}, testing loss is: {mean_losses_value:.2f}')
        print(f'Epoch {e+1}, testing accuracy is: {round(mean_accuracy_value, 2)}%')
        testing_array_accuracy.append(round(float(mean_accuracy_value), 2))
        testing_array_loss.append(mean_losses_value)
    
        repeat_array.append(e+1) 

else:
    print("Wrong parameter")
    sys.exit()

plt.plot(repeat_array, training_array_accuracy)
plt.xlabel('epoch')
plt.ylabel('Accuracy %')
plt.title("Trainig accuracy") # training plot - accuracy
plt.show()

plt.plot(repeat_array, testing_array_accuracy)
plt.xlabel('epoch')
plt.ylabel('Accuracy %')
plt.title("testing accuracy") # testing plot - accuracy
plt.show()

plt.plot(repeat_array, training_array_loss)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title("Training Loss") # training plot - loss
plt.show()

plt.plot(repeat_array, testing_array_loss)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title("Testing Loss") # testing plot - loss
plt.show()