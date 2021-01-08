import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset
import os
from modules import Net

# This script feeds a CNN a melspectogram and compares the result to the target audio

class AudioDataset(Dataset):
    # Constructs a dataset
    def __init__(self, filePathInput, filePathOutput, fileNamesInput, fileNamesOutput):
        # init vars
        self.fileNamesInput = fileNamesInput
        self.fileNamesOutput = fileNamesOutput
        self.filePathInput = filePathInput
        self.filePathOutput = filePathOutput

    def __getitem__(self, index):
        # compute filename based on index
        fileName = self.filePathInput + str(self.fileNamesInput[index])
        # load sound data into Tensor
        sound, sr = torchaudio.load(fileName)
        sound = torchaudio.transforms.MelSpectrogram()(sound[0])
        fileName = self.filePathOutput + str(self.fileNamesOutput[index])
        target, sr = torchaudio.load(fileName)
        
        target = target[0]
        # return Tensor of the sound file
        return sound, target
    
    def __len__(self):
        return len(self.fileNamesInput)

# declared list to store filenames
cleanFileNames = []
# iterate through files in the folder and add them to the list
for filename in os.listdir('target/'):
    cleanFileNames.append(filename)
# shuffle the list
random.shuffle(cleanFileNames)
# copy list
convolvedFileNames = cleanFileNames.copy()
index = 0
# alter names
for string in convolvedFileNames:
    convolvedFileNames[index] = 'v' + string[1:]
    index = index + 1

# find the split point in the list for 70:30-train:test sets
splitPoint = int(np.floor(len(cleanFileNames) * 0.9))

# split data lists
trainConvolvedFileNames = convolvedFileNames[0:splitPoint]
trainCleanFileNames = cleanFileNames[0:splitPoint]
testConvolvedFileNames = convolvedFileNames[splitPoint + 1:len(convolvedFileNames)]
testCleanFileNames = cleanFileNames[splitPoint + 1:len(cleanFileNames)]
# Construct DataLoader
train_loader = AudioDataset('input/', 'target/', trainConvolvedFileNames, trainCleanFileNames)
test_loader = AudioDataset('input/', 'target/', testConvolvedFileNames, testCleanFileNames)

# make sure lengths fit alright
print('train length = {}'.format(len(train_loader)))
print('test length = {}'.format(len(test_loader)))

# init dataloaders
train_loader = torch.utils.data.DataLoader(train_loader, batch_size = 1, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size = 1, shuffle = True)

print('DataLoaders Constructed')

# init model (to GPU)
model = Net(1, 1).cuda()

# variables
nTotalSteps = len(train_loader)
numEpochs = 1
learning_rate = 0.001

# loss and optimiser definitions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

lossStorage = []

# training loop
for epoch in range(numEpochs):
    print(f"epoch {epoch}")
    for i, (inputs, target) in enumerate(train_loader):
        # reshape for conv1d lyr
        # tensor.shape() = [1, 1, 40000]
        # "         "    = [1 batch, 1channel, 40000frames]
        # print(f'input.shape: {inputs.shape}')
        inputs = torch.unsqueeze(inputs, 0)
        # print(f'input.shape: {inputs.shape}')
        inputs = inputs.cuda()
        target = target.cuda()
        # forward pass
        outputs = model(inputs).cuda()
        
        # print(f'output.shape: {outputs.shape}')
        # print(f'input.shape: {target.shape}')
        loss = criterion(outputs, target).cuda()

        # back pass
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            lossStorage.append(loss.item())

        if (i + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{numEpochs}, loss = {loss.item():.4f}")

steps = np.linspace(1, len(lossStorage), len(lossStorage))
plt.plot(steps, lossStorage)
plt.show()

# # testing loop
# with torch.no_grad():
#     nCorrect = 0
#     nSamples = 0

#     # for inputs, target in test_loader:
