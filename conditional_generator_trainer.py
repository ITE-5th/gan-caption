import os
import time

import torch
from pretrainedmodels import utils
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from file_path_manager import FilePathManager
from g_coco_dataset import GCocoDataset
from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))
extractor = Vgg16Extractor(use_gpu=True, transform=False)
tf_img = utils.TransformImage(extractor.cnn)
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
print("Corpus loaded")

captions_per_image = 2
max_length = 16
batch_size = 32
dataset = GCocoDataset(corpus, transform=tf_img)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = ConditionalGenerator(corpus).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Adam(generator.parameters(), lr=1e-2, weight_decay=1e-5)
pretrained = False
if pretrained:
    st = torch.load("./models/generator.pth")
    generator.load_state_dict(st['state_dict'])
    generator.eval()
    optimizer.load_state_dict(st['optimizer'])
generator.train(True)

epochs = 100
print(f"number of batches = {len(dataset) // batch_size}")
start = time.time()
print("Begin Training")
for epoch in range(epochs):
    epoch_loss = 0
    for i, (images, inputs, targets) in enumerate(dataloader, 0):
        # print(f"Batch = {i}, Time: {time.time() - start}, Loss: {epoch_loss}")

        images = Variable(images).cuda()
        images = extractor.forward(images)

        k = images.shape[0]
        images = torch.stack([images] * captions_per_image).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])
        inputs = inputs.view(-1, max_length, inputs.shape[-1])
        targets = targets.view(-1, max_length)

        inputs = pack_padded_sequence(inputs[:, :-1], [max_length] * captions_per_image * k, True).cuda()
        targets = pack_padded_sequence(targets[:, 1:], [max_length] * captions_per_image * k, True).cuda()[0]

        optimizer.zero_grad()
        outputs = generator.forward(images, inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()

    end = time.time()
    print(f"Epoch: {epoch}, Time: {end - start}, Loss: {epoch_loss}")
    start = end

    torch.save({"state_dict": generator.state_dict(), 'optimizer': optimizer.state_dict()},
               FilePathManager.resolve("models/generator.pth"))
print(f"Epoch = {epoch + 1}")
