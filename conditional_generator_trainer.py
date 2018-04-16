import os
from os import cpu_count

import torch
from pretrainedmodels import utils
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from coco_dataset import CocoDataset
from conditional_generator import ConditionalGenerator
from corpus import Corpus
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

# torch.backends.cudnn.enabled = False
if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))
extractor = Vgg16Extractor(use_gpu=True, transform=False)
tf_img = utils.TransformImage(extractor.cnn)
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
print("Corpus loaded")

batch_size = 2
dataset = CocoDataset(corpus, transform=tf_img)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())

generator = ConditionalGenerator(corpus).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Adam(generator.parameters(), lr=0.0001, weight_decay=1e-5)

epochs = 20

print("Begin Training")
for epoch in range(epochs):
    for i, (images, inputs, targets) in enumerate(dataloader, 0):
        print(f"Batch = {i + 1}")
        images = Variable(images).cuda()
        inputs = pack_padded_sequence(inputs, 17, True).cuda()
        targets = pack_padded_sequence(targets, 17, True).cuda()

        images = extractor(images)

        optimizer.zero_grad()
        outputs = generator.forward(images, inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        torch.save({"state_dict": generator.state_dict()}, FilePathManager.resolve("models/generator.pth"))
    print(f"Epoch = {epoch + 1}")
