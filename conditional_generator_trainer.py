import os
import time

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
# torch.backends.cudnn.enabled = False
from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))
extractor = Vgg16Extractor(use_gpu=True, transform=False)
tf_img = utils.TransformImage(extractor.cnn)
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
print("Corpus loaded")

batch_size = 32
dataset = CocoDataset(corpus, transform=tf_img)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

generator = ConditionalGenerator(corpus).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=3).cuda()
optimizer = Adam(generator.parameters(), lr=0.0001, weight_decay=1e-5)

epochs = 20

start = time.time()
print("Begin Training")
for epoch in range(1, epochs):
    for i, (images, inputs, targets) in enumerate(dataloader, 0):
        print(f"Batch = {i + 1}")
        images = Variable(images).cuda()
        images = extractor.forward(images)
        for k in range(5):
            input = Variable(inputs[k])[:, :-1, :].cuda()
            target = Variable(targets[k])[:, 1:].cuda()

            input = pack_padded_sequence(input, [17] * len(images), True)

            optimizer.zero_grad()
            outputs = generator.forward(images, input)

            loss = criterion(outputs, target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
    end = time.time()
    print(end - start)
    start = end

    torch.save({"state_dict": generator.state_dict()}, FilePathManager.resolve("models/generator.pth"))
print(f"Epoch = {epoch + 1}")
