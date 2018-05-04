import os
import time

import torch
from pretrainedmodels import utils
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from g_coco_dataset import GCocoDataset
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
dataset = GCocoDataset(corpus, transform=tf_img)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = ConditionalGenerator(corpus).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=corpus.word_index(corpus.PAD), size_average=True).cuda()
optimizer = Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5)
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
for epoch in range(1, epochs):
    for i, (images, inputs, targets) in enumerate(dataloader, 0):
        print(f"Batch = {i + 1}")
        images = Variable(images).cuda()
        images = extractor.forward(images)
        for k in range(inputs.shape[1]):
            input = Variable(inputs[:, k, :-1, :]).cuda()
            target = Variable(targets[:, k, 1:]).cuda()

            input = pack_padded_sequence(input, [17] * input.shape[0], True)
            target = pack_padded_sequence(target, [17] * target.shape[0], True)[0]

            optimizer.zero_grad()
            outputs = generator.forward(images, input)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    end = time.time()
    print(end - start)
    start = end

    torch.save({"state_dict": generator.state_dict(), 'optimizer': optimizer.state_dict()},
               FilePathManager.resolve("models/generator.pth"))
print(f"Epoch = {epoch + 1}")
