import os
import time

import torch
from pretrainedmodels import utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from file_path_manager import FilePathManager
from g_coco_dataset import GCocoDataset
from iterator import Iterator
from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))
extractor = Vgg16Extractor(use_gpu=True, transform=False)
tf_img = utils.TransformImage(extractor.cnn)
max_length = 16
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)
print("Corpus loaded")

captions_per_image = 1
batch_size = 2
dataset = GCocoDataset(corpus, transform=tf_img, captions_per_image=captions_per_image)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = ConditionalGenerator(corpus, hidden_size=1024, max_sentence_length=max_length).cuda()
generator.train(True)
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = Adam(generator.parameters(), lr=4e-4, betas=(0.8, 0.999), weight_decay=1e-5)
it = Iterator(dataloader, 10)
epochs = 1
print(f"number of batches = {len(dataset) // batch_size}")
start = time.time()
print("Begin Training")
for epoch in range(epochs):
    epoch_loss = 0
    it.reset()
    # for i, (images, inputs, targets) in enumerate(dataloader):
    for i, (images, inputs, targets) in enumerate(it):
        images = extractor.forward(images.cuda())

        k = images.shape[0]
        inputs = inputs.view(-1, max_length, inputs.shape[-1])
        targets = targets.view(-1, max_length)

        inputs = pack_padded_sequence(inputs[:, :-1], [max_length] * k, True).cuda()
        targets = pack_padded_sequence(targets[:, 1:], [max_length] * k, True).cuda()[0]
        # targets = targets[:, 1:].cuda()

        optimizer.zero_grad()
        outputs = generator.forward(images, inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    end = time.time()
    print(f"Epoch: {epoch}, Time: {end - start}, Loss: {epoch_loss}")
    start = end

generator.save()
