import os
import time
from multiprocessing import cpu_count

import torch
from pretrainedmodels import utils
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from e_coco_dataset import ECocoDataset
from evaluator import Evaluator
from evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))

    extractor = Vgg16Extractor(transform=False)
    all_losses = []
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    evaluator = Evaluator(corpus).cuda()
    generator = ConditionalGenerator.load(corpus, training=False).cuda()
    dataset = ECocoDataset(corpus, tranform=utils.TransformImage(extractor.cnn))
    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    criterion = EvaluatorLoss(1, 1).cuda()
    optimizer = optim.Adam(evaluator.parameters(), lr=1e-5, weight_decay=1e-5)
    epochs = 5
    print(f"number of batches = {len(dataset) // batch_size}")

    print("Begin Training")
    epoch_loss = 0
    start = time.time()
    for epoch in range(epochs):
        for i, (images, captions, other_captions) in enumerate(dataloader):
            if i % 100 == 0:
                print(f"Batch = {i + 1}, Time: {time.time() - start}")

            print(f"Batch: {i + 1}, Elapsed Time: {time.time() - start}")
            images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()
            images = extractor(Variable(images))
            captions = captions.view(-1, 18, captions.shape[-1])
            other_captions = other_captions.view(-1, 18, other_captions.shape[-1])

            k = images.shape[0]
            images = torch.stack([images] * 5).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])

            captions = pack_padded_sequence(captions, [18] * k * 5, True)
            other_captions = pack_padded_sequence(other_captions, [18] * k * 5, True)
            optimizer.zero_grad()
            generator_outputs = generator.sample_with_embedding(images)
            evaluator_outputs = evaluator(images, captions)
            generator_outputs = evaluator(images, generator_outputs)
            other_outputs = evaluator(images, other_captions)
            loss = criterion(evaluator_outputs, generator_outputs, other_outputs)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        end = time.time()
        all_losses.append(epoch_loss)
        print(f"Epoch: {epoch}, Time: {end - start}, Loss: {all_losses[-1]}")
        start = end
        file_name = f"evaluator-c{epoch}.pth"
        torch.save({"state_dict": evaluator.state_dict(), 'optimizer': optimizer.state_dict()},
                   FilePathManager.resolve(f"models/{file_name}"))
