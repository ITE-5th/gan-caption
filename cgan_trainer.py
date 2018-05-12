import time
from multiprocessing import cpu_count

import numpy as np
import torch
from pretrainedmodels import utils
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from coco_dataset import CocoDataset
from conditional_generator import ConditionalGenerator
from corpus import Corpus
from evaluator import Evaluator
from evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager
from rl_loss import RLLoss
from vgg16_extractor import Vgg16Extractor

if __name__ == '__main__':
    e_lr = 1e-5
    g_lr = 1e-5
    alpha = 1
    beta = 1
    captions_per_image = 2
    max_length = 17

    torch.manual_seed(2016)
    np.random.seed(2016)
    epochs = 25
    batch_size = 2
    monte_carlo_count = 16
    extractor = Vgg16Extractor(transform=False)
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)
    evaluator = Evaluator.load(corpus, path="models/evaluator-4.pth").cuda()
    generator = ConditionalGenerator.load(corpus, max_sentence_length=max_length, path="models/generator.pth").cuda()

    dataset = CocoDataset(corpus, tranform=utils.TransformImage(extractor.cnn), captions_per_image=captions_per_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    evaluator_criterion = EvaluatorLoss(alpha, beta).cuda()
    generator_criterion = RLLoss().cuda()
    generator.unfreeze()
    evaluator.unfreeze()
    evaluator_optimizer = optim.Adam(evaluator.parameters(), lr=e_lr, weight_decay=1e-5)
    generator_optimizer = optim.Adam(generator.parameters(), lr=g_lr, weight_decay=1e-5)

    print(f"number of batches = {len(dataset) // batch_size}")
    print("Begin Training")
    for epoch in range(epochs):
        start = time.time()
        generator_loss = 0
        evaluator_loss = 0
        for i, (images, captions, other_captions) in enumerate(dataloader, 0):
            print(f"Batch = {i + 1}")
            images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()

            images = extractor.forward(Variable(images))
            captions = captions.view(-1, max_length, captions.shape[-1])
            other_captions = other_captions.view(-1, max_length, other_captions.shape[-1])

            temp = images.shape[0]
            images = torch.stack([images] * captions_per_image).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])

            # generator
            generator.unfreeze()
            evaluator.freeze()
            rewards, props = generator.reward_forward(images, evaluator, monte_carlo_count=monte_carlo_count, steps=2)
            generator_optimizer.zero_grad()
            loss = generator_criterion(rewards, props)
            generator_loss += loss.item()
            loss.backward()
            generator_optimizer.step()

            # evaluator
            evaluator.unfreeze()
            generator.freeze()
            captions = pack_padded_sequence(captions, [max_length] * temp * captions_per_image, True)
            other_captions = pack_padded_sequence(other_captions, [max_length] * temp * captions_per_image, True)
            generator_outputs = generator.sample_with_embedding(images)
            evaluator_outputs = evaluator(images, captions)
            generator_outputs = evaluator(images, generator_outputs)
            other_outputs = evaluator(images, other_captions)
            evaluator_criterion.zero_grad()
            loss = evaluator_criterion(evaluator_outputs, generator_outputs, other_outputs)
            evaluator_loss += loss.item()
            loss.backward()
            evaluator_optimizer.step()
            end = time.time()
            print(f"Batch Time {end - start}")
            start = end
        print(f"Epoch: {epoch + 1}, Loss: {generator_loss + evaluator_loss}, G: {generator_loss}, E:{evaluator_loss}")
        if generator_loss < -50:
            generator.save()
    generator.save()
    evaluator.save()

# generator.save()
# evaluator.save()
