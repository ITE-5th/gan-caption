import time
from multiprocessing import cpu_count

import numpy as np
import torch
from pretrainedmodels import utils
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from captions import Captions
from conditional_generator import ConditionalGenerator
from corpus import Corpus
from dataset_test import CocoDataset
from evaluator import Evaluator
from evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager
from rl_loss import RLLoss
from vgg16_extractor import Vgg16Extractor

if __name__ == '__main__':
    e_lr = 4e-4
    g_lr = 4e-4
    alpha = 1
    beta = 1
    captions_per_image = 1
    max_length = 16

    torch.manual_seed(2016)
    np.random.seed(2016)
    epochs = 20
    batch_size = 32
    monte_carlo_count = 16
    extractor = Vgg16Extractor(transform=False)
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)
    evaluator = Evaluator.load(corpus, path="models/evaluator-4.pth").cuda()
    generator = ConditionalGenerator.load(corpus, max_sentence_length=max_length, path="models/generator.pth").cuda()

    dataset = CocoDataset(tranform=utils.TransformImage(extractor.cnn))
    captions_loader = Captions(dataset, corpus, captions_per_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    evaluator_criterion = EvaluatorLoss(alpha, beta).cuda()
    generator_criterion = RLLoss().cuda()
    generator.unfreeze()
    evaluator.unfreeze()
    evaluator_optimizer = optim.Adam(evaluator.parameters(), lr=e_lr, betas=(0.8, 0.999), weight_decay=1e-5)
    generator_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.8, 0.999), weight_decay=1e-5)

    print(f"number of batches = {len(dataset) // batch_size}")
    print("Begin Training")
    for epoch in range(epochs):
        print(epoch)
        start = time.time()

        # generator
        generator.unfreeze()
        evaluator.freeze()
        generator.train(True)

        generator_loss = 0

        # d = iter(dataloader)
        # while True:
        #     images = next(d, None)
        #     images2 = next(d, None)
        #     if images is None:
        #         break
        #
        #     images = images[0]
        #     if images2 is not None:
        #         images2 = images2[0]
        #         images = torch.cat([images, images2], dim=0)

        for images, _ in dataloader:
            images = images.cuda()
            images = extractor.forward(Variable(images))

            rewards, props = generator.reward_forward(images, evaluator, monte_carlo_count=monte_carlo_count, steps=1)
            generator_optimizer.zero_grad()
            loss = generator_criterion(rewards, props)
            generator_loss += loss.item()
            loss.backward()
            generator_optimizer.step()

        # # evaluator
        # evaluator.unfreeze()
        # generator.freeze()
        # evaluator.train(True)
        # evaluator_loss = 0
        # for i, (images, indices) in enumerate(dataloader):
        #     captions, other_captions = captions_loader.get_captions(indices)
        #     images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()
        #
        #     images = extractor.forward(Variable(images))
        #     images = images.unsqueeze(1).repeat(1, captions_per_image, 1).view(-1, images.shape[-1])
        #
        #     captions = captions.view(-1, max_length, captions.shape[-1])
        #     other_captions = other_captions.view(-1, max_length, other_captions.shape[-1])
        #
        #     generator_outputs = generator.sample_with_embedding(images)
        #     evaluator_outputs = evaluator(images, captions)
        #     generator_outputs = evaluator(images, generator_outputs)
        #
        #     other_outputs = evaluator(images, other_captions)
        #     evaluator_criterion.zero_grad()
        #     loss = evaluator_criterion(evaluator_outputs, generator_outputs, other_outputs)
        #     evaluator_loss += loss.item()
        #     loss.backward()
        #     evaluator_optimizer.step()
        #
        # end = time.time()
        # print(f"Epoch: {(epoch + 1):1}, Time: {(end - start):1.0f}, "
        #       f"Loss: {(generator_loss + evaluator_loss):6.4f}, "
        #       f"G: {generator_loss:3.4f}, E:{evaluator_loss:3.4f}")
        #
        # start = end

    generator.save()
    # evaluator.save()
