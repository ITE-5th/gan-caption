import time

import torch
import torchvision.datasets as dset
from pretrainedmodels import utils
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from captions import Captions
from conditional_generator import ConditionalGenerator
from corpus import Corpus
from evaluator import Evaluator
from evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor


class CocoDataset(Dataset):

    def __init__(self, tranform=None):
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=tranform)

        self.length = len(self.captions)
        self.s = set(range(self.length))

    def __getitem__(self, index):
        return self.captions[index][0], index

    def __len__(self):
        return self.length


if __name__ == '__main__':
    lr = 1e-4
    alpha = 1
    beta = 1
    captions_per_image = 3
    max_length = 17

    epochs = 50
    batch_size = 36
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)

    extractor = Vgg16Extractor(transform=False)
    # extractor.cnn = copy.deepcopy(vgg)
    # extractor.cnn.eval()
    # extractor.cnn.cuda()

    generator = ConditionalGenerator(corpus=corpus, max_sentence_length=max_length).cuda()
    state_dict = torch.load(FilePathManager.resolve('models/generator.pth'))
    generator.load_state_dict(state_dict['state_dict'])
    generator.eval()

    evaluator = Evaluator(corpus).cuda()
    criterion = EvaluatorLoss(alpha, beta).cuda()
    optimizer = optim.Adam(evaluator.parameters(), lr=lr, betas=(0.8, 0.999), weight_decay=1e-5)

    dataset = CocoDataset(tranform=utils.TransformImage(extractor.cnn))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    captions_loader = Captions(dataset, corpus, captions_per_image)
    all_losses = []

    print("Begin Training")
    epoch_loss = 0
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, indices) in enumerate(dataloader):
            # if i % 100 == 0:
            #     print(f"Batch = {i}, Time: {time.time() - start}, Loss: {epoch_loss}")
            captions, other_captions = captions_loader.get_captions(indices)

            images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()

            images = extractor.forward(Variable(images))
            images = images.unsqueeze(1).repeat(1, captions_per_image, 1).view(-1, images.shape[-1])
            generator_outputs = generator.sample_with_embedding(images)

            captions = captions.view(-1, max_length, captions.shape[-1])
            other_captions = other_captions.view(-1, max_length, other_captions.shape[-1])

            optimizer.zero_grad()

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
    torch.save({"state_dict": evaluator.state_dict(), 'losses': all_losses,
                'alpha': alpha, 'beta': beta, 'lr': lr},
               FilePathManager.resolve(f"models/{file_name}"))

    print(all_losses)
