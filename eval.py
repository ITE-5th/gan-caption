import os

import torch
from pretrainedmodels import utils
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from e_coco_dataset import ECocoDataset
from evaluator import Evaluator
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))

extractor = Vgg16Extractor(transform=False)

captions_per_image = 2
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
evaluator = Evaluator(corpus).cuda()
dataset = ECocoDataset(corpus, tranform=utils.TransformImage(extractor.cnn), captions_per_image=captions_per_image)
generator = ConditionalGenerator(corpus).cuda()
state_dict = torch.load('./models/generator.pth')
generator.load_state_dict(state_dict['state_dict'])

for param in generator.parameters():
    param.requires_grad = False
generator.eval()
state_dict = torch.load('./models/evaluator.pth')
evaluator.load_state_dict(state_dict['state_dict'])
evaluator.eval()
evaluator.train(False)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
print("Begin Evaluating")
for i, (images, captions, other_captions) in enumerate(dataloader):
    images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()
    images = extractor.forward(images)
    captions = captions.view(-1, 18, captions.shape[-1])
    other_captions = other_captions.view(-1, 18, other_captions.shape[-1])

    k = images.shape[0]
    images = torch.stack([images] * captions_per_image).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])

    captions = pack_padded_sequence(captions, [18] * k * captions_per_image, True)
    other_captions = pack_padded_sequence(other_captions, [18] * k * captions_per_image, True)

    generator_outputs = generator.sample_with_embedding(images)

    evaluator_outputs = evaluator(images, captions)
    generator_outputs = evaluator(images, generator_outputs)
    other_outputs = evaluator(images, other_captions)

    print(f"Real: {evaluator_outputs}, Generated: {generator_outputs}, Others: {other_outputs}")
