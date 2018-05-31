import os

import torch
from pretrainedmodels import utils
from torch.utils.data import DataLoader

from captions import Captions
from conditional_generator import ConditionalGenerator
from corpus import Corpus
from dataset_test import CocoDataset
from evaluator import Evaluator
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))


def round_list(l, f=6):
    return [round(x, f) for x in l]


extractor = Vgg16Extractor(transform=False)

max_length = 17
captions_per_image = 1
corpus = Corpus.load(FilePathManager.resolve("data/corpus-old.pkl"), max_length)
evaluator = Evaluator(corpus).cuda()
dataset = CocoDataset(tranform=utils.TransformImage(extractor.cnn))

captions_loader = Captions(dataset, corpus, captions_per_image)

generator = ConditionalGenerator(corpus, max_sentence_length=max_length).cuda()
state_dict = torch.load('./models/generator.pth')
generator.load_state_dict(state_dict['state_dict'])
generator.eval()

models = [
    # "./models/evaluator-0.pth",
    # "./models/evaluator-1.pth",
    # "./models/evaluator-2.pth",
    # "./models/evaluator-3.pth",
    "./models/evaluator-4.pth",
    # "./models/evaluator-7.pth",
    # "./models/evaluator1.pth",
    # "./models/evaluator2.pth",
    # "./models/evaluator3.pth",
    "./models/evaluator4.pth",
    # "./models/evaluator-c99.pth",
    # "./models/evaluator.pth",
    "./models/Evaluator.pth",
]
batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

for i, (images, indices) in enumerate(dataloader):
    print("======================================")
    captions, other_captions = captions_loader.get_captions(indices)
    images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()
    images = extractor.forward(images)
    captions = captions.view(-1, max_length, captions.shape[-1])
    other_captions = other_captions.view(-1, max_length, other_captions.shape[-1])
    images = images.unsqueeze(1).repeat(1, captions_per_image, 1).view(-1, images.shape[-1])

    # k = images.shape[0]
    # images = torch.stack([images] * captions_per_image * k).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])

    # captions = pack_padded_sequence(captions, [max_length] * k * captions_per_image, True)
    # other_captions = pack_padded_sequence(other_captions, [max_length] * k * captions_per_image, True)

    # generator_outputs = generator.sample(images)
    generator_outputs = generator.sample_with_embedding(images)
    # print(f"generated: {generator_outputs}")
    # generator_outputs = corpus.embed_sentence(generator_outputs).unsqueeze(0).cuda()#.repeat(2, 1, 1).cuda()

    for model in models:
        state_dict = torch.load(model)
        evaluator.load_state_dict(state_dict['state_dict'])
        evaluator.eval()
        evaluator_outputs = evaluator(images, captions)
        generator_output = evaluator(images, generator_outputs)
        other_outputs = evaluator(images, other_captions)

        print(f"{model:35}:"
              f"Real: {round_list(evaluator_outputs.tolist())}, "
              f"Generated: {round_list(generator_output.tolist())}, "
              f"Others: {round_list(other_outputs.tolist())}")
