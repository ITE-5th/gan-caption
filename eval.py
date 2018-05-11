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


def round_list(l, f=6):
    return [round(x, f) for x in l]


extractor = Vgg16Extractor(transform=False)

max_length = 17
captions_per_image = 2
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)
evaluator = Evaluator(corpus).cuda()
dataset = ECocoDataset(corpus, tranform=utils.TransformImage(extractor.cnn), captions_per_image=captions_per_image)
generator = ConditionalGenerator(corpus, max_sentence_length=max_length).cuda()
state_dict = torch.load('./models/generator.pth')
generator.load_state_dict(state_dict['state_dict'])
generator.eval()

model1 = "./models/evaluator.pth"
# model2 = "./models/evaluator-old2.pth"
# model3 = "./models/evaluator-c49.pth"
# model4 = "./models/evaluator-c99.pth"
# models = [model1, model2, model3, model4]
models = [model1]
batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

for model in models:
    state_dict = torch.load(model)
    evaluator.load_state_dict(state_dict['state_dict'])
    evaluator.eval()
    print(f"Begin Evaluating: {model}")
    for i, (images, captions, other_captions) in enumerate(dataloader):
        images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()
        images = extractor.forward(images)
        captions = captions.view(-1, max_length, captions.shape[-1])
        other_captions = other_captions.view(-1, max_length, other_captions.shape[-1])

        k = images.shape[0]
        images = torch.stack([images] * captions_per_image * k).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])

        captions = pack_padded_sequence(captions, [max_length] * k * captions_per_image, True)
        other_captions = pack_padded_sequence(other_captions, [max_length] * k * captions_per_image, True)

        # generator_outputs = generator.sample(images)
        generator_outputs = generator.sample_with_embedding(images)
        # print(f"generated: {generator_outputs}")
        # generator_outputs = corpus.embed_sentence(generator_outputs).unsqueeze(0).cuda()#.repeat(2, 1, 1).cuda()

        evaluator_outputs = evaluator(images, captions)
        generator_outputs = evaluator(images, generator_outputs)
        other_outputs = evaluator(images, other_captions)

        print(f"Real: {round_list(evaluator_outputs.tolist())}, "
              f"Generated: {round_list(generator_outputs.tolist())}, "
              f"Others: {round_list(other_outputs.tolist())}")
