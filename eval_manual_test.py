import os

import torch
from pretrainedmodels import utils

from corpus import Corpus
from evaluator import Evaluator
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))


def round_list(l, f=6):
    return [round(x, f) for x in l]


extractor = Vgg16Extractor(transform=True)
ld_img = utils.LoadImage()

max_length = 17
captions_per_image = 1
corpus = Corpus.load(FilePathManager.resolve("data/corpus-old.pkl"), max_length)
evaluator = Evaluator(corpus).cuda()

models = [
    # "./models/evaluator.pth",
    # "./models/evaluator-0.pth",
    # "./models/evaluator-1.pth",
    # "./models/evaluator-2.pth",
    # "./models/evaluator-3.pth",
    "./models/evaluator-4.pth",
    # "./models/evaluator-7.pth",
    "./models/evaluator1.pth",
    "./models/evaluator2.pth",
    "./models/evaluator3.pth",
    "./models/evaluator4.pth",
    # "./models/evaluator-c99.pth",
]

images = ld_img("./test_images/image_5.jpg")
images = extractor(images)

# def t(s):
captions = torch.cat([corpus.embed_sentence("A room with blue walls and a white sink and door.").unsqueeze(0),
                      corpus.embed_sentence("A colored car parked in front of a garage.").unsqueeze(0)]).cuda()

k = images
images = k.unsqueeze(1).repeat(1, 2, 1).view(-1, images.shape[-1])

for model in models:
    state_dict = torch.load(model)
    evaluator.load_state_dict(state_dict['state_dict'])
    evaluator.eval()
    evaluator_outputs = evaluator(images, captions)

    print(f"{model:35} - Score: {round_list(evaluator_outputs.tolist())}")
