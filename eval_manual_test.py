import os

import torch

from corpus import Corpus
from evaluator import Evaluator
from file_path_manager import FilePathManager

# from vgg16_extractor import Vgg16Extractor

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))


def round_list(l, f=6):
    return [round(x, f) for x in l]


# extractor = Vgg16Extractor(transform=True)
# ld_img = utils.LoadImage()

max_length = 16
captions_per_image = 1
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)
evaluator = Evaluator(corpus).cuda()

models = [
    # "./models/evaluator.pth",
    # "./models/evaluator-0.pth",
    # "./models/evaluator-1.pth",
    # "./models/evaluator-2.pth",
    # "./models/evaluator-3.pth",
    # "./models/evaluator-4.pth",
    # "./models/evaluator-7.pth",
    # "./models/evaluator1.pth",
    # "./models/evaluator2.pth",
    # "./models/evaluator3.pth",
    # "./models/evaluator4.pth",
    # "./models/evaluator-c99.pth",
    "./models/evaluator-0.pth",
    "./models/evaluator-4.pth",
]
# def extract(path):
#     image=  ld_img(path + ".jpg")
#     image = extractor(image)
#     print(image.shape)
#     torch.save(image, path + ".pth")

# images = ld_img("./test_images/image_5.jpg")
# images = extractor(images)
images = torch.load("./test_images/image_1.pth")
# def t(s):
captions = torch.cat([corpus.embed_sentence("A room with blue walls and a white sink and door.").unsqueeze(0),
                      corpus.embed_sentence("A room with a white sink and door and blue walls.").unsqueeze(0),
                      corpus.embed_sentence("A room with a blue sink and door and white walls.").unsqueeze(0),
                      corpus.embed_sentence("A room with a sink, door and walls.").unsqueeze(0),
                      corpus.embed_sentence("A dirty house but dark kids.").unsqueeze(0),
                      corpus.embed_sentence("A colored car parked in front of a garage.").unsqueeze(0),

                      corpus.embed_sentence("A group of giraffes standing next to each other.").unsqueeze(0),
                      corpus.embed_sentence("A group of giraffes next to a fence.").unsqueeze(0),
                      corpus.embed_sentence("A group of zebras next to a fence.").unsqueeze(0),
                      ]).cuda()

k = images
images = k.unsqueeze(1).repeat(1, captions.shape[0], 1).view(-1, images.shape[-1])

for model in models:
    state_dict = torch.load(model)
    evaluator.load_state_dict(state_dict['state_dict'])
    evaluator.eval()
    evaluator_outputs = evaluator(images, captions)

    print(f"{model:35} - Score: {round_list(evaluator_outputs.tolist())}")
