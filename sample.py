import torch
from pretrainedmodels import utils

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

extractor = Vgg16Extractor()
load_img = utils.LoadImage()
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
generator = ConditionalGenerator(corpus=corpus, max_sentence_length=16).cuda()
state_dict = torch.load(FilePathManager.resolve('models/generator00001111.pth'))
generator.load_state_dict(state_dict['state_dict'])
generator.eval()

image_folder = FilePathManager.resolve("test_images/")
# images = glob.glob(image_folder + "*")
image1 = image_folder + "image_1.png"
image2 = image_folder + "image_2.jpg"
image3 = image_folder + "image_3.jpg"
image4 = image_folder + "image_4.jpg"
image5 = image_folder + "image_5.jpg"
image6 = image_folder + "image_6.jpg"
images = [image1, image2, image3, image4, image5, image6]
print("Greedy Sample:")
for i, image in enumerate(images):
    print(f"{i+1}th Image:")
    image = load_img(image)
    for _ in range(5):
        features = extractor.forward(image)
        result = generator.sample(features)
        print(result)

print("Beam Search Sample:")
for i, image in enumerate(images):
    print(f"{i+1}th Image:")
    image = load_img(image)
    for _ in range(5):
        features = extractor.forward(image)
        result = generator.beam_sample(features, 10)
        print(result)
