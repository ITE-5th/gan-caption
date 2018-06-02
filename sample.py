import torch
from pretrainedmodels import utils

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor
state_dict = torch.load(FilePathManager.resolve('models/generator00001111.pth'))

corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
generator = ConditionalGenerator(corpus=corpus, max_sentence_length=16, hidden_size=1024).cuda()
generator.load_state_dict(state_dict['state_dict'])
generator.eval()

extractor = Vgg16Extractor()
load_img = utils.LoadImage()
image_folder = FilePathManager.resolve("test_images/")
# images = glob.glob(image_folder + "*")
image1 = image_folder + "image_1.png"
image2 = image_folder + "image_2.jpg"
image3 = image_folder + "image_3.jpg"
image4 = image_folder + "image_4.jpg"
image5 = image_folder + "image_5.jpg"
image6 = image_folder + "image_6.jpg"
images = [image1, image2, image3, image4, image5, image6]

greedy_samples = []
beam_samples = []
for i, image in enumerate(images):
    image = load_img(image)

    gr_samples = []
    be_samples = []
    features = extractor.forward(image)
    for _ in range(5):
        gr_samples.append(generator.sample(features))
        be_samples.append(generator.beam_sample(features, 10))

    greedy_samples.append(gr_samples)
    beam_samples.append(be_samples)

for i in range(len(images)):
    print(f"{i+1}th Image:")
    print("\n".join(greedy_samples[i]))
    print("\n".join(beam_samples[i]))
