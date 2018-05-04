import pickle
import torch

import torchvision.datasets as dset
from joblib import cpu_count
from pretrainedmodels import utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from corpus import Corpus
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor


class GCocoDataset(Dataset):

    def __init__(self, corpus: Corpus, transform=None):
        self.corpus = corpus
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=transform)

    def __getitem__(self, index):
        image, caption = self.captions[index]
        # self.captions.coco.anns[]
        inputs = torch.stack([self.corpus.embed_sentence(caption[i], one_hot=False) for i in range(len(caption))])
        targets = torch.stack([self.corpus.sentence_indices(caption[i]) for i in range(len(caption))])
        return image, inputs, targets

    def __len__(self):
        return len(self.captions)


if __name__ == '__main__':
    extractor = Vgg16Extractor(transform=False)
    captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                 annFile=FilePathManager.resolve(
                                     f"data/annotations/captions_train2017.json"),
                                 transform=utils.TransformImage(extractor.cnn))
    batch_size = 3
    dataloader = DataLoader(captions, batch_size=batch_size, shuffle=True, num_workers=cpu_count())

    print(f"number of images = {len(captions.coco.imgs)}")
    images = []
    i = 1
    for image, _ in dataloader:
        print(f"batch = {i}")
        item = extractor.forward(image).cpu().data
        images.append(item)
        i += 1
    with open(FilePathManager.resolve("data/embedded_images.pkl"), "wb") as f:
        pickle.dump(images, f)
    # corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    # one_hot = []
    # i = 1
    # for _, capts in captions:
    #     print(f"caption = {i}")
    #     for capt in capts:
    #         one_hot.append(corpus.embed_sentence(capt, one_hot=True))
    #     i += 1
    # with open(FilePathManager.resolve("data/one_hot_sentences.pkl"), "wb") as f:
    #     pickle.dump(one_hot, f)
    # i = 1
    # embedded_sentences = []
    # for _, capts in captions:
    #     print(f"caption = {i}")
    #     for capt in capts:
    #         embedded_sentences.append(corpus.embed_sentence(capt, one_hot=False))
    #     i += 1
    # with open(FilePathManager.resolve("data/embedded_sentences.pkl"), "wb") as f:
    #     pickle.dump(embedded_sentences, f)
