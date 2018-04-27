import pickle
import random

import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms

from corpus import Corpus
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor


class ECocoDataset(Dataset):

    def __init__(self, corpus: Corpus, evaluator: bool = True, tranform=None):
        self.corpus = corpus
        self.evaluator = evaluator
        self.k = 3
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=tranform)
        self.length = len(self.captions.ids) * self.k

    def __getitem__(self, index):
        temp = index // self.k
        image, item = self.captions[temp]
        caption = item[index % self.k]
        caption = self.corpus.embed_sentence(caption, one_hot=False)
        if self.evaluator:
            other_index = random.choice([k for k in range(self.length // self.k) if k != temp])
            other_caption = self.captions[other_index]
            other_index = random.choice(range(self.k))
            other_caption = other_caption[1][other_index]
            other_caption = self.corpus.embed_sentence(other_caption, one_hot=False)
            return image, caption, other_caption
        else:
            one_hot = self.corpus.sentence_indices(caption)
            return image, caption, one_hot

    def __len__(self):
        return self.length


if __name__ == '__main__':
    captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                 annFile=FilePathManager.resolve(
                                     f"data/annotations/captions_train2017.json"),
                                 transform=transforms.ToTensor())
    print(f"number of images = {len(captions.coco.imgs)}")
    extractor = Vgg16Extractor(use_gpu=True)
    images = []
    i = 1
    for image, _ in captions:
        print(f"caption = {i}")
        item = extractor(image).cpu().data
        images.append(item)
        i += 1
    with open(FilePathManager.resolve("data/embedded_images.pkl"), "wb") as f:
        pickle.dump(images, f)
