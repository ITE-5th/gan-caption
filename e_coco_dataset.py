import random

import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset

from corpus import Corpus
from file_path_manager import FilePathManager


class ECocoDataset(Dataset):

    def __init__(self, corpus: Corpus, evaluator: bool = True, tranform=None, captions_per_image=5):
        self.corpus = corpus
        self.evaluator = evaluator
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=tranform)

        self.captions_per_image = captions_per_image
        self.length = len(self.captions)

    def __getitem__(self, index):
        image, caption = self.captions[index]
        captions = torch.stack(
            [self.corpus.embed_sentence(caption[i], one_hot=False) for i in range(self.captions_per_image)])
        others = []

        s = set(range(self.length))
        s.remove(index)
        s = list(s)
        for i in range(self.captions_per_image):
            other_index = random.choice(s)
            other_caption = self.get_captions(other_index)
            other_index = random.choice(range(self.captions_per_image))
            other_caption = other_caption[1][other_index]
            other_caption = self.corpus.embed_sentence(other_caption, one_hot=False)
            others.append(other_caption)

        others = torch.stack(others)
        return image, captions, others

    def get_captions(self, index):
        coco = self.captions.coco
        img_id = self.captions.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        return target

    def __len__(self):
        return self.length
