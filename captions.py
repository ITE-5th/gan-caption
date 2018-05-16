import random

import numpy as np
import torch


class Captions:
    def __init__(self, dataset, corpus, captions_per_image):
        super().__init__()
        self.dataset = dataset
        self.corpus = corpus
        self.captions_per_image = captions_per_image

    def get_gt(self, indices):
        result = []
        ids = np.asarray(self.dataset.captions.ids)
        for i in range(indices.shape[0]):
            anns = self.dataset.captions.coco.loadAnns(self.dataset.captions.coco.getAnnIds(imgIds=ids[indices[i]]))
            target = np.asarray([ann['caption'] for ann in anns])

            gt = torch.stack(
                [self.corpus.embed_sentence(target[i], one_hot=False) for i in range(self.captions_per_image)])
            result.append(gt)

        return torch.stack(result)

    def get_caption(self, index, caption_index):
        img_id = self.dataset.captions.ids[index]
        ann_ids = self.dataset.captions.coco.getAnnIds(imgIds=img_id)
        anns = self.dataset.captions.coco.loadAnns(ann_ids)
        return anns[caption_index]['caption']

    def get_others(self, indices):
        others = []
        for i in range(len(indices)):
            s = set(np.arange(self.dataset.length))
            s.remove(indices[i])
            s = list(s)
            for _ in range(self.captions_per_image):
                index = random.choice(s)
                caption_index = random.choice(range(self.captions_per_image))
                caption = self.get_caption(index, caption_index)
                others.append(self.corpus.embed_sentence(caption, one_hot=False))

        return torch.stack(others)

    def get_captions(self, indices, other: bool = True):
        indices = indices.cpu().numpy()
        gt = self.get_gt(indices)
        gt = gt.view(-1, gt.shape[-2], gt.shape[-1])
        if other:
            others = self.get_others(indices)
            return gt, others

        return gt
