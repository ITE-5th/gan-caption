import torchvision.datasets as dset
from torch.utils.data import Dataset

from corpus import Corpus
from file_path_manager import FilePathManager


class TestDataset(Dataset):

    def __init__(self, corpus: Corpus, evaluator: bool = True, transform=None):
        self.corpus = corpus
        self.evaluator = evaluator
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=transform)

    def __getitem__(self, index):
        image, caption = self.captions[index]
        inputs = self.corpus.embed_sentence(caption[0], one_hot=False)[:-1]
        targets = self.corpus.embed_sentence(caption[0], one_hot=True)[1:]
        return image, inputs, targets

    def __len__(self):
        return len(self.captions)

# if __name__ == '__main__':
# path = FilePathManager.resolve("test_data.data")
# captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
#                              annFile=FilePathManager.resolve(
#                                  f"data/annotations/captions_train2017.json"),
#                              transform=transforms.ToTensor())
# captions = [captions[0], captions[1]]
# with open(path, "wb") as f:
#     pickle.dump(captions, f)
