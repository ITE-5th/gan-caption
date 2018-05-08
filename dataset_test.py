import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from captions import Captions
from corpus import Corpus
from file_path_manager import FilePathManager


class CocoDataset(Dataset):

    def __init__(self, tranform=None):
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=tranform)

        self.captions_per_image = captions_per_image
        self.length = len(self.captions)
        self.s = set(range(self.length))

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.length


if __name__ == '__main__':
    lr1 = 4e-4
    lr2 = 4e-4
    alpha = 1
    beta = 1
    captions_per_image = 3
    max_length = 17

    epochs = 50
    batch_size = 2
    monte_carlo_count = 16
    # extractor = Vgg16Extractor(transform=False)
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_length)

    # generator = ConditionalGenerator(corpus=corpus).cuda()
    # state_dict = torch.load(FilePathManager.resolve('models/generator.pth'))
    # generator.load_state_dict(state_dict['state_dict'])
    # generator.eval()

    dataset = CocoDataset(tranform=torchvision.transforms.ToTensor())
    # dataset = CocoDataset(tranform=utils.TransformImage(extractor.cnn))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    captions = Captions(dataset, corpus, captions_per_image)

    for indices in dataloader:
        gt, others = captions.get_captions(indices)
        print(f"1: {gt.shape}, 2: {others.shape}")
