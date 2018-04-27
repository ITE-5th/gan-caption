import pretrainedmodels
import torchvision.datasets as dset
from pretrainedmodels import utils
from torch.utils.data import DataLoader

from file_path_manager import FilePathManager

extractor = pretrainedmodels.vgg16()

captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                             annFile=FilePathManager.resolve(
                                 f"data/annotations/captions_train2017.json"),
                             # transform=None)
                             transform=utils.TransformImage(extractor))
batch_size = 1
dataloader = DataLoader(captions, batch_size=batch_size, shuffle=True, num_workers=1)
for i, caps in dataloader:
    print(f"size: {len(caps)}, {caps}")
