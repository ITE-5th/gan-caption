import pretrainedmodels
import torch
from pretrainedmodels import utils

from file_path_manager import FilePathManager


class Vgg16Extractor:

    def __init__(self, use_gpu: bool = True, transform: bool = True):
        super().__init__()

        self.cnn = pretrainedmodels.vgg16()
        self.tf_image = utils.TransformImage(self.cnn)
        self.transform = transform
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.cnn = self.cnn.cuda()
        self.cnn.eval()

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, image):
        if self.transform:
            image = self.tf_image(image)

        if len(image.size()) == 3:
            image = image.unsqueeze(0)

        if self.use_gpu:
            image = image.cuda()

        temp = self.cnn.features(image)
        return temp

    def __call__(self, image):
        return self.forward(image)


if __name__ == '__main__':
    def extract(path: str):
        image = load_img(path)
        image = extractor.forward(image)
        new_location = path.split(".")[0] + ".pth"
        torch.save(image, new_location)


    extractor = Vgg16Extractor()
    load_img = utils.LoadImage()
    image_path = [
        FilePathManager.resolve("test_images/image_1.jpg"),
        FilePathManager.resolve("test_images/image_2.jpg"),
        FilePathManager.resolve("test_images/image_3.jpg"),
        FilePathManager.resolve("test_images/image_4.jpg"),
        FilePathManager.resolve("test_images/image_6.jpg"),
    ]

    for path in image_path:
        extract(path)
