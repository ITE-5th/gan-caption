import pretrainedmodels
from pretrainedmodels import utils
from torch.autograd import Variable

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

        temp = self.cnn.features(Variable(image))
        return temp

    def __call__(self, image):
        return self.forward(image)


if __name__ == '__main__':
    extractor = Vgg16Extractor()
    load_img = utils.LoadImage()
    image_path = FilePathManager.resolve("test_images/image_1.png")

    print(extractor.forward(load_img(image_path)))
