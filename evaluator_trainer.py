import os
import time
from multiprocessing import cpu_count

from pretrainedmodels import utils
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from conditional_generator import ConditionalGenerator
from corpus import Corpus
from e_coco_dataset import ECocoDataset
from evaluator import Evaluator
from evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager
from vgg16_extractor import Vgg16Extractor

if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))

    extractor = Vgg16Extractor(transform=False)

    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    evaluator = Evaluator(corpus).cuda()
    generator = ConditionalGenerator.load(corpus, training=False).cuda()
    dataset = ECocoDataset(corpus, tranform=utils.TransformImage(extractor.cnn))
    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    criterion = EvaluatorLoss(1, 1).cuda()
    optimizer = optim.Adam(evaluator.parameters(), lr=1e-5, weight_decay=1e-5)
    epochs = 50
    print(f"number of batches = {len(dataset) // batch_size}")
    print("Begin Training")
    for epoch in range(epochs):
        total_loss = 0
        start = time.time()
        for i, (images, captions, other_captions) in enumerate(dataloader, 0):
            # print(f"Batch = {i + 1}")
            images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()

            images = extractor(images)
            captions = pack_padded_sequence(captions, [17] * len(images), True)
            other_captions = pack_padded_sequence(other_captions, [17] * len(images), True)
            optimizer.zero_grad()
            generator_outputs = generator.sample_with_embedding(images)
            evaluator_outputs = evaluator(images, captions)
            generator_outputs = evaluator(images, generator_outputs)
            other_outputs = evaluator(images, other_captions)
            loss = criterion(evaluator_outputs, generator_outputs, other_outputs)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        end = time.time()
        print(f"Epoch = {epoch + 1}, Loss: {total_loss}, Time: {end - start}")
        start = end
        evaluator.save()
