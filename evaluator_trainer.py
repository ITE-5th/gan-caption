import os

import torch

from e_coco_dataset import ECocoDataset
from evaluator import Evaluator
from evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager

if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))

    import os
    import time

    from pretrainedmodels import utils
    from torch import optim
    from torch.autograd import Variable
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader
    from vgg16_extractor import Vgg16Extractor
    from conditional_generator import ConditionalGenerator

    from corpus import Corpus
    from file_path_manager import FilePathManager

    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))

    extractor = Vgg16Extractor(transform=False)
    # extractor.cnn = copy.deepcopy(vgg)
    # extractor.cnn.eval()
    extractor.cnn.cuda()

    lr = 1e-4
    alpha = 1
    beta = 1
    captions_per_image = 2
    max_length = 17

    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"), max_sentence_length=max_length)
    evaluator = Evaluator(corpus).cuda()
    dataset = ECocoDataset(corpus, tranform=utils.TransformImage(extractor.cnn), captions_per_image=captions_per_image)
    criterion = EvaluatorLoss(alpha, beta).cuda()
    optimizer = optim.Adam(evaluator.parameters(), lr=lr, betas=(0.8, 0.999), weight_decay=1e-5)

    batch_size = 24
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    epochs = 10
    print(f"number of batches = {len(dataset) // batch_size}")

    all_losses = []
    generator = ConditionalGenerator(corpus, max_sentence_length=max_length).cuda()
    state_dict = torch.load('./models/generator.pth')
    generator.load_state_dict(state_dict['state_dict'])

    for param in generator.parameters():
        param.requires_grad = False
    generator.eval()

    pretrained = False
    if pretrained:
        state_dict = torch.load('./models/evaluator-c99.pth')
        # optimizer.load_state_dict(state_dict['optimizer'])
        evaluator.load_state_dict(state_dict['state_dict'])
        evaluator.eval()
        evaluator.train(True)

    print("Begin Training")
    epoch_loss = 0
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, captions, other_captions) in enumerate(dataloader):
            #
            # if i % 100 == 0:
            #     print(f"Batch = {i}, Time: {time.time() - start}, Loss: {epoch_loss}")

            images, captions, other_captions = images.cuda(), captions.cuda(), other_captions.cuda()
            images = extractor.forward(Variable(images))
            t = captions

            captions = captions.transpose(0, 1).contiguous().view(-1, max_length, captions.shape[-1])
            other_captions = other_captions.transpose(0, 1).contiguous().view(-1, max_length,
                                                                                  other_captions.shape[-1])

            k = images.shape[0]
            # images = torch.stack([images] * captions_per_image).permute(1, 0, 2).contiguous().view(-1, images.shape[-1])
            images = images.unsqueeze(1).repeat(1, 2, 1).view(-1, images.shape[-1])

            captions = pack_padded_sequence(captions, [max_length] * k * captions_per_image, True)
            other_captions = pack_padded_sequence(other_captions, [max_length] * k * captions_per_image, True)
            optimizer.zero_grad()

            generator_outputs = generator.sample_with_embedding(images)  # .permute(1, 0, 2).contiguous()
            generator_outputs = pack_padded_sequence(generator_outputs, [max_length] * k * captions_per_image, True)
            evaluator_outputs = evaluator(images, captions)
            generator_outputs = evaluator(images, generator_outputs)
            other_outputs = evaluator(images, other_captions)

            loss = criterion(evaluator_outputs, generator_outputs, other_outputs)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        end = time.time()
        all_losses.append(epoch_loss)
        print(f"Epoch: {epoch}, Time: {end - start}, Loss: {all_losses[-1]}")
        start = end

    file_name = f"evaluator-c{epoch}.pth"
    torch.save({"state_dict": evaluator.state_dict(), 'losses': all_losses,
                'alpha': alpha, 'beta': beta, 'lr': lr},
               FilePathManager.resolve(f"models/{file_name}"))

    print(all_losses)
