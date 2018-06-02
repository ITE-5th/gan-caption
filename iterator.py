import torch


class Iterator:
    """
    Useless Iterator, DON'T USE IT
    """

    def __init__(self, dataloader, count):
        super().__init__()

        self.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.count = count

    def __iter__(self):
        return self

    def reset(self):
        self.iter = iter(self.dataloader)

    def __next__(self):
        batch = next(self.iter, None)
        # End of the epoch
        if batch is None:
            raise StopIteration

        images, inputs, targets = batch[0], batch[1], batch[2]
        #
        for i in range(1, self.count):
            result = next(self.iter, None)
            if result is None:
                break
            images2, inputs2, targets2 = batch[0], batch[1], batch[2]

            images = torch.cat([images, images2], dim=0)
            inputs = torch.cat([inputs, inputs2], dim=0)
            targets = torch.cat([targets, targets2], dim=0)

        return images, inputs, targets
