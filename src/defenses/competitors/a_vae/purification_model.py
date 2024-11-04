from torch import nn


class AVaeDefenseModel(nn.Module):

    def __init__(self, base_classifier, purifier, kernel_size):
        super().__init__()

        self.base_classifier = base_classifier
        self.purifier = purifier
        self.kernel_size = kernel_size

        self.transform = lambda x: (x * 2) - 1
        self.anti_transform = lambda x: (x + 1) / 2

    def purify(self, x):
        x = nn.functional.avg_pool2d(self.transform(x), self.kernel_size)
        x_cln = self.purifier(x, inference=True)
        x_cln = self.anti_transform(x_cln)
        return x_cln

    def forward(self, x):
        x_cln = self.purify(x)
        preds = self.base_classifier(x_cln)
        return preds
