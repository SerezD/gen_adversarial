import torch


class EoTWrapper(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, eot_steps: int):
        """
        Simple wrapper implementing Expectation over Transformation
        """
        super(EoTWrapper, self).__init__()

        self.model = model
        self.eot_steps = eot_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 3, h, w)
        """

        x = x.repeat(self.eot_steps, 1, 1, 1)
        preds = self.model(x)
        preds = torch.mean(preds, dim=0, keepdim=True)

        return preds
