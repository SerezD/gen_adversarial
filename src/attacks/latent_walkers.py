import torch
import torch.nn as nn


class NonlinearWalker(nn.Module):
    def __init__(self, n_chunks: int, chunk_size: list, init_weights: bool = False):

        super().__init__()

        self.n_chunks = n_chunks
        self.chunk_size = chunk_size

        self.walkers = nn.ModuleList()

        for i in range(n_chunks):
            self.walkers.add_module(f'walker_{i}',
                                    nn.Sequential(
                                        nn.Linear(chunk_size[i], chunk_size[i]),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(chunk_size[i], chunk_size[i])))

        if init_weights:
            for m in self.walkers.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, 0., 0.001)
                    torch.nn.init.uniform_(m.bias, -0.0001, 0.0001)

    def forward(self, chunk: torch.Tensor, index: int):
        """
        :param chunk: (B, CHUNKS_DIM)
        :param index: index chunk, from 0 to n_chunks
        :return chunk + noise(chunk) (B, CHUNKS_DIM)
        """

        b,  _ = chunk.shape
        noise = self.walkers.get_submodule(f'walker_{index}')(chunk)
        return chunk + noise
