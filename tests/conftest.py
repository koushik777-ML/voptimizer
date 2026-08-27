import torch
from torch import nn


class Block(nn.Module):
    """A leaf-bearing block: two linears with a parameterless activation."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class Stack(nn.Module):
    def __init__(self, layers: int = 3, hidden: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class SharedBlockStack(nn.Module):
    """Invokes one block twice, so a module maps to two plan steps."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.block = Block(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.block(x))


class BranchingStack(nn.Module):
    """Data-dependent control flow: the traced order is not always taken."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.left = nn.Linear(hidden, hidden)
        self.right = nn.Linear(hidden, hidden)
        self.take_left = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) if self.take_left else self.right(x)
