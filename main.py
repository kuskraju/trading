import asyncio

from functions import trade
from hidden_module.train import train, train_latest


if __name__ == "__main__":
    # asyncio.run(trade())
    train()
    # train_latest()
