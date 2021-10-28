import pandas as pd
import transformers
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import SummaryDataset
from model.kobart import get_kobart_tokenizer


class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        valid_size: float = 0.2,
        batch_size: int = 8,
        num_workers=8,
    ):
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # load data & tokenizer
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        tokenizer = get_kobart_tokenizer(self.tokenizer_path)

        # split train/valid
        train, valid = train_test_split(train, test_size=self.valid_size, shuffle=True)

        # train/valid/test Dataset
        self.trainset = SummaryDataset(train, tokenizer, self.max_seq_len, phase="train")
        self.validset = SummaryDataset(valid, tokenizer, self.max_seq_len, phase="valid")
        self.testset = SummaryDataset(test, tokenizer, self.max_seq_len, phase="test")

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    tokenizer_path = "./kobart"
    max_seq_len = 512
    valid_size = 0.2
    batch_size = 2
    num_workers = 4

    data_module = SummaryDataModule(
        train_path=train_path,
        test_path=test_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        valid_size=valid_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # for batch in test_loader:
    #     batch = batch
    #     break
