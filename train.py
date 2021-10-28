import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from model import KoBARTAbstractiveSummarization
from data_loader import SummaryDataModule


def main():
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    tokenizer_path = "./model/kobart"
    model_path = "./model/kobart"
    max_seq_len = 512
    valid_size = 0.2
    batch_size = 16
    num_workers = 8
    max_epochs = 50
    lr = 3e-5
    warmup_ratio = 0.1

    data_module = SummaryDataModule(
        train_path=train_path,
        test_path=test_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        valid_size=valid_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = KoBARTAbstractiveSummarization(
        model_path=model_path,
        batch_size=batch_size,
        lr=lr,
        warmup_ratio=warmup_ratio,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )

    ckpt_callback = ModelCheckpoint(
        # dirpath="",
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-{val_loss:.5f}",
        verbose=True,
        save_last=True,
        save_top_k=3,
    )

    trainer = pl.Trainer(max_epochs=max_epochs, gpus=1, callbacks=[ckpt_callback])
    trainer.fit(model, data_module)


if __name__ == "__main__":
    # run train
    main()
