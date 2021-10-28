import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import BartForConditionalGeneration
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from .kobart import get_pytorch_kobart_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class KoBARTAbstractiveSummarization(pl.LightningModule):
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        lr: float = 3e-5,
        warmup_ratio: float = 0.1,
        max_epochs: int = 50,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # BART model for summarization
        self.model = BartForConditionalGeneration.from_pretrained(
            get_pytorch_kobart_model(model_path)
        )

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        outs = self.forward(batch)
        loss = outs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self.forward(batch)
        loss = outs.loss
        self.log("loss", loss, prog_bar=True, on_step=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f"number of workers {num_workers}, data length {data_len}")
        num_train_steps = int(
            data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs
        )
        logging.info(f"num_train_steps : {num_train_steps}")
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f"num_warmup_steps : {num_warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
