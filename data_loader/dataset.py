import numpy as np
import pandas as pd
import transformers

from torch.utils.data import Dataset
from model.kobart import get_kobart_tokenizer


class SummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerFast,
        max_seq_len: int = 512,
        phase: str = "train",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.phase = phase
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        if self.phase in ["train", "valid"]:
            document, summary = item["total"], item["summary"]
            encoder_input_id, encoder_attention_mask = self.encode_and_pad(document)
            decoder_input_id, decoder_attention_mask = self.encode_and_pad(summary)

            output_id = self.tokenizer.encode(summary)
            output_id += [self.tokenizer.eos_token_id]
            if len(output_id) < self.max_seq_len:
                pad_len = self.max_seq_len - len(output_id)
                output_id += [self.ignore_index] * pad_len
            else:
                output_id = output_id[: self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            return {
                "input_ids": np.array(encoder_input_id, dtype=np.int_),
                "attention_mask": np.array(encoder_attention_mask, dtype=np.float32),
                "decoder_input_ids": np.array(decoder_input_id, dtype=np.int_),
                "decoder_attention_mask": np.array(decoder_attention_mask, dtype=np.float32),
                "labels": np.array(output_id, dtype=np.int_),
                "summary": summary,
            }
        else:
            document = item["total"]
            encoder_input_id, encoder_attention_mask = self.encode_and_pad(document)
            return {
                "input_ids": np.array(encoder_input_id, dtype=np.int_),
                "attention_mask": np.array(encoder_attention_mask, dtype=np.float32),
            }

    def encode_and_pad(self, text: str):
        # token_to_id
        # encoder_input_id = self.tokenizer.encode(document)
        tokens = (
            [self.tokenizer.bos_token] + self.tokenizer.tokenize(text) + [self.tokenizer.eos_token]
        )
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)

        # padding
        if len(input_id) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_id)
            input_id += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_id = input_id[: self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask[: self.max_seq_len]
        return input_id, attention_mask
