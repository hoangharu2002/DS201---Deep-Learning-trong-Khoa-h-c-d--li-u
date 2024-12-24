import torch
from torch.utils.data import Dataset
import json

from vocab import Vocab

def collate_fn(items: list[dict]) -> dict:

    def pad_value(input: torch.Tensor, value: int, max_length: int) -> torch.Tensor:
        delta_len = max_length - input.shape[-1]
        pad_tensor = torch.tensor([value]*delta_len).long()
        input = torch.cat([input, pad_tensor], dim=-1)

        return input

    max_len = max([item["input_ids"].shape[-1] for item in items])
    batch_input_ids = []
    batch_labels = []
    for item in items:
        input_ids = item["input_ids"]
        input_ids = pad_value(input_ids, value=0, max_length=max_len)
        batch_input_ids.append(input_ids.unsqueeze(0))

        labels = item["labels"]
        labels = pad_value(labels, value=-1, max_length=max_len)
        batch_labels.append(labels.unsqueeze(0))

    return {
        "input_ids": torch.cat(batch_input_ids),
        "labels": torch.cat(batch_labels)
    }

class PhoNER(Dataset):
    def __init__(self, path: str, vocab: PhoNER_Vocab):
        super().__init__()

        _data = json.load(open(path, encoding='utf-8'))
        self._data = list(_data.values())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
      
        item = self._data[index]
        
        sentence = item["review"]
        
        label = item["label"]

        encoded_sentence = self._vocab.encode_sentence(sentence)
        encoded_label = self._vocab.encode_label(label)
      
        return {
            "input_ids": encoded_sentence,
            "label": encoded_label
        }
