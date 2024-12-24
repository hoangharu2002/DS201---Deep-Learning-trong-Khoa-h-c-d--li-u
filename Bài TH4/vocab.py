import torch
import re
import json
from typing import List

def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    sentence = re.sub(r"%", " % ", sentence)

    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()

    return tokens

class Vocab:
    def __init__(self, train_path: str, dev_path: str, test_path: str):
        self.initialize_special_tokens()
        self.make_vocab(train_path, dev_path, test_path)

    def initialize_special_tokens(self) -> None:
        self.cls_token = "<cls"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.specials = [self.pad_token, self.cls_token, self.unk_token]

        self.pad_idx = 0
        self.cls_idx = 1
        self.unk_idx = 2

    def make_vocab(self, train_path: str, dev_path: str, test_path: str):
        json_dirs = [train_path, dev_path, test_path]
        words = set()
        labels = set()
        for json_dir in json_dirs:
            data = json.load(open(json_dir,  encoding='utf-8'))
            for key in data:
                tokens = preprocess_sentence(data[key]["review"])
                words.update(tokens)
                label = data[key]["label"]
                labels.add(label)

        itos = self.specials + list(words)

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        
        labels = list(labels)
        self.i2l = {i: label for i, label in enumerate(labels)}
        self.l2i = {label: i for i, label in enumerate(labels)}

    @property
    def total_tokens(self) -> int:
        return len(self.itos)
    
    @property
    def total_labels(self) -> int:
        return len(self.i2l)
    
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = sentence.split()
        vec = [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence]
        vec = torch.Tensor(vec).long()

        return vec

    def encode_label(self, labels: list) -> torch.Tensor:
        
        labels = [self.l2i[label] for label in labels]
 
        return torch.Tensor(labels).long()
    
    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        label_vecs: (bs)
        """
        result = []
        labels = label_vecs.tolist()
        for label in labels:
            result.append(self.i2l[label])
        
        return result
    
    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)
