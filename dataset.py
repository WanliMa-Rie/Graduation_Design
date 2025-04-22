import numpy as np
import pandas as pd
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader, random_split
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence


class tokenizer():
    def __init__(self):
        self.start = '^'
        self.end = '$'
        self.pad = ' '
        self.build_vocab()

    def build_vocab(self):
        chars = []
        # atoms
        chars = chars + ['H', 'B', 'C', 'c', 'N', 'n', 'O', 'o', 'P', 'S', 's', 'F', 'I']
        # replace Si for Q, Cl for R, Br for V
        chars = chars + ['Q', 'R', 'V', 'Y', 'Z', 'G', 'T', 'U']
        # hidrogens: H2 to W, H3 to X
        chars = chars + ['[', ']', '+', 'W', 'X']
        # bounding
        chars = chars + ['-', '=', '#', '.', '/', '@', '\\']
        # branches
        chars = chars + ['(', ')']
        # cycles
        chars = chars + ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.tokenlist = [self.pad,self.start,self.end] + list(chars)

        self.char_to_int = {c:i for i,c in enumerate(self.tokenlist)}
        self.int_to_char = {i:c for c,i in self.char_to_int.items()}

    @property # 像访问属性一样
    def vocab_size(self):
        return len(self.int_to_char)

    def encode(self,smi):
        encoded = []
        smi = smi.replace('Si', 'Q')
        smi = smi.replace('Cl', 'R')
        smi = smi.replace('Br', 'V')
        smi = smi.replace('Pt', 'Y')
        smi = smi.replace('Se', 'Z')
        smi = smi.replace('Li', 'T')
        smi = smi.replace('As', 'U')
        smi = smi.replace('Hg', 'G')
        # hydrogens
        smi = smi.replace('H2', 'W')
        smi = smi.replace('H3', 'X')

        return [self.char_to_int[self.start]] + [self.char_to_int[s] for s in smi] + [self.char_to_int[self.end]]

    def decode(self, ords):
        smi = ''.join([self.int_to_char[o] for o in ords])
        # hydrogens
        smi = smi.replace('W', 'H2')
        smi = smi.replace('X', 'H3')
        # replace proxy atoms for double char atoms symbols
        smi = smi.replace('Q', 'Si')
        smi = smi.replace('R', 'Cl')
        smi = smi.replace('V', 'Br')
        smi = smi.replace('Y', 'Pt')
        smi = smi.replace('Z', 'Se')
        smi = smi.replace('T', 'Li')
        smi = smi.replace('U', 'As')
        smi = smi.replace('G', 'Hg')

        return smi

class Gene_Data(Dataset):
    def __init__(
            self,
            path,
            cell_name,

    ):
        self.path = path
        self.cell_name = cell_name

        # data = pd.read_csv(path + cell_name + '.csv',
        # sep=',',
        # usecols=[1]
        # )
        self.tokenizer = tokenizer()
        self.data = self.load_data()

    def load_data(self):
        smiles_list = []
        with open(self.path + self.cell_name + '.txt', 'r') as f:
            for line in f:
                smiles = line.strip()
                smiles_list.append(smiles)
        
        encoded_data = [self.tokenizer.encode(smi) for smi in smiles_list]
        return np.array(encoded_data, dtype=object)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

class Gene_Dataloader():
    def __init__(self,
                 batch_size,
                 path,
                 cell_name,
                 train_rate,
                 ):
        self.batch_size = batch_size
        self.path = path
        self.cell_name = cell_name
        self.train_rate = train_rate

    def get_dataloader(self):
        data = Gene_Data(self.path, self.cell_name)
        dataset_size = len(data)
        train_size = int(dataset_size * self.train_rate)
        test_size = dataset_size - train_size

        # 创建索引列表并打乱
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        # 分割索引
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # 使用 Subset 创建子数据集
        train_data = torch.utils.data.Subset(data, train_indices)
        test_data = torch.utils.data.Subset(data, test_indices)

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0,
            drop_last=True  # 丢弃最后一个不完整的批次
        )

        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0,
            drop_last=True  # 丢弃最后一个不完整的批次
        )
        return train_loader, test_loader

    def collate_fn(self, batch):
        # batch 是一个列表，包含多个 SMILES 序列 (每个序列都是一个 tensor)
        # 使用 pad_sequence 将它们填充到相同的长度
        smi_tensors = [torch.tensor(smi) for smi in batch]
        smi_pad = pad_sequence(smi_tensors, batch_first=True, padding_value=0)  # 假设 0 是填充值
        return smi_pad