import os
import numpy as np
from typing import Tuple, List, Sequence
import torch
import torch.utils.data as data
from io import BytesIO
from Bio.SeqIO.FastaIO import SimpleFastaParser as FastaParser
np.random.seed(5)

class Alphabet():
    # prepend_bos:bool
    # append_eos:bool
    # padding_idx:padding的index
    # cls_idx:bos的index
    # eos_idx:eos的index
    def __init__(self,padding_idx, cls_idx, eos_idx, prepend_bos=True, append_eos=True):
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

def load_dataset(batch_labels, seq_str_list):
    seqs = [list(seq) for seq in seq_str_list]

    labels = batch_labels

    return labels, seqs

def parse_fasta(filename, datasource=None):
    names = []
    seqs = []
    def iter_fasta(it):
        for name, seq in it:
            if datasource == 'swissprot':
                name = name.split('|')[1]
            elif datasource in ['rcsb', 'norm']:
                name = name.split()[0]
            yield name, seq
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for name, seq in iter_fasta(FastaParser(f)):
                yield name, seq
    with open(filename, 'r', encoding='utf-8') as f:
        for name, seq in iter_fasta(FastaParser(f)):
            names.append(name)
            seqs.append(seq)
    return names, seqs
  
def read_fasta(filename):
    #提取fatsa_path文件夹下protein_id的序列
    #str->str
    #dir = os.path.join(fasta_path, protein_id+'.fasta')
    seqs = []
    seq_names = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        seq_name = lines[0].replace(">","")
        seq = "".join(lines[1:]).replace("\n","")
    seqs.append(seq)
    seq_names.append(seq_name)
    return seqs, seq_names

class MyAlignDataset(data.Dataset):
    def __init__(self):
        self.FASTA = 'data/casp13/train/fasta' #fasta目录
        self.NAME = 'data/casp13/train'#name.idx目录
        self.NPZ = 'data/casp13/train/npz' #npz目录
        self.MAP = 'data/casp13/train' #mapping.idx目录
        self.pids = []
        self.cnt = 0
        with open(os.path.join(self.NAME, 'name.idx')) as f:
            self.pids = list(map(lambda x:x.strip().split(), f))
        self.mapping = {}
        if self.MAP is not None:
            with open(os.path.join(self.MAP, 'mapping.idx')) as f:
                for line in filter(lambda x:len(x)>0, map(lambda x:x.strip(), f)):
                    v, k = line.split()
                    self.mapping[k] = v

    def __getitem__(self, idx):
        pids = self.pids[idx]
        idx = np.random.randint(len(pids))
        while True: #处理npz和fasta不存在的情况
            pid = pids[idx]
            pkey = self.mapping[pid] if pid in self.mapping.keys() else pid
            if os.path.exists(f'{self.NPZ}/{pkey}.npz') and os.path.exists(f'{self.FASTA}/{pkey}.fasta'):
                break
            else:
                idx = (idx + 1) % len(pids)

        seq_feat = self.get_seq_features(pkey)
        coord_dict = self.get_structure_label_npz(pkey, seq_feat)
        return pid, seq_feat, coord_dict

    def __len__(self):
        return len(self.pids)

    def get_seq_features(self, pids):
        if isinstance(pids, str):
            input_seqs, input_decs = read_fasta(f'{self.FASTA}/{pids}.fasta')
            if len(input_seqs) != 1:
                print(input_seqs)
                raise ValueError(f"More than one input sequence found in {self.FASTA}/{pids}.fasta")
            input_sequence = input_seqs[0]
            input_description = input_decs[0]
            return input_sequence

    def get_structure_label_npz(self, pids, str_seqs):
        coords_dicts = []
        if os.path.exists(f'{self.NPZ}/{pids}.npz'):
            with open(f'{self.NPZ}/{pids}.npz', 'rb') as f:
                a = f.read()
                structure = np.load(BytesIO(a))
                return dict(coord = torch.from_numpy(structure['coord']),
                            coord_mask = torch.from_numpy(structure['coord_mask'])
                )
        else:
            self.cnt += 1
            print(f"{self.cnt}:{pids} donnot exist npz")

class AlignDataset(data.Dataset):
    def __init__(self, pids, str_seqs, coord_dicts):
        self.pids = pids
        self.str_seqs = str_seqs
        self.coord_dicts = coord_dicts
    
    def __getitem__(self, index):
        return self.pids[index], self.str_seqs[index], self.coord_dicts[index]

    def __len__(self):
        return len(self.pids)

def getDataset(idx1=20, idx2=40):
    data = MyAlignDataset()
    pids = []
    str_seqs = []
    coord_dicts = []
    for i in range(idx1):
        pid, str_seq, coord_dict = data[i]
        pids.append(pid)
        str_seqs.append(str_seq)
        coord_dict['coord'] = coord_dict['coord'][:, 1, :]
        coord_dict['coord_mask'] = coord_dict['coord_mask'][:,1]
        coord_dicts.append(coord_dict)
        assert len(str_seq) == len(coord_dict['coord'])
    train_dataset = AlignDataset(pids, str_seqs, coord_dicts)

    pids = []
    str_seqs = []
    coord_dicts = []
    for i in range(idx1, idx2):
        if i == 29: #卡住不动， 需要debug
            continue
        pid, str_seq, coord_dict = data[i]
        pids.append(pid)
        str_seqs.append(str_seq)
        coord_dict['coord'] = coord_dict['coord'][:, 1, :]
        coord_dict['coord_mask'] = coord_dict['coord_mask'][:,1]
        coord_dicts.append(coord_dict)
        assert len(str_seq) == len(coord_dict['coord'])
    val_dataset = AlignDataset(pids, str_seqs, coord_dicts)
    return train_dataset, val_dataset
