from pickletools import uint8
import re
from socket import TIPC_DEST_DROPPABLE
from face import FaceCube
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle as pkl
from enums import Move
import cubie
from coord import CoordCube
from defs import N_MOVE
import moves as mv

class RubicDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx])
        return self.data[idx]

def R222ShortestAll(transform=None, size=None):
    data = pkl.load(open('R222ShortestAll.pkl', 'rb'))
    if size is not None:
        data = data[:size]
    return RubicDataset(data, transform=transform)

def char2move(char):
    if char == "U1":
        return Move.U1
    elif char == "U2":
        return Move.U2
    elif char == "U3":
        return Move.U3
    elif char == "R1":
        return Move.R1
    elif char == "R2":
        return Move.R2
    elif char == "R3":
        return Move.R3
    elif char == "F1":
        return Move.F1
    elif char == "F2":
        return Move.F2
    elif char == "F3":
        return Move.F3
    else:
        raise ValueError("Invalid move char: {}".format(char))



def NOPLoader(train_rate=0.9, batch_size=32, size=None):
    def collate_fn(batch):
        FACE = 0
        MOVE = 1

        inputs = [face_str2int(item[FACE]) + [0] for item in batch]
        targets = []

        for item in batch:
            moves = []
            for i in range(0, len(item[MOVE]), 2):
                moves.append(int(char2move(item[MOVE][i:i+2]) + 1 + 6))
            targets.append(torch.tensor(moves))
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        return torch.tensor(inputs), targets 

    data = R222ShortestAll(size=size)
    train_dataloader = DataLoader(data[1:int(len(data)*train_rate)],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(data[int(len(data)*train_rate):],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return train_dataloader, test_dataloader

def face_str2int(face_str):
    ret = []
    for c in face_str:
        if c == 'U':
            ret.append(1)
        elif c == 'R':
            ret.append(2)
        elif c == 'F':
            ret.append(3)
        elif c == 'D':
            ret.append(4)
        elif c == 'L':
            ret.append(5)
        elif c == 'B':
            ret.append(6)
        else:
            raise ValueError("Invalid face char: {}".format(c))
    return ret

def StateLoader(train_rate=0.9, batch_size=32, size=None):
    def collate_fn(batch):
        FACE = 0
        MOVE = 1
        inputs = [face_str2int(item[FACE]) for item in batch]
        targets = []
        for item in batch:
            faces = []
            fc = FaceCube()
            s = fc.from_string(item[FACE])
            if s is not True:
                raise ValueError("Error in facelet cube")
            cc = fc.to_cubie_cube()
            s = cc.verify()
            if s != cubie.CUBE_OK:
                raise ValueError("Error in cubie cube")
            co_cube = CoordCube(cc)
            for i in range(0, len(item[MOVE]), 2):
                m = char2move(item[MOVE][i:i+2])
                co_cube.corntwist = mv.corntwist_move[N_MOVE * co_cube.corntwist + m]
                co_cube.cornperm = mv.cornperm_move[N_MOVE * co_cube.cornperm + m]
                cc = cubie.CubieCube()
                cc.set_corners(co_cube.cornperm)
                cc.set_cornertwist(co_cube.corntwist)
                faces.append(face_str2int(cc.to_facelet_cube().to_string()))
            targets.append(torch.tensor(faces))
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        return torch.tensor(inputs), targets

    data = R222ShortestAll(size=size)
    train_dataloader = DataLoader(data[1:int(len(data)*train_rate)],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(data[int(len(data)*train_rate):],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    data = R222ShortestAll()
    print("data size", len(data))
    data = R222ShortestAll(size=100000)
    print("data size", len(data))
    print("example data[0]", data[0])
    train_dataloader = DataLoader(data[:3000000], batch_size=4, shuffle=True)
    for i in train_dataloader:
        print("batch_size", len(i))
        print(i[0])
        break
    print("NOPLoader")
    train_dataloader, test_dataloader = NOPLoader()
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("tgt[0]", i[1][0])
        break

    print("StateLoader")
    train_dataloader, test_dataloader = StateLoader(batch_size=10)
    for i in train_dataloader:
        print("src, tgt", len(i))
        print("src.shape", i[0].shape)
        print("tgt.shape", i[1].shape)
        print("src[0]", i[0][0])
        print("tgt[0]", i[1][0])
        break
