from cube_learn import GPTNumNet, CubeLieNumNet
import torch
import time
import cubie
from data_loader import face_str2int
from enums import Move
from coord import CoordCube
from defs import N_MOVE
import moves as mv
import copy

def move(co_cube, m):
    co_cube.corntwist = mv.corntwist_move[N_MOVE * co_cube.corntwist + m]
    co_cube.cornperm = mv.cornperm_move[N_MOVE * co_cube.cornperm + m]

def evaluate(model, p_num=100):
    model.eval()
    start = time.time()
    solved = 0
    for i in range(p_num):
        # create problem
        cc = cubie.CubieCube()
        cc.randomize()
        co_cube = CoordCube(cc)
        seq = torch.zeros(12, 1, 24, dtype=torch.long)
        seq_i = 0
        seq[seq_i, :] = torch.tensor(face_str2int(cc.to_facelet_cube().to_string()))
        while (co_cube.corntwist != 0 or co_cube.cornperm != 0) and seq_i < 10:
            seq_i += 1
            next = []
            for op in Move:
                tmp = copy.deepcopy(co_cube)
                move(tmp, op)
                cc = cubie.CubieCube()
                cc.set_corners(tmp.cornperm)
                cc.set_cornertwist(tmp.corntwist)
                next.append(face_str2int(cc.to_facelet_cube().to_string()))
            min = 100.
            min_face = None
            for j, face in enumerate(next):
                seq[seq_i, :] = torch.tensor(face)
                len = model(seq)
                # target = len[seq_i - 1, 0].item() - 1
                # distance = abs(target - len[seq_i, 0].item())
                distance = len[seq_i, 0].item()
                if distance < min:
                    min = distance
                    tmp = copy.deepcopy(co_cube)
                    move(tmp, j)
                    min_face = tmp
            # print(co_cube.corntwist, co_cube.cornperm)
            co_cube = min_face
            cc = cubie.CubieCube()
            cc.set_corners(min_face.cornperm)
            cc.set_cornertwist(min_face.corntwist)
            seq[seq_i, :] = torch.tensor(face_str2int(cc.to_facelet_cube().to_string()))
        if co_cube.corntwist == 0 and co_cube.cornperm == 0:
            print("Solved")
            solved += 1
        else:
            print("Not solved")
            print(co_cube.corntwist, co_cube.cornperm)


    end = time.time()
    print("Time: ", (end-start)/p_num)
    print(f"Accuracy: {solved/p_num}")

def main(pth_path, model="lie", lie_mode="base", d_model=128, n_layers=3):
    if model == "GPT":
        model = GPTNumNet(d_model=d_model, n_layers=n_layers)
    else:
        model = CubeLieNumNet(d_model=d_model, n_layers=n_layers, mode=lie_mode)
    result = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(result['model'])

    evaluate(model)
    

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, required=True, help="Path to the model")
    parser.add_argument('--model', type=str, default="lie")
    parser.add_argument('--lie_mode', type=str, default="base")
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    args = parser.parse_args()
    main(args.pth_path, args.model, args.lie_mode, args.d_model, args.n_layers)