import face
import cubie
import coord
from defs import N_TWIST, N_CORNERS
import pruning as pr
from solver import solve
import pickle as pkl

# CoordCube
solutions = []
for i in range(N_TWIST):
    for j in range(N_CORNERS):
        cc = cubie.CubieCube()
        cc.set_corners(j)
        cc.set_cornertwist(i)
        face_let_str = cc.to_facelet_cube().to_string()
        s = solve(face_let_str)
        one_solution = s.split('\n')[0].split('(')
        solution = "".join(one_solution[0].split(" "))
        solutions.append([face_let_str, solution])
    print(".", end="", flush=True)
with open('solutions.pkl', "wb") as f:
    f.write(pkl.dumps(solutions))
