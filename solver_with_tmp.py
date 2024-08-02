import face
import cubie
import coord
import enums as en
import moves as mv
import pruning as pr
from defs import N_TWIST, N_CORNERS
import pickle as pkl

solutions = []


def search(cornperm, corntwist, togo):
    global solutions
    stack = [(cornperm, corntwist, [], togo)]
    states = []

    while stack:
        current_cornperm, current_corntwist, sofar, remaining_moves = stack.pop()
        if remaining_moves == 0:
            if len(solutions) == 0 or (len(solutions[-1]['history']) == len(sofar)):
                solutions.append({
                    'state': (current_cornperm, current_corntwist),
                    'done': True,
                    'stack': stack[:],  # 現在のスタック状態を記録
                    'history': sofar[:]
                })
            break
        else:
            for m in en.Move:
                # successive moves on same face
                if len(sofar) > 0 and sofar[-1] // 3 == m // 3:
                    continue

                cornperm_new = mv.cornperm_move[9 * current_cornperm + m]
                corntwist_new = mv.corntwist_move[9 * current_corntwist + m]

                if pr.corner_depth[N_TWIST * cornperm_new + corntwist_new] >= remaining_moves:
                    # impossible to reach solved cube in togo - 1 moves
                    continue

                new_sofar = sofar + [m]
                if cornperm_new == 0 and corntwist_new == 0:
                    done = True
                else:
                    done = False

                states.append({
                    'state': (cornperm_new, corntwist_new),
                    'done': done,
                    'stack': stack[:] + [
                        (cornperm_new, corntwist_new,
                         new_sofar, remaining_moves - 1)],
                    'history': new_sofar[:]
                })
                stack.append((cornperm_new, corntwist_new,
                              new_sofar, remaining_moves - 1))
    return states


def solve(cubestring):
    """Solves a 2x2x2 cube defined by its cube definition string.
    :param cubestring: The format of the string is
                       given in the Facelet class defined in the file enums.py
    :return A list of all optimal solving maneuvers in the desired format
    """
    global solutions
    fc = face.FaceCube()
    s = fc.from_string(cubestring)
    if s is not True:
        return s  # Error in facelet cube
    cc = fc.to_cubie_cube()
    s = cc.verify()
    if s != cubie.CUBE_OK:
        return s  # Error in cubie cube

    solutions = []
    co_cube = coord.CoordCube(cc)
    togo = pr.corner_depth[N_TWIST * co_cube.cornperm + co_cube.corntwist]
    states = search(co_cube.cornperm, co_cube.corntwist, togo)
    """
    print("Solutions")
    for solution in solutions:
        print(solution["state"], solution["done"])
        print(solution["stack"])
        print(solution["history"])

    print("States")
    for state in states:
        print(state["state"], state["done"])
        print(len(state["stack"]), state["stack"])
        print(state["history"])
    """
    return states


def make_dfs_data_set():
    solutions = []
    for i in range(N_TWIST):
        for j in range(N_CORNERS):
            cc = cubie.CubieCube()
            cc.set_corners(j)
            cc.set_cornertwist(i)
            face_let_str = cc.to_facelet_cube().to_string()
            states = solve(face_let_str)
            solutions.append(states)
        print(".", end="", flush=True)
        if i % 10 == 9:
            with open(f'dfs_solutions_{i}.pkl', 'wb') as f:
                f.write(pkl.dumps(solutions))
            solutions = []


if __name__ == '__main__':
    cubestring = 'FFBLBRDLDUBRRFDDLRLUUUFB'
    solve(cubestring)
    make_dfs_data_set()
