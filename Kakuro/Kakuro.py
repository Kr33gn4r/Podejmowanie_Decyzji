import numpy as np
import re
import subprocess
import json
from timeit import default_timer as timer
from random import uniform, randint

class Kakuro:
    def __init__(self):
        pass

    def Creator(self, nrows: int, ncols: int):
        flag = True
        while flag:
            flag, e = self._grider(nrows, ncols)
        self.E = e
        (self.a, self.b) = np.shape(self.E)

        flag = True
        while flag:
            flag, s = self._filler(e.copy(), nrows, ncols)
        self.S = s

        t = self._triangulizer(s, nrows, ncols)
        self.T = t

        l, ll = self._lengther(e, t)
        self.L, self.n = l, ll

    def CplexSolver(self, a, b, n, E, T, L):
        path_dat = "/home/rskay/opl/Kakuro/Kakuro.dat"
        opl = r"/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/oplrun"
        cmd = [opl, r"-p", r"/home/rskay/opl/Kakuro", r"Configuration2"]

        with open(path_dat, "w") as f:
            f.writelines(f"a = {a};\n"
                         f"b = {b};\n"
                         f"n = {n};\n"
                         f"E = {repr(E)[6:-1]};\n"
                         f"T = {repr(T)[6:-1]};\n"
                         f"L = {repr(L)[6:-1]};")

        output = str(subprocess.run(cmd, capture_output=True))
        self.timeCPlex = float(str(re.findall(r'Total.*?sec', output)[0])[28:-4])
        result = str(re.findall(r'OBJECTIVE:.*?<', output)[0])[16:-6].replace(" ", "").split(r"\n")
        self.resultCPlex = np.array([[int(x) for x in s[1:-1]] for s in result])

    def PrepSA(self, E: np.ndarray, T: np.ndarray, L: np.ndarray, max_iter=1000, start_T=1000, alpha=0.999):
        # No end temperature, because it can only quit due to max_iter or getting a solved puzzle

        # Initialize combinations
        a, b = np.shape(E)
        HS = [[[1, 2, 3, 4, 5, 6, 7, 8, 9] if E[i, j] == 0 else [0] for j in range(b)] for i in range(a)]

        # Initialize intersections
        sumcombinations = {3:
                               {2: [1, 2]},
                           4:
                               {2: [1, 3]},
                           5:
                               {2: [1, 2, 3, 4]},
                           6:
                               {2: [1, 2, 4, 5],
                                3: [1, 2, 3]},
                           7:
                               {2: [1, 2, 3, 4, 5, 6],
                                3: [1, 2, 4]},
                           8:
                               {2: [1, 2, 3, 5, 6, 7],
                                3: [1, 2, 3, 4, 5]},
                           9:
                               {2: [1, 2, 3, 4, 5, 6, 7, 8],
                                3: [1, 2, 3, 4, 5, 6]},
                           10:
                               {2: [1, 2, 3, 4, 6, 7, 8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7],
                                4: [1, 2, 3, 4]},
                           11:
                               {2: [2, 3, 4, 5, 6, 7, 8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8],
                                4: [1, 2, 3, 5]},
                           12:
                               {2: [3, 4, 5, 7, 8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6]},
                           13:
                               {2: [4, 5, 6, 7, 8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7]},
                           14:
                               {2: [5, 6, 8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8]},
                           15:
                               {2: [6, 7, 8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5]},
                           16:
                               {2: [7, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 6]},
                           17:
                               {2: [8, 9],
                                3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7]},
                           18:
                               {3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8]},
                           19:
                               {3: [2, 3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           20:
                               {3: [3, 4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           21: {3: [4, 5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6]},
                           22:
                               {3: [5, 6, 7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 7]},
                           23:
                               {3: [6, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8]},
                           24:
                               {3: [7, 8, 9],
                                4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           25:
                               {4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           26:
                               {4: [2, 3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           27:
                               {4: [3, 4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           28:
                               {4: [4, 5, 6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7]},
                           29:
                               {4: [5, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 8]},
                           30:
                               {4: [6, 7, 8, 9],
                                5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           31:
                               {5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           32:
                               {5: [2, 3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           33:
                               {5: [3, 4, 5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           34:
                               {5: [4, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           35:
                               {5: [5, 6, 7, 8, 9],
                                6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                           36:
                               {6: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 3, 4, 5, 6, 7, 8]},
                           37:
                               {6: [2, 3, 4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 3, 4, 5, 6, 7, 9]},
                           38:
                               {6: [3, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 3, 4, 5, 6, 8, 9]},
                           39:
                               {6: [4, 5, 6, 7, 8, 9],
                                7: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 3, 4, 5, 7, 8, 9]},
                           40:
                               {7: [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 3, 4, 6, 7, 8, 9]},
                           41:
                               {7: [2, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 3, 5, 6, 7, 8, 9]},
                           42:
                               {7: [3, 4, 5, 6, 7, 8, 9],
                                8: [1, 2, 4, 5, 6, 7, 8, 9]},
                           43:
                               {8: [1, 3, 4, 5, 6, 7, 8, 9]},
                           44:
                               {8: [2, 3, 4, 5, 6, 7, 8, 9]},
                           45:
                               {9: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
                           }
        for i, tr in enumerate(T):
            if tr[2] == 0:
                for j in range(tr[1], tr[1] + L[i]):
                    HS[tr[0] - 1][j] = [x for x in HS[tr[0] - 1][j] if x in sumcombinations[tr[3]][L[i]]]
            else:
                for j in range(tr[0], tr[0] + L[i]):
                    HS[j][tr[1] - 1] = [x for x in HS[j][tr[1] - 1] if x in sumcombinations[tr[3]][L[i]]]

        # Identify if a number appears only once in sum
        # If it does, check if there is only one option for a sum
        # If it is, append that number to that block, and remove the number from row and column
        flag = True
        while flag:
            flag = False
            for i, tr in enumerate(T):
                for j in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    if tr[2] == 0:
                        x = [1 if j in HS[tr[0] - 1][k] else 0 for k in range(tr[1], tr[1] + L[i])]
                        if sum(x) == 1:
                            if len(sumcombinations[tr[3]][L[i]]) == L[i]:
                                if HS[tr[0] - 1][tr[1] + x.index(1)] == [j]:
                                    break
                                else:
                                    flag = True
                                    HS[tr[0] - 1][tr[1] + x.index(1)] = [j]
                                    k = tr[0] - 2
                                    while HS[k][tr[1]] != [0]:
                                        try:
                                            HS[k][tr[1]].remove(j)
                                        except ValueError:
                                            pass
                                        k -= 1
                                    k = tr[0]
                                    while k != np.shape(E)[1] and HS[k][tr[1]] != [0]:
                                        try:
                                            HS[k][tr[1]].remove(j)
                                        except ValueError:
                                            pass
                                        k += 1
                    else:
                        x = [1 if j in HS[k][tr[1] - 1] else 0 for k in range(tr[0], tr[0] + L[i])]
                        if sum(x) == 1:
                            if len(sumcombinations[tr[3]][L[i]]) == L[i]:
                                if HS[tr[0] + x.index(1)][tr[1] - 1] == [j]:
                                    break
                                else:
                                    flag = True
                                    HS[tr[0] + x.index(1)][tr[1] - 1] = [j]
                                    k = tr[1] - 2
                                    while HS[tr[0]][k] != [0]:
                                        try:
                                            HS[tr[0]][k].remove(j)
                                        except ValueError:
                                            pass
                                        k -= 1
                                    k = tr[1]
                                    while k != np.shape(E)[0] and HS[tr[0]][k] != [0]:
                                        try:
                                            HS[tr[0]][k].remove(j)
                                        except ValueError:
                                            pass
                                        k += 1

        # Using modified SA algorithm generate 'correct' row sums (value and num uniq)
        # and then check using a function :
        # sum per every column sum(|real sum - created sum| + uniqueness error)
        curr_W = np.array([[0 for _ in range(b)] for __ in range(a)])
        for i, tr in enumerate(T):
            if tr[2] == 0:
                curr_W[tr[0]-1, tr[1]:tr[1]+L[i]] = \
                self._rower([HS[tr[0]-1][h] for h in range(tr[1], tr[1] + L[i])],
                            tr[3])
        curr_ERR = 0
        for i, tr in enumerate(T):
            if tr[2] == 1:
                curr_ERR += abs(np.sum(curr_W[tr[0] : tr[0]+L[i], tr[1]-1]) - tr[3]) + \
                            5 * (len(curr_W[tr[0] : tr[0]+L[i], tr[1]-1]) - len(np.unique(curr_W[tr[0] : tr[0]+L[i], tr[1]-1])))

        return self.ModifiedSA(T, L, HS, curr_W, curr_ERR, start_T, max_iter, alpha)

    def ModifiedSA(self, T, L, HS, curr_W, curr_ERR, start_T, max_iter, alpha):
        curr_T = start_T
        no_change = 0
        for iter in range(max_iter):
            while True:
                adj = randint(0, len(T) - 1)
                if T[adj][2] == 0: break
            adj_W = np.copy(curr_W)
            adj_W[T[adj][0] - 1, T[adj][1]:T[adj][1] + L[adj]] = \
                self._rower([HS[T[adj][0] - 1][h] for h in range(T[adj][1], T[adj][1] + L[adj])],
                            T[adj][3])
            adj_ERR = 0
            for i, tr in enumerate(T):
                if tr[2] == 1:
                    adj_ERR += abs(np.sum(adj_W[tr[0]: tr[0] + L[i], tr[1] - 1]) - tr[3]) + \
                               5 * (len(adj_W[tr[0]: tr[0] + L[i], tr[1] - 1]) - len(
                        np.unique(adj_W[tr[0]: tr[0] + L[i], tr[1] - 1])))
            if adj_ERR == 0:
                self.resultSA = adj_W
                return True, True, True
            elif adj_ERR < curr_ERR:
                curr_ERR = adj_ERR
                curr_W = adj_W
                no_change = 0
            else:
                p_acc = np.exp(np.float128((adj_ERR - curr_ERR) / curr_T))
                if p_acc > uniform(0, 1):
                    curr_ERR = adj_ERR
                    curr_W = adj_W
                    no_change = 0
                else: no_change += 1
            if no_change >= 10:
                return curr_W, curr_ERR, HS
            curr_T *= alpha
        return curr_W, curr_ERR, HS

    def _rower(self, c, s):
        while True:
            num_chosen, rest_sum = [], s
            for n in c:
                try:
                    r = np.random.choice([x for x in n if (x not in num_chosen) and (x <= rest_sum)])
                    num_chosen.append(r)
                    rest_sum -= r
                except Exception: break
            if rest_sum == 0 and len(c) == len(num_chosen): return np.array(num_chosen)

    def _grider(self, nrows: int, ncols: int):
        e = np.ones((nrows, ncols), dtype=int)
        e[1, 1], e[1, 2], e[2, 1] = 0, 0, 0
        onechance = 0.4
        for i in range(1, nrows):
            for j in range(1, ncols):
                if e[i, j] == 0:
                    continue
                else:
                    srow, scol = 0, 0
                    for k in range(j):
                        if e[i, k] == 0:
                            srow += 1
                        else:
                            srow = 0
                    for k in range(i):
                        if e[k, j] == 0:
                            scol += 1
                        else:
                            scol = 0

                    if srow == 9 and scol == 1:
                        e[i, j] = 1
                        e[i - 1, j] = 1
                    elif srow == 1 and scol == 9:
                        e[i, j] = 1
                        e[i, j - 1] = 1
                    elif srow == 1 or scol == 1:
                        e[i, j] = 0
                    elif srow == 9 or scol == 9:
                        e[i, j] = 1
                    elif i == nrows - 1 and scol == 0:
                        e[i, j] = 1
                    elif j == ncols - 1 and srow == 0:
                        e[i, j] = 1
                    else:
                        e[i, j] = 1 if onechance > uniform(0, 1) else 0

        for i in range(1, nrows):
            s = 0
            for j in range(1, ncols):
                if s > 9: return True, None
                if e[i, j] == 0:
                    s += 1
                else:
                    if s == 1:
                        return True, None
                    else:
                        s = 0
                if j == ncols - 1 and s == 1: return True, None

        for j in range(1, ncols):
            s = 0
            for i in range(1, nrows):
                if s > 9: return True, None
                if e[i, j] == 0:
                    s += 1
                else:
                    if s == 1:
                        return True, None
                    else:
                        s = 0
                if i == nrows - 1 and s == 1: return True, None
        return False, e

    def _filler(self, e: np.ndarray, nrows: int, ncols: int):
        for i in range(nrows):
            for j in range(ncols):
                if e[i, j] == 1:
                    e[i, j] = 0
                else:
                    e[i, j] = 1

        for i in range(1, nrows):
            for j in range(1, ncols):
                if e[i, j] == 0: continue
                avail = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                taken = []
                for k in range(j):
                    if e[i, k] >= 1:
                        taken.append(e[i, k])
                    else:
                        taken = []
                avail = [x for x in avail if x not in taken]
                taken = []
                for k in range(i):
                    if e[k, j] >= 1:
                        taken.append(e[k, j])
                    else:
                        taken = []
                avail = [x for x in avail if x not in taken]
                if avail:
                    e[i, j] = np.random.choice(avail, 1)
                else:
                    return True, None
        return False, e

    def _triangulizer(self, s: np.ndarray, nrows: int, ncols: int):
        t = []
        for i in range(nrows):
            tempsum, tempcords = 0, (0, 0)
            for j in range(ncols):
                if s[i, j] == 0:
                    if tempsum > 0:
                        t.append([tempcords[0] + 1, tempcords[1] + 1, 0, tempsum])
                        tempsum = 0
                    tempcords = (i, j)
                else:
                    tempsum += s[i, j]
            if tempsum > 0:
                t.append([tempcords[0] + 1, tempcords[1] + 1, 0, tempsum])

        for j in range(ncols):
            tempsum, tempcords = 0, (0, 0)
            for i in range(nrows):
                if s[i, j] == 0:
                    if tempsum > 0:
                        t.append([tempcords[0] + 1, tempcords[1] + 1, 1, tempsum])
                        tempsum = 0
                    tempcords = (i, j)
                else:
                    tempsum += s[i, j]
            if tempsum > 0:
                t.append([tempcords[0] + 1, tempcords[1] + 1, 1, tempsum])
        return np.array(t)

    def _lengther(self, e: np.ndarray, t: np.ndarray):
        a, b = e.shape
        n, _ = t.shape

        L = [0 for _ in range(n)]
        for i, triangle in enumerate(t):
            tau = np.zeros(c := (max(a, b)), dtype=int)
            if triangle[2] == 0:  # row
                for j in range(triangle[1], b):
                    tau[j] = 1 if sum(e[triangle[0] - 1, triangle[1]:j + 1]) == 0 else 0
            else:
                for j in range(triangle[0], a):
                    tau[j] = 1 if sum(e[triangle[0]:j + 1, triangle[1] - 1]) == 0 else 0
            L[i] = sum(tau)
        return np.array(L), len(L)

    def _checker(self, s: np.ndarray, t: np.ndarray, l: np.ndarray):
        err = []
        for i, tr in enumerate(t):
            if tr[2] == 0:
                if sum(set(s[tr[0] - 1, tr[1]: tr[1] + l[i]])) != tr[3]:
                    err.append(list(tr))
            else:
                if sum(set(s[tr[0]: tr[0] + l[i], tr[1] - 1])) != tr[3]:
                    err.append(list(tr))
        return err

k = Kakuro()

size = (7, 7)
for i in range(1, 11):
    print(size, i)
    name = f"/home/rskay/PycharmProjects/Projekty/Sem4/Podejmowanie Decyzji/Kakuro/Results/{size[0]}x{size[1]}_{i}.json"
    k.Creator(*size)

    startCPLEX = timer()
    k.CplexSolver(k.a, k.b, k.n, k.E, k.T, k.L)
    stopCPLEX = timer() - startCPLEX

    SA_time_arr = []
    SA_result_arr = []
    new = True
    W, ERR, HS = True, True, True
    for j in range(25):
        startSA = timer()
        if new == True:
            W, ERR, HS = k.PrepSA(k.E, k.T, k.L, max_iter=100000, start_T=100000, alpha=0.99)
            if W is not True:
                new = False
            else:
                SA_result_arr.append(k.resultSA)
        else:
            W, ERR, HS = k.ModifiedSA(k.T, k.L, HS, W, ERR, max_iter=100000, start_T=100000, alpha=0.99)
            if W is True:
                new = True
                SA_result_arr.append(k.resultSA)
        SA_time_arr.append([float(timer() - startSA), new])

    SAtime = []
    t = 0
    for time, flag in SA_time_arr:
        if flag == True:
            SAtime.append(t + time)
            t = 0
        else:
            t += time
    print(len(SAtime))
    js = {"cplex" : {"solved": k.resultCPlex.tolist(),
                     "time": float(stopCPLEX),
                     "errors": k._checker(k.resultCPlex, k.T, k.L)}}

    js["SA"] = [{"solved": SA_result_arr[i].tolist(),
               "time": SAtime[i],
               "errors": k._checker(SA_result_arr[i], k.T, k.L)} for i in range(len(SAtime))]

    with open(name, "w") as f:
        json.dump(js, f, indent=4)
