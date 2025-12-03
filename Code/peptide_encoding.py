import re

import re
import numpy as np

def checkFasta(fastas):
    status = True
    lenList = set()
    for i in fastas:
        lenList.add(len(i[1]))
    if len(lenList) == 1:
        return True
    else:
        return False

def minSequenceLength(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def minSequenceLengthWithNormalAA(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(re.sub('-', '', i[1])):
            minLen = len(re.sub('-', '', i[1]))
    return minLen


def BINARY(fastas, **kw):
    if checkFasta(fastas) == False:
        print('Error: for "BINARY" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0

    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = ['#']
    for i in range(1, len(fastas[0][1]) * 20 + 1):
        header.append('BINARY.F'+str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for aa in sequence:
            if aa == '-':
                code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                continue
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        encodings.append(code)
    return encodings
#
def BLOSUM62(fastas, **kw):
    if checkFasta(fastas) == False:
        print('Error: for "BLOSUM62" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0

    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }
    encodings = []
    header = ['#']
    for i in range(1, len(fastas[0][1]) * 20 + 1):
        header.append('blosum62.F'+str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for aa in sequence:
            code = code + blosum62[aa]
        encodings.append(code)
    return encodings


def ZSCALE(fastas, **kw):
    if checkFasta(fastas) == False:
        print('Error: for "ZSCALE" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0

    zscale = {
        'A': [0.24,  -2.32,  0.60, -0.14,  1.30], # A
        'C': [0.84,  -1.67,  3.71,  0.18, -2.65], # C
        'D': [3.98,   0.93,  1.93, -2.46,  0.75], # D
        'E': [3.11,   0.26, -0.11, -0.34, -0.25], # E
        'F': [-4.22,  1.94,  1.06,  0.54, -0.62], # F
        'G': [2.05,  -4.06,  0.36, -0.82, -0.38], # G
        'H': [2.47,   1.95,  0.26,  3.90,  0.09], # H
        'I': [-3.89, -1.73, -1.71, -0.84,  0.26], # I
        'K': [2.29,   0.89, -2.49,  1.49,  0.31], # K
        'L': [-4.28, -1.30, -1.49, -0.72,  0.84], # L
        'M': [-2.85, -0.22,  0.47,  1.94, -0.98], # M
        'N': [3.05,   1.62,  1.04, -1.15,  1.61], # N
        'P': [-1.66,  0.27,  1.84,  0.70,  2.00], # P
        'Q': [1.75,   0.50, -1.44, -1.34,  0.66], # Q
        'R': [3.52,   2.50, -3.50,  1.99, -0.17], # R
        'S': [2.39,  -1.07,  1.15, -1.39,  0.67], # S
        'T': [0.75,  -2.18, -1.12, -1.46, -0.40], # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02], # V
        'W': [-4.36,  3.94,  0.59,  3.44, -1.59], # W
        'Y': [-2.54,  2.44,  0.43,  0.04, -1.47], # Y
        '-': [0.00,   0.00,  0.00,  0.00,  0.00], # -
    }
    encodings = []
    header = ['#']
    for p in range(1, len(fastas[0][1])+1):
        for z in ('1', '2', '3', '4', '5'):
            header.append('Pos'+str(p) + '.ZSCALE' + z)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for aa in sequence:
            code = code + zscale[aa]
        encodings.append(code)
    return encodings



import torch
import numpy as np



def feature_generator(file_path, max_rows=50, truncate_mode="head"):

    with open(file_path, 'r', encoding='utf-8') as f:
        fasta_list = np.array(f.readlines())

    datas = []
    sequences = []

    for flag in range(0, len(fasta_list), 2):
        header = fasta_list[flag].strip('\n').strip()
        seq    = fasta_list[flag + 1].strip('\n').strip()
        sequences.append(seq)


        fasta_str = [[header, seq]]
        bin_output = BINARY(fasta_str)
        blo_output = BLOSUM62(fasta_str)
        zsl_output = ZSCALE(fasta_str)


        bin_output[1].remove(bin_output[1][0])
        blo_output[1].remove(blo_output[1][0])
        zsl_output[1].remove(zsl_output[1][0])


        bin_feature = [bin_output[1][i:i + 20] for i in range(0, len(bin_output[1]), 20)]
        blo_feature = [blo_output[1][i:i + 20] for i in range(0, len(blo_output[1]), 20)]
        zsl_feature = [zsl_output[1][i:i + 5]  for i in range(0, len(zsl_output[1]), 5)]


        aa_fea_matrx = np.hstack([
            np.array(bin_feature),
            np.array(blo_feature),
            np.array(zsl_feature)
        ]).astype(np.float32)

        datas.append(aa_fea_matrx)


    padded_datas = []
    truncated_cnt = 0

    for data in datas:
        L, C = data.shape
        if L > max_rows:
            truncated_cnt += 1
            if truncate_mode == "head":
                data_proc = data[:max_rows, :]
            elif truncate_mode == "tail":
                data_proc = data[-max_rows:, :]
            elif truncate_mode == "center":
                start = (L - max_rows) // 2
                data_proc = data[start:start + max_rows, :]
            else:
                raise ValueError(f"Unknown truncate_mode: {truncate_mode}")
        elif L < max_rows:
            pad = np.zeros((max_rows - L, C), dtype=data.dtype)
            data_proc = np.vstack([data, pad])
        else:
            data_proc = data

        padded_datas.append(data_proc)

    padded_datas_array = np.stack(padded_datas, axis=0).astype(np.float32)
    return padded_datas_array
