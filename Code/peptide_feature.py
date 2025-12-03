import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import iFeatureOmegaCLI
import re
import math

def DDE(fastas, **kw):

    AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6, 'M': 1, 'N': 2, 'P': 4,
                'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}

    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#'] + diPeptides
    encodings.append(header)
    myTM = [(myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61) for pair in diPeptides]
    AADict = {aa: i for i, aa in enumerate(AA)}

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            if sequence[j] in AADict and sequence[j + 1] in AADict:
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] += 1
        if sum(tmpCode) != 0:
            tmpCode = [x / sum(tmpCode) for x in tmpCode]
        myTV = [(myTM[j] * (1 - myTM[j]) / (len(sequence) - 1)) for j in range(len(myTM))]
        tmpCode = [(tmpCode[j] - myTM[j]) / math.sqrt(myTV[j]) if myTV[j] != 0 else 0 for j in range(len(tmpCode))]
        encodings.append([name] + tmpCode)
    return encodings

def feature_DDE(file_path):

    fasta_list = open(file_path, 'r', encoding='utf-8').readlines()
    aa_feature_list = []
    for flag in range(0, len(fasta_list), 2):
        fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]
        dpc_output = DDE(fasta_str)
        dpc_output[1].remove(dpc_output[1][0])
        aa_feature_list.append(dpc_output[1][:])
    aa_feature_list = pd.DataFrame(aa_feature_list)
    aa_feature_list.columns = [f'DDE{i + 1}' for i in range(len(aa_feature_list.columns))]
    return aa_feature_list

def generate_features(input_txt_path):

    descriptors = ["DistancePair", "CKSAAGP type 2", "QSOrder"]
    features = []

    for descriptor in descriptors:
        protein_descriptor = iFeatureOmegaCLI.iProtein(input_txt_path)
        protein_descriptor.get_descriptor(descriptor)
        protein_descriptor.display_feature_types()
        protein_descriptor.encodings = protein_descriptor.encodings.reset_index(drop=True)
        features.append(protein_descriptor.encodings)

    dde = feature_DDE(input_txt_path).reset_index(drop=True)
    features.append(dde)


    result = pd.concat(features, axis=1)
    return result

def main(input_txt_path, output_txt_path):

    test_df = generate_features(input_txt_path)
    x2 = torch.tensor(test_df.values, dtype=torch.float32)
    print(f"Feature shape: {x2.shape}")
    np.savetxt(output_txt_path, x2.numpy())


inputfile = 'train.txt'
outputfile = 'train_feature.txt'

dataset = main(inputfile, outputfile)
