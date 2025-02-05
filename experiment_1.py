import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import Counter
import pandas as pd
import random

# Hyperparameters
n = 1000  # Number of DNA sequences
D = 400  # Length of each DNA sequence
k = 4  # Length of each segment
H = 8
beta = 15 / 16
epochs = 2000
lr = 0.15
patience = 10

class DNAAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_segment, segment_count, k):
        super(DNAAutoencoder, self).__init__()
        self.segment_count = segment_count
        self.k = k
        self.hidden_dim_segment = hidden_dim_segment

        self.encoders = nn.ModuleList([nn.Linear(4 * k, hidden_dim_segment) for _ in range(segment_count)])
        self.decoders = nn.ModuleList([nn.Linear(hidden_dim_segment, 4 * k) for _ in range(segment_count)])

        self.gelu = torch.nn.GELU()

    def forward(self, x):
        segments = x.view(-1, self.segment_count, 4 * self.k)  
        encoded_segments = []
        decoded_segments = []

        for i in range(self.segment_count):
            encoded_segment = self.gelu(self.encoders[i](segments[:, i, :]))
            decoded_segment = torch.sigmoid(self.decoders[i](encoded_segment))
            encoded_segments.append(encoded_segment)
            decoded_segments.append(decoded_segment)

        decoded_output = torch.cat(decoded_segments, dim=1)
        return decoded_output

def one_hot_encode(sequence):
    base_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoding = []
    for base in sequence:
        encoding.extend(base_dict[base])
    return np.array(encoding)

segment_count = D // k
hidden_dim_segment = 4  
input_dim = 4 * D  
hidden_dim = hidden_dim_segment * segment_count 

dna_bases = ['A', 'C', 'G', 'T']
probabilities = [0.7, 0.05, 0.2, 0.05]
data = []
for _ in range(n):
    sequence = []
    for _ in range(D // k):
        if np.random.rand() < 0.7:
            sequence.append('CCGT')
        elif np.random.rand() < 0.7:
            sequence.append('GTTG')
        elif np.random.rand() < 0.1:
            sequence.append('TGCT')
        else:
            if np.random.rand() < 0.3:
                sequence.append('TC' + ''.join(np.random.choice(['A', 'C', 'G', 'T'], 2, p=probabilities)))
            elif np.random.rand() < 0.6:
                sequence.append(''.join(np.random.choice(['A', 'C', 'G', 'T'], 2, p=probabilities)) + 'TC')
            elif np.random.rand() < 0.7:
                sequence.append(''.join(np.random.choice(['A', 'C', 'G', 'T'], 2, p=probabilities)) + 'CG')
            elif np.random.rand() < 0.8:
                sequence.append('G' + ''.join(np.random.choice(['A', 'C', 'G', 'T'], 2, p=probabilities)) + 'C')
            else:
                sub_sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], k, p=probabilities))
                sequence.append(sub_sequence)
    data.append(''.join(sequence))

encoded_data = [one_hot_encode(seq) for seq in data] 
encoded_data = np.array(encoded_data)

kmer_counter = Counter()
for sequence in data:
    for segment_index in range(segment_count):
        start_idx = segment_index * k
        end_idx = start_idx + k
        if end_idx <= len(sequence):
            kmer = sequence[start_idx:end_idx]
            kmer_counter[kmer] += 1

kmer_total = sum(kmer_counter.values())
kmer_probabilities = {kmer: count / kmer_total for kmer, count in kmer_counter.items()}
sorted_kmers = sorted(kmer_probabilities.items(), key=lambda item: item[1], reverse=True)
high_prob_kmers = [kmer for kmer, _ in sorted_kmers[:16]]

def assign_labels_for_segments(data, k, segment_count, high_prob_kmers):
    labels = []
    for sequence in data:
        segment_labels = []
        for segment_index in range(segment_count):
            start_idx = segment_index * k
            end_idx = start_idx + k
            kmer = sequence[start_idx:end_idx]

            if kmer in high_prob_kmers:
                segment_labels.extend(one_hot_encode(kmer))
            else:
                segment_labels.extend(one_hot_encode(random.choice(high_prob_kmers)))

        labels.append(np.array(segment_labels))
    return labels

labels = assign_labels_for_segments(data, k, segment_count, high_prob_kmers)
labels = np.array(labels)  

input_data = torch.tensor(encoded_data, dtype=torch.float32) 
label_data = torch.tensor(labels, dtype=torch.float32) 

model = DNAAutoencoder(input_dim=input_dim, hidden_dim_segment=hidden_dim_segment, segment_count=segment_count, k=k)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_loss = float('inf')
patience_counter = 0
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(input_data)
    loss = criterion(outputs, label_data)

    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    if  loss.item() - best_loss < -1e-6:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

def decode_one_hot(one_hot_array):
    base_dict = {tuple([1, 0, 0, 0]): 'A', tuple([0, 1, 0, 0]): 'C', tuple([0, 0, 1, 0]): 'G', tuple([0, 0, 0, 1]): 'T'}
    sequence = []
    for i in range(0, len(one_hot_array), 4):
        base_encoding = tuple(one_hot_array[i:i + 4].tolist())
        base = base_dict.get(base_encoding, '?')  
        sequence.append(base)
    return ''.join(sequence)

def decode_one_hot2(one_hot_array):
    sequence = []
    for i in range(len(one_hot_array)):
        base_encoding = one_hot_array[i]
        if base_encoding == 0:
            base = 'A'
        elif base_encoding == 1:
            base = 'C'
        elif base_encoding == 2:
            base = 'G'
        elif base_encoding == 3:
            base = 'T'
        else:
            base = '?'
        sequence.append(base)
    return ''.join(sequence)

model.eval()
with torch.no_grad():
    reconstructed = model(input_data)
    mse_loss = criterion(reconstructed, label_data)

    reconstructed = reconstructed.view(-1, D, 4) 
    label_data_bases = label_data.view(-1, D, 4)
    input_data_bases = input_data.view(-1, D, 4)
    p_errs = []
    p_errs_m = []

    for sequence_idx in range(n):
        total_bases = D//k

        k_count = 0
        incorrect_flag = 0
        incorrect_bases = 0
        for i in range(D):
            if torch.argmax(reconstructed[sequence_idx, i, :]) != torch.argmax(input_data_bases[sequence_idx, i, :]):
                incorrect_flag = 1
            k_count += 1
            if k_count == 4:
                if incorrect_flag == 1:
                    incorrect_bases += 1
                incorrect_flag = 0
                k_count = 0
        
        p_err = incorrect_bases / total_bases
        p_errs.append(p_err)

        k_count = 0
        incorrect_flag = 0
        incorrect_bases = 0
        for i in range(D):
            if torch.argmax(reconstructed[sequence_idx, i, :]) != torch.argmax(label_data_bases[sequence_idx, i, :]):
                incorrect_flag = 1
            k_count += 1
            if k_count == 4:
                if incorrect_flag == 1:
                    incorrect_bases += 1
                incorrect_flag = 0
                k_count = 0
        
        p_err_m = incorrect_bases / total_bases
        p_errs_m.append(p_err_m)

    input_sequence = decode_one_hot(input_data_bases[sequence_idx].flatten())
    label_sequence = decode_one_hot(label_data_bases[sequence_idx].flatten())
    output_sequence = decode_one_hot2(torch.argmax(reconstructed[sequence_idx], dim=1))

    print(f"Sequence {sequence_idx + 1}:")
    print(f"  Input Sequence:   {input_sequence}")
    print(f"  Label Sequence:   {label_sequence}")
    print(f"  Output Sequence:  {output_sequence}")
    print(f"  P_err:            {p_err:.4f}")
    print("")

    df_p_errs = pd.DataFrame({'Sequence_Index': range(n), 'P_err': p_errs})
    df_p_errs.to_excel('/data/wuchenyao/workspace/autodna/experiment_1_p_err_values.xlsx', index=False)

    alpha_H = 0
    for p in kmer_probabilities.values():
        if p > 0:
            alpha_H -= p * math.log(p, 2)
    alpha = alpha_H / H

    Perr_limit = (2 * alpha * k) / (math.log(1 - beta, 2) + 2 * k)

    print(f"Final MSE Loss: {mse_loss.item():.4f}")
    print(f"Validation Error Rate (P_err) saved to experiment_1_p_err_values.xlsx")
    print(f"Perr_limit: {Perr_limit:.4f}")

    print(f"sorted_kmers: {sorted_kmers}")
    print(f"alpha: {alpha:.4f}")
    print(f"sum of highest 16 p: {sum([prob for _, prob in sorted_kmers[:16]]):.4f}")
    print(f"Max p_err: {np.max(p_errs):.4f}")
    print(f"Average p_err: {np.average(p_errs):.4f}")
    print(f"Average model p_err: {np.average(p_errs_m):.4f}")
