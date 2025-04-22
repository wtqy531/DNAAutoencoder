import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import Counter
from itertools import product
import pandas as pd

# Hyperparameters
n = 300  # Number of DNA sequences
D = 60  # Length of each DNA sequence
k = 3  # Length of each segment
lr = 0.15
epochs = 1000
patience = 10

segment_count = D // k
hidden_dim_segment = 3 
input_dim = 4 * D 
hidden_dim = hidden_dim_segment * segment_count  

dna_bases = ['A', 'C', 'G', 'T']
data = [''.join(np.random.choice(dna_bases, D)) for _ in range(n)]

def one_hot_encode(sequence):
    base_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoding = []
    for base in sequence:
        encoding.extend(base_dict[base])
    return np.array(encoding)

encoded_data = [one_hot_encode(seq) for seq in data]  # Shape: (n, 4*D)
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
high_prob_kmers = [kmer for kmer, _ in sorted_kmers[:8]]

def hamming_distance(kmer1, kmer2):
    return sum(c1 != c2 for c1, c2 in zip(kmer1, kmer2))

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
                # Find the nearest high probability k-mer
                nearest_kmer = min(high_prob_kmers, key=lambda x: hamming_distance(kmer, x))
                segment_labels.extend(one_hot_encode(nearest_kmer))

        labels.append(np.array(segment_labels))
    return labels

labels = assign_labels_for_segments(data, k, segment_count, high_prob_kmers)
labels = np.array(labels)  
labels_ = np.copy(encoded_data) 

input_data = torch.tensor(encoded_data, dtype=torch.float32) 
label_data = torch.tensor(labels, dtype=torch.float32) 

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

    if loss.item() - best_loss < -1e-6:
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
    input_data_bases = input_data.view(-1, D, 4)
    label_data_bases = label_data.view(-1, D, 4)
    p_errs = []

    for sequence_idx in range(n):
        incorrect_bases = 0
        total_bases = D

        for i in range(D):
            if torch.argmax(reconstructed[sequence_idx, i, :]) != torch.argmax(input_data_bases[sequence_idx, i, :]):
                incorrect_bases += 1
        
        p_err = incorrect_bases / total_bases
        p_errs.append(p_err)

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
    df_p_errs.to_excel('/data/experiment_2_p_err_values.xlsx', index=False)

    print(f"Final MSE Loss: {mse_loss.item():.4f}")
    print(f"Validation Error Rate (P_err) saved to experiment_2_p_err_values.xlsx")
    print(f"Perr_limit: 1/3")
    print(f"Average p_errs: {np.average(p_errs):.4f}")
    print(f"sorted_kmers: {[f'({kmer}, {prob:.4f})' for kmer, prob in sorted_kmers]}")
