import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from collections import Counter
import pandas as pd
import random
from Bio import AlignIO

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
    base_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '-':[0, 0, 0, 0]}
    encoding = []
    for base in sequence:
        encoding.extend(base_dict[base])
    return np.array(encoding)

def assign_labels_for_segments(data, k, segment_count, segment_kmer_probabilities):
    labels = []
    for sequence in data:
        segment_labels = []
        for segment_index in range(segment_count):
            high_prob_kmers, _ = segment_kmer_probabilities[segment_index]

            start_idx = segment_index * k
            end_idx = start_idx + k
            kmer = sequence[start_idx:end_idx]

            if kmer in high_prob_kmers:
                segment_labels.extend(one_hot_encode(kmer))
            else:
                segment_labels.extend(one_hot_encode(random.choice(high_prob_kmers)))

        labels.append(np.array(segment_labels))
    return labels

def process_group(data, model, criterion, optimizer, epochs, patience, D, k, segment_count, segment_kmer_probabilities, lr):
    n = len(data)

    encoded_data = [one_hot_encode(seq) for seq in data]
    encoded_data = np.array(encoded_data)

    labels = assign_labels_for_segments(data, k, segment_count, segment_kmer_probabilities)
    labels = np.array(labels)

    input_data = torch.tensor(encoded_data, dtype=torch.float32)
    label_data = torch.tensor(labels, dtype=torch.float32)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(input_data)
        loss = criterion(outputs, label_data)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.eval()
    with torch.no_grad():
        reconstructed = model(input_data)

        reconstructed = reconstructed.view(-1, D, 4)
        input_data_bases = input_data.view(-1, D, 4)

        error_item = np.zeros((n,D//k))
        for sequence_idx in range(n):
            k_count = 0
            k_i = 0
            for i in range(D):
                reconstructed_item = torch.argmax(reconstructed[sequence_idx, i, :])
                input_item = torch.argmax(input_data_bases[sequence_idx, i, :])
                if torch.all(torch.abs(reconstructed[sequence_idx, i, :]) < 0.1):
                    reconstructed_item = torch.tensor(-1, dtype=torch.int64)
                if torch.all(input_data_bases[sequence_idx, i, :] == torch.tensor([0, 0, 0, 0], dtype=input_data_bases.dtype)):
                    input_item = torch.tensor(-1, dtype=torch.int64)
                if reconstructed_item != input_item:
                    error_item[sequence_idx,k_i] += 1
                k_count += 1
                if k_count == 4:
                    k_count = 0
                    k_i += 1
            
        p_errs = np.average(error_item, 1) / 4

        alpha_values = []
        for segment_index in range(segment_count):
            alpha_H = 0
            _, kmer_probabilities = segment_kmer_probabilities[segment_index]
            for p in kmer_probabilities.values():
                if p > 0:
                    alpha_H -= p * math.log(p, 2)
            alpha = alpha_H / 8
            alpha_values.append(alpha)

        max_alpha = max(alpha_values)
        Perr_limit = (2 * max_alpha * k) / (math.log(1 - 15/16, 2) + 2 * k)

        return Perr_limit, max_alpha, np.max(p_errs), np.average(p_errs), n, (sum(alpha_values)/len(alpha_values))

# Main code
D = 200
k = 4
    
alignment_files = [
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_000314.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_000457.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_000572.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_000594.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_001876.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_002524.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_004333.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_005522.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_005910.aln",
    "/data/wuchenyao/workspace/autodna/aligned_sequences_NM_033360.aln"
]

results = []

for alignment_file in alignment_files:
    alignment = AlignIO.read(alignment_file, "clustal")
    with open(alignment_file, "r") as file:
        raw_data = file.read()
    
    groups = raw_data.strip().split('\n\n')
    for group in groups:
        data = [line.split()[1][:200] for line in group.split('\n') if len(line.split()) > 1 and line.split()[1].count('-') < D // 2]

        if len(data) < 2 or len(data[0]) < D or len(set(data)) == 1:
            continue
        
        segment_count = D // k
        segment_kmer_probabilities = []

        for segment_index in range(segment_count):
            kmer_counter = Counter()
            for sequence in data:
                start_idx = segment_index * k
                end_idx = start_idx + k
                if end_idx <= len(sequence):
                    kmer = sequence[start_idx:end_idx]
                    kmer_counter[kmer] += 1
            kmer_total = sum(kmer_counter.values())
            kmer_probabilities = {kmer: count / kmer_total for kmer, count in kmer_counter.items()}
            sorted_kmers = sorted(kmer_probabilities.items(), key=lambda item: item[1], reverse=True)
            high_prob_kmers = [kmer for kmer, _ in sorted_kmers[:16]]
            segment_kmer_probabilities.append((high_prob_kmers, kmer_probabilities))

        input_dim = 4 * D
        hidden_dim_segment = 4
        try:
            model = DNAAutoencoder(input_dim=input_dim, hidden_dim_segment=hidden_dim_segment, segment_count=segment_count, k=k)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.15)
            result = process_group(data, model, criterion, optimizer, 2000, 10, D, k, segment_count, segment_kmer_probabilities, 0.15)
            result = result + (alignment_file,)
            results.append(result)
        except:
            print(group)
            print(alignment_file)
            print(kmer_probabilities)
        
results_df = pd.DataFrame(results, columns=['Perr_limit', 'Max_Alpha', 'Max_p_err', 'Average_p_err', 'n', 'Ave_alpha', 'alignment_file'])
results_df.to_excel('/data/wuchenyao/workspace/alignment_results.xlsx', index=False)