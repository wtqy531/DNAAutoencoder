from Bio import SeqIO, Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
import os

Entrez.email = "xxx@xxx.com" # your email address

def download_sequence(gene_id, output_file="S0.fasta"):
    handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text")
    with open(output_file, "w") as file:
        file.write(handle.read())

def run_blast(input_fasta):
    with open(input_fasta) as fasta_file:
        result_handle = NCBIWWW.qblast("blastn", "nt", fasta_file.read(), expect=0.01)
    return result_handle

def parse_blast_results(result_handle, num_hits=100, D_hits=400, output_file="similar_sequences.fasta"):
    blast_records = NCBIXML.parse(result_handle)
    sequences_written = 0
    sequences = []
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.align_length > D_hits:
                    sequences_written += 1
                    sequences.append(f">{alignment.title}\n{hsp.sbjct}\n")
    if sequences_written >= num_hits:
        with open(output_file, "w") as output:
            for sequence in sequences:
                output.write(sequence)
        print(f"Similar sequences saved to {output_file} (total: {sequences_written} sequences)")
    else:
        print(f"Not enough sequences found (total: {sequences_written} sequences). File not saved.")
    return sequences_written

def multiple_sequence_alignment(input_file, output_file="aligned_sequences.aln"):
    clustalomega_cline = ClustalOmegaCommandline(
        infile=input_file,
        outfile=output_file,
        verbose=True,
        auto=True,
        outfmt="clu",
        force=True,
        wrap=200,
        maxnumseq=10000
    )
    stdout, stderr = clustalomega_cline()
    alignment = AlignIO.read(output_file, "clustal")
    print(alignment)
    print(f"Alignment saved to {output_file}")

def main():
    gene_ids = [
        "NM_000314",  # PTEN
        "NM_033360",  # KRAS 
        "NM_004333",  # BRAF
        "NM_002524",  # NRAS
        "NM_000594",  # TNF
        "NM_000572",  # IL10
        "NM_005910",  # MAPT
        "NM_005522",  # HOXA1
        "NM_000457",  # HNF4A
        "NM_001876",  # CPT1A
    ]
    
    fasta_file = "/data/S0.fasta"
    num_hits = 60  
    D_hits = 200

    selected_gene_id_list = []
    for gene_id in gene_ids:
        print(f"Processing gene ID: {gene_id}")

        download_sequence(gene_id, fasta_file)

        result_handle = run_blast(fasta_file)

        similar_sequences_file = f"/data/similar_sequences_{gene_id}.fasta"
        sequence_count = parse_blast_results(result_handle, num_hits, D_hits, similar_sequences_file)

        if sequence_count >= num_hits:
            print(f"Gene ID {gene_id} successfully yielded {sequence_count} similar sequences.")
            selected_gene_id_list.append(gene_id)

            aligned_output_file = f"/data/aligned_sequences_{gene_id}.aln"
            multiple_sequence_alignment(similar_sequences_file, aligned_output_file)
        else:
            print(f"Gene ID {gene_id} did not yield enough similar sequences. Trying next gene ID...")

    print("Pipeline completed successfully!")
    print(selected_gene_id_list)

if __name__ == "__main__":
    main()
