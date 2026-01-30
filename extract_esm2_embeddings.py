#!/usr/bin/env python3
"""
Extract ESM2 Embeddings for Lipidated Peptides

This script extracts protein language model embeddings from ESM2 for lipidated
peptide sequences. It generates both mean-pooled representations and per-residue
embeddings for specific positions (the variable lipidation site).

Usage:
    python extract_esm2_embeddings.py \
        --model_name esm2_t33_650M_UR50D \
        --fasta_file data/all_sequences.fasta \
        --output_dir embeddings/ \
        --lipid_positions 2 3 4

The script outputs PyTorch .pt files containing:
    - mean_representations: Mean-pooled embedding over all residues
    - lipid_site_representations: Per-residue embeddings for specified positions
    - sequence: The input sequence
    - sequence_length: Length of the sequence

Author: [Your Name]
"""

import pathlib
import torch
import argparse
from typing import List, Dict

from esm import FastaBatchedDataset, pretrained


def extract_embeddings(
    model_name: str,
    fasta_file: pathlib.Path,
    output_dir: pathlib.Path,
    lipid_positions: List[int] = [2, 3, 4],
    tokens_per_batch: int = 4096,
    seq_length: int = 1022,
    repr_layers: List[int] = [33]
) -> None:
    """
    Extract ESM2 embeddings from protein sequences.

    Args:
        model_name: Name of pretrained ESM model (e.g., 'esm2_t33_650M_UR50D')
        fasta_file: Path to input FASTA file
        output_dir: Directory to save extracted embeddings
        lipid_positions: List of 1-indexed positions to extract (default: [2, 3, 4])
        tokens_per_batch: Maximum tokens per batch
        seq_length: Maximum sequence length
        repr_layers: Which model layers to extract representations from

    Returns:
        None. Embeddings are saved as .pt files in output_dir.
    """

    # Load model
    print(f"Loading model: {model_name}")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    # Use GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Load sequences
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting embeddings for {len(dataset)} sequences")
    print(f"Lipidation site positions: {lipid_positions}")
    print(f"Representation layers: {repr_layers}")
    print(f"Output directory: {output_dir}\n")

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations: Dict[int, torch.Tensor] = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                entry_id: str = label.split()[0]
                filename: pathlib.Path = output_dir / f"{entry_id}.pt"
                truncate_len: int = min(seq_length, len(strs[i]))

                result: Dict = {"entry_id": entry_id}

                # Mean representation (averaged over all residues)
                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                # Per-residue embeddings for lipidation sites
                # Note: tensor indices are offset by 1 due to BOS token
                result["lipid_site_representations"] = {}
                for pos in lipid_positions:
                    if pos <= truncate_len:
                        tensor_idx: int = pos  # Position in tensor (1-indexed in sequence)
                        result["lipid_site_representations"][f"position_{pos}"] = {
                            layer: t[i, tensor_idx].clone()
                            for layer, t in representations.items()
                        }
                    else:
                        print(f"Warning: Position {pos} exceeds sequence length for {entry_id}")

                # Store sequence metadata
                result["sequence"] = strs[i]
                result["sequence_length"] = len(strs[i])
                result["lipid_positions"] = lipid_positions

                torch.save(result, filename)

                # Print example for first sequence
                if batch_idx == 0 and i == 0:
                    print(f"\nExample output for {entry_id}:")
                    print(f"  Sequence: {strs[i][:50]}...")
                    print(f"  Length: {len(strs[i])}")
                    print(f"  Mean embedding shape: {result['mean_representations'][repr_layers[0]].shape}")
                    print(f"  Site embeddings: {list(result['lipid_site_representations'].keys())}\n")

    print(f"\nExtraction complete. Saved {len(dataset)} embeddings to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ESM2 embeddings for lipidated peptide sequences."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="Pretrained ESM model name (default: esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--fasta_file",
        type=pathlib.Path,
        required=True,
        help="Path to input FASTA file"
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Directory to save extracted embeddings"
    )
    parser.add_argument(
        "--lipid_positions",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="1-indexed positions of lipidation sites (default: 2 3 4)"
    )
    parser.add_argument(
        "--repr_layers",
        type=int,
        nargs="+",
        default=[33],
        help="Model layers to extract representations from (default: 33)"
    )

    args = parser.parse_args()

    extract_embeddings(
        args.model_name,
        args.fasta_file,
        args.output_dir,
        lipid_positions=args.lipid_positions,
        repr_layers=args.repr_layers
    )
