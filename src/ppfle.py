"""
Privacy-Preserving Fixed-Length Encoding (PPFLE)
SecurityBERT Paper — Section III-C, Algorithm 1.

PPFLE converts raw network traffic features into hashed
text sequences that BERT can process. Two objectives:
1. Linguistic alignment  — converts numbers to text format
2. Privacy preservation  — MD5 hashing hides original values

Algorithm 1:
    procedure PPFLE(M):
        DL ← []
        for m = 1 to i do
            L = []
            for n = 1 to j do
                L ← H(s(m, n))
            end for
            DL ← L
        end for
        return DL
    end procedure
"""

import hashlib
import time
from pathlib import Path
from typing  import List, Tuple, Optional

import pandas as pd
import numpy  as np


# ─────────────────────────────────────────────────────────────────────────────
# Core PPFLE functions
# ─────────────────────────────────────────────────────────────────────────────

def H(x: str) -> str:
    """
    H(x) — Cryptographic hash function.

    Uses MD5 (Message Digest 5):
    - Output: 32-character hexadecimal string (fixed-length)
    - Privacy: original value cannot be recovered
    - Speed: fast enough for 2M+ rows

    Parameters
    ----------
    x : str — input string to hash

    Returns
    -------
    str — 32-character MD5 hex digest
    """
    return hashlib.md5(x.encode('utf-8')).hexdigest()


def s(column_name: str, value) -> str:
    """
    s(i, j) = column_name ∥ "$" ∥ value

    Concatenation operation — Equation (1) of the paper.
    Adds context to the value by prepending its column name.

    Parameters
    ----------
    column_name : str — feature column name (cj)
    value       : any — feature value at row i, column j

    Returns
    -------
    str — concatenated string ready for hashing
    """
    return f'{column_name}${value}'


def ppfle(
    M          : pd.DataFrame,
    target_col : str,
) -> Tuple[List[str], pd.Series, List[str]]:
    """
    Privacy-Preserving Fixed-Length Encoding — Algorithm 1.

    For each row i and each feature column j:
        L[j] = H( s(column_name_j, M[i, j]) )
             = MD5( column_name_j + "$" + str(value) ).hexdigest()

    Full row encoding:
        DL[i] = space-joined list of all hashed tokens

    Parameters
    ----------
    M          : pd.DataFrame — feature matrix (rows=samples, cols=features)
    target_col : str          — label column name (excluded from encoding)

    Returns
    -------
    encoded_sequences : List[str]  — one PPFLE sequence per row
    labels            : pd.Series  — label column (unchanged)
    feature_cols      : List[str]  — feature columns that were encoded
    """
    feature_cols = [c for c in M.columns if c != target_col]
    labels       = M[target_col].reset_index(drop=True)

    col_names  = feature_cols
    values_arr = M[feature_cols].astype(str).values

    # ── Algorithm 1 ──────────────────────────────────────────────────────────
    # DL ← []
    encoded_sequences = []

    for row_vals in values_arr:                    # for m = 1 to i
        L = [
            H(s(col_names[n], row_vals[n]))        # L ← H(s(m, n))
            for n in range(len(col_names))         # for n = 1 to j
        ]
        encoded_sequences.append(' '.join(L))      # DL ← L

    return encoded_sequences, labels, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Chunked PPFLE for large datasets
# ─────────────────────────────────────────────────────────────────────────────

def ppfle_chunked(
    df         : pd.DataFrame,
    target_col : str,
    output_path: Path,
    chunk_size : int = 50_000,
    verbose    : bool = True,
) -> int:
    """
    Apply PPFLE to a large DataFrame in chunks,
    streaming output directly to CSV to avoid OOM.

    Parameters
    ----------
    df          : full DataFrame
    target_col  : label column name
    output_path : path to write encoded CSV
    chunk_size  : rows per chunk
    verbose     : print progress

    Returns
    -------
    int — total rows written
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows  = 0
    write_header= True
    t_start     = time.time()

    for chunk_idx, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start : start + chunk_size].copy()

        encoded_seqs, lbls, _ = ppfle(chunk, target_col=target_col)

        chunk_out = pd.DataFrame({
            'encoded_sequence' : encoded_seqs,
            target_col         : lbls.values,
        })

        chunk_out.to_csv(
            output_path,
            mode   = 'w' if write_header else 'a',
            header = write_header,
            index  = False,
        )
        write_header  = False
        total_rows   += len(chunk_out)

        if verbose and (chunk_idx + 1) % 5 == 0:
            elapsed      = time.time() - t_start
            rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
            pct_done     = total_rows / len(df) * 100
            print(
                f'   chunk {chunk_idx+1:>4}  |  '
                f'{total_rows:>10,} rows ({pct_done:>5.1f}%)  |  '
                f'{rows_per_sec:>8,.0f} rows/s'
            )

    if verbose:
        elapsed = time.time() - t_start
        print(
            f'\n✅ PPFLE complete: {total_rows:,} rows in {elapsed:.1f}s  '
            f'({total_rows/elapsed:,.0f} rows/s)'
        )

    return total_rows


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_ppfle_output(
    encoded_sequences: List[str],
    expected_n_tokens: Optional[int] = None,
) -> dict:
    """
    Validate PPFLE encoded sequences.

    Checks:
    1. All tokens are exactly 32 chars (MD5 fixed-length)
    2. All sequences have the same number of tokens
    3. No empty sequences

    Parameters
    ----------
    encoded_sequences : list of PPFLE sequences
    expected_n_tokens : expected number of tokens per sequence

    Returns
    -------
    dict — validation results
    """
    all_lengths  = []
    all_n_tokens = []
    errors       = []

    for idx, seq in enumerate(encoded_sequences):
        if not seq or not seq.strip():
            errors.append(f'Row {idx}: empty sequence')
            continue
        toks = seq.split()
        all_n_tokens.append(len(toks))
        for tok in toks:
            all_lengths.append(len(tok))
            if len(tok) != 32:
                errors.append(f'Row {idx}: token length {len(tok)} ≠ 32')

    unique_tok_len  = set(all_lengths)
    unique_n_tokens = set(all_n_tokens)

    passed = (
        unique_tok_len == {32}
        and len(errors) == 0
        and (expected_n_tokens is None
             or unique_n_tokens == {expected_n_tokens})
    )

    return {
        'passed'         : passed,
        'unique_tok_lens': unique_tok_len,
        'unique_n_tokens': unique_n_tokens,
        'errors'         : errors,
        'n_sequences'    : len(encoded_sequences),
    }
