"""
BBPE Tokenizer wrapper for SecurityBERT.
SecurityBERT Paper — Section III-D.

Wraps HuggingFace ByteLevelBPETokenizer with
paper-exact parameters:
    vocab_size    = 5,000
    min_frequency = 2
    special_tokens= ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    max_seq_len   = 512
"""

import json
import time
from pathlib import Path
from typing  import List, Optional, Tuple, Union

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


# ─────────────────────────────────────────────────────────────────────────────
# PPFLETokenizer
# ─────────────────────────────────────────────────────────────────────────────

class PPFLETokenizer:
    """
    ByteLevelBPE tokenizer trained on PPFLE-encoded data.

    Paper Section III-D:
    "During the training of the tokenizer, a vocabulary size of 5000
    was employed, along with a set of specific tokens, including
    ['<s>', '<pad>', '</s>', '<unk>', '<mask>']."

    Parameters
    ----------
    vocab_size     : vocabulary size (paper: 5000)
    min_frequency  : minimum token frequency (paper: 2)
    max_seq_len    : maximum sequence length (paper: 512)
    special_tokens : list of special tokens
    """

    SPECIAL_TOKENS = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']

    def __init__(
        self,
        vocab_size    : int  = 5_000,
        min_frequency : int  = 2,
        max_seq_len   : int  = 512,
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab_size     = vocab_size
        self.min_frequency  = min_frequency
        self.max_seq_len    = max_seq_len
        self.special_tokens = special_tokens or self.SPECIAL_TOKENS
        self._tokenizer     = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, corpus_path: Union[str, Path]) -> None:
        """
        Train ByteLevelBPETokenizer on corpus file.

        Parameters
        ----------
        corpus_path : path to plain text corpus (one sequence per line)
        """
        corpus_path = Path(corpus_path)
        assert corpus_path.exists(), f'Corpus not found: {corpus_path}'

        print(f'🏋️  Training ByteLevelBPETokenizer …')
        print(f'   Corpus        : {corpus_path}  '
              f'({corpus_path.stat().st_size/1e6:.1f} MB)')
        print(f'   Vocab size    : {self.vocab_size:,}')
        print(f'   Min frequency : {self.min_frequency}')
        print(f'   Special tokens: {self.special_tokens}')

        self._tokenizer = ByteLevelBPETokenizer()
        t0 = time.time()

        self._tokenizer.train(
            files         = [str(corpus_path)],
            vocab_size    = self.vocab_size,
            min_frequency = self.min_frequency,
            special_tokens= self.special_tokens,
        )

        elapsed = time.time() - t0
        self._add_post_processor()
        self._enable_truncation_padding()

        print(f'✅ Training complete in {elapsed:.1f}s  '
              f'(vocab: {self._tokenizer.get_vocab_size():,})')

    def _add_post_processor(self) -> None:
        """Add BERT-style CLS/SEP post-processing."""
        self._tokenizer.post_processor = BertProcessing(
            sep=('</s>', self._tokenizer.token_to_id('</s>')),
            cls=('<s>',  self._tokenizer.token_to_id('<s>')),
        )

    def _enable_truncation_padding(self) -> None:
        """Enable truncation and padding to max_seq_len."""
        self._tokenizer.enable_truncation(max_length=self.max_seq_len)
        self._tokenizer.enable_padding(
            pad_id    = self._tokenizer.token_to_id('<pad>'),
            pad_token = '<pad>',
            length    = self.max_seq_len,
        )

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, tokenizer_dir: Union[str, Path]) -> None:
        """
        Save tokenizer vocab.json and merges.txt.

        Parameters
        ----------
        tokenizer_dir : directory to save tokenizer files
        """
        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        assert self._tokenizer is not None, 'Train tokenizer first.'
        self._tokenizer.save_model(str(tokenizer_dir))
        print(f'💾 Tokenizer saved → {tokenizer_dir}')

    @classmethod
    def load(
        cls,
        tokenizer_dir : Union[str, Path],
        max_seq_len   : int = 512,
        special_tokens: Optional[List[str]] = None,
    ) -> 'PPFLETokenizer':
        """
        Load a trained tokenizer from saved vocab.json + merges.txt.

        Parameters
        ----------
        tokenizer_dir : directory containing vocab.json + merges.txt
        max_seq_len   : maximum sequence length
        special_tokens: special tokens list

        Returns
        -------
        PPFLETokenizer instance
        """
        tokenizer_dir = Path(tokenizer_dir)
        vocab_path    = tokenizer_dir / 'vocab.json'
        merges_path   = tokenizer_dir / 'merges.txt'

        assert vocab_path.exists(),  f'vocab.json not found: {vocab_path}'
        assert merges_path.exists(), f'merges.txt not found: {merges_path}'

        instance = cls(max_seq_len=max_seq_len,
                       special_tokens=special_tokens)
        instance._tokenizer = ByteLevelBPETokenizer(
            vocab  = str(vocab_path),
            merges = str(merges_path),
        )
        instance._add_post_processor()
        instance._enable_truncation_padding()
        print(f'✅ Tokenizer loaded from {tokenizer_dir}  '
              f'(vocab: {instance._tokenizer.get_vocab_size():,})')
        return instance

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, sequence: str) -> dict:
        """
        Encode a single PPFLE sequence.

        Parameters
        ----------
        sequence : str — space-separated MD5 hash tokens

        Returns
        -------
        dict:
            input_ids      : List[int]
            attention_mask : List[int]
        """
        assert self._tokenizer is not None, 'Train or load tokenizer first.'
        enc = self._tokenizer.encode(sequence)
        return {
            'input_ids'     : enc.ids,
            'attention_mask': enc.attention_mask,
        }

    def encode_batch(
        self,
        sequences : List[str],
        chunk_size: int = 5_000,
        verbose   : bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithm 2 — Encode Evaluation Data Sequences.
        SecurityBERT Paper Section III-E.

        Processes sequences in chunks of chunk_size to avoid OOM.

        Parameters
        ----------
        sequences  : list of PPFLE-encoded sequences
        chunk_size : paper default = 5000
        verbose    : print progress

        Returns
        -------
        input_ids_eval       : np.ndarray (N, max_seq_len)
        attention_masks_eval : np.ndarray (N, max_seq_len)
        """
        assert self._tokenizer is not None, 'Train or load tokenizer first.'

        N          = len(sequences)
        num_chunks = -(-N // chunk_size)   # ceiling division

        # Algorithm 2 lines 3-4: initialize empty lists
        input_ids_eval       = []
        attention_masks_eval = []

        if verbose:
            print(f'🔄 Algorithm 2: encoding {N:,} sequences '
                  f'(chunk_size={chunk_size:,}, chunks={num_chunks})')

        t_start = time.time()

        # Algorithm 2 line 5: for i = 0 to num_chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size          # line 6
            end_idx   = (i + 1) * chunk_size    # line 7
            chunk     = sequences[start_idx:end_idx]  # line 8

            # Line 9: encoded_seqs ← encode(chunk)
            encoded_seqs = self._tokenizer.encode_batch(chunk)

            # Line 10: iic, amc ← UNPACK(encoded_seqs)
            iic = [enc.ids             for enc in encoded_seqs]
            amc = [enc.attention_mask  for enc in encoded_seqs]

            # Lines 11-12: append to lists
            input_ids_eval.extend(iic)
            attention_masks_eval.extend(amc)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                done    = min((i + 1) * chunk_size, N)
                print(
                    f'   chunk {i+1:>4}/{num_chunks}  |  '
                    f'{done:>8,}/{N:,}  |  '
                    f'{done/elapsed:>6,.0f} rows/s'
                )

        # Algorithm 2 line 14: concatenate along dimension 0
        input_ids_eval       = np.array(input_ids_eval,       dtype=np.int32)
        attention_masks_eval = np.array(attention_masks_eval, dtype=np.int8)

        if verbose:
            elapsed = time.time() - t_start
            print(f'✅ Encoding complete: {elapsed:.1f}s')

        return input_ids_eval, attention_masks_eval

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size_actual(self) -> int:
        """Actual vocabulary size after training."""
        assert self._tokenizer is not None
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        """Get token ID by token string."""
        assert self._tokenizer is not None
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Get token string by token ID."""
        assert self._tokenizer is not None
        return self._tokenizer.id_to_token(token_id)

    def get_vocab(self) -> dict:
        """Return full vocabulary dictionary."""
        assert self._tokenizer is not None
        return self._tokenizer.get_vocab()
