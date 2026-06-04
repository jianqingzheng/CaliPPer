"""
Compatibility shim for torchtext.legacy

This module provides the legacy torchtext API for older code that depends on it.
Modern torchtext (>=0.14) removed the legacy module, so we recreate the essential
parts here for backward compatibility.

Author: Jianqing Zheng
Date: 2025.02.06
"""

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from collections import Counter, OrderedDict
import warnings

warnings.filterwarnings('ignore')


class Vocab:
    """Vocabulary class compatible with torchtext.legacy.vocab.Vocab"""

    def __init__(self, counter=None, max_size=None, min_freq=1, specials=('<unk>', '<pad>'),
                 vectors=None, unk_init=None, vectors_cache=None, specials_first=True):
        self.freqs = counter if counter is not None else Counter()
        self.stoi = {}  # string to index
        self.itos = []  # index to string

        # Add special tokens
        if specials_first:
            for tok in specials:
                self.itos.append(tok)
                self.stoi[tok] = len(self.itos) - 1

    def __len__(self):
        return len(self.itos)


class Field:
    """Field class compatible with torchtext.legacy.data.Field"""

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token='<pad>', unk_token='<unk>',
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = tokenize or (lambda s: s.split())
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.stop_words = stop_words
        self.is_target = is_target
        self.vocab = None

    def build_vocab(self, *args, **kwargs):
        """Build vocabulary"""
        self.vocab = Vocab()

    def preprocess(self, x):
        """Preprocess a single example"""
        if self.sequential and isinstance(x, str):
            x = self.tokenize(x)
        if self.lower:
            x = [w.lower() for w in x] if isinstance(x, list) else x.lower()
        if self.preprocessing is not None:
            x = self.preprocessing(x)
        return x

    def process(self, batch, device=None):
        """Process a batch of examples"""
        # For non-sequential fields (like labels), skip padding
        if not self.sequential:
            # Directly numericalize without padding
            tensor = self.numericalize(batch, device=device)
        else:
            # For sequential fields, pad first then numericalize
            padded = self.pad(batch)
            tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a minibatch"""
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded = []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) + list(x[-max_len:]) if self.truncate_first else
                    [self.pad_token] * max(0, max_len - len(x)) + list(x[:max_len]))
            else:
                padded.append(
                    list(x[-max_len:]) + [self.pad_token] * max(0, max_len - len(x)) if self.truncate_first else
                    list(x[:max_len]) + [self.pad_token] * max(0, max_len - len(x)))
        return padded

    def numericalize(self, arr, device=None):
        """Convert tokens to indices"""
        if self.use_vocab and self.vocab is not None:
            # Convert tokens to vocabulary indices
            if self.sequential:
                arr = [[self.vocab.stoi.get(x, self.vocab.stoi.get(self.unk_token, 0)) for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi.get(x, self.vocab.stoi.get(self.unk_token, 0)) for x in arr]
        elif not self.use_vocab and not self.sequential:
            # For non-vocab, non-sequential fields (like labels), ensure proper type conversion
            # Handle cases where values might be strings or need type conversion
            if self.dtype in [torch.float32, torch.float, torch.float64, torch.double, torch.float16, torch.half]:
                arr = [float(x) if not isinstance(x, (float, int)) else x for x in arr]
            elif self.dtype in [torch.int32, torch.int, torch.int64, torch.long, torch.int16, torch.short, torch.int8, torch.uint8]:
                arr = [int(x) if not isinstance(x, int) else x for x in arr]

        # Handle device: -1 means CPU in torchtext convention
        device_param = None if device == -1 else device
        arr = torch.tensor(arr, dtype=self.dtype)
        if device_param is not None:
            if isinstance(device_param, int) and device_param >= 0:
                arr = arr.to(f'cuda:{device_param}')
            elif device_param != -1:
                arr = arr.to(device_param)
        return arr


class RawField:
    """Raw field that doesn't process data"""

    def __init__(self, preprocessing=None, postprocessing=None, is_target=False):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.is_target = is_target

    def preprocess(self, x):
        if self.preprocessing is not None:
            return self.preprocessing(x)
        return x


class Example:
    """Example class compatible with torchtext.legacy.data.Example"""

    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if hasattr(field, 'sequential') and not field.sequential:
                    # For non-sequential fields, just keep the value as is
                    # torch.dtype is not callable, so we don't try to convert
                    val = val
                else:
                    # For sequential fields, preprocess if available
                    val = field.preprocess(val) if hasattr(field, 'preprocess') else val
                setattr(ex, name, val)
        return ex


class Dataset(TorchDataset):
    """Dataset class compatible with torchtext.legacy.data.Dataset"""

    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)


class Batch:
    """Batch class for holding batch data"""

    def __init__(self, data, dataset, device):
        self.dataset = dataset
        self.batch_size = len(data)
        self.fields = dataset.fields.keys()

        for field_name in dataset.fields:
            field = dataset.fields[field_name]
            batch_data = [getattr(ex, field_name) for ex in data]

            if hasattr(field, 'process'):
                # Use field's own process method if available
                tensor = field.process(batch_data, device=device)
            elif hasattr(field, 'sequential') and not field.sequential:
                # For non-sequential fields (like labels), convert directly
                tensor = torch.tensor(batch_data, dtype=getattr(field, 'dtype', torch.float32))
                # Handle device: -1 means CPU
                if device is not None and device != -1 and device >= 0:
                    tensor = tensor.to(f'cuda:{device}')
            else:
                # For sequential fields without process method, simple tensor conversion
                tensor = torch.tensor(batch_data)
                if device is not None and device != -1 and device >= 0:
                    tensor = tensor.to(f'cuda:{device}')

            setattr(self, field_name, tensor)


class Iterator:
    """Iterator class compatible with torchtext.legacy.data.Iterator"""

    def __init__(self, dataset, batch_size=32, device=None, repeat=False,
                 shuffle=False, sort=None, sort_within_batch=None, sort_key=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.repeat = repeat
        self.shuffle = shuffle
        self.iterations = 0
        self.epoch = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset)).tolist()
            else:
                indices = list(range(len(self.dataset)))

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_data = [self.dataset[idx] for idx in batch_indices]
                yield Batch(batch_data, self.dataset, self.device)

            self.epoch += 1
            if not self.repeat:
                break

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class Pipeline:
    """Pipeline class for data processing"""

    def __init__(self, convert_token=None):
        self.convert_token = convert_token or (lambda x: x)

    def __call__(self, x):
        return self.convert_token(x)


def get_tokenizer(tokenizer, language='en'):
    """Get a tokenizer function"""
    if tokenizer == 'basic_english':
        return lambda s: s.split()
    elif callable(tokenizer):
        return tokenizer
    else:
        return lambda s: s.split()


# Create legacy module structure
class LegacyData:
    Pipeline = Pipeline
    Dataset = Dataset
    Field = Field
    Iterator = Iterator
    Example = Example
    RawField = RawField
    get_tokenizer = get_tokenizer


class LegacyVocab:
    Vocab = Vocab


# For backwards compatibility
data = LegacyData()
vocab = LegacyVocab()
