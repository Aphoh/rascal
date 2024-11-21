import torch
from torch.utils.data import Dataset, DataLoader
import random

class DummySequenceDataset(Dataset):
    def __init__(self, num_samples, vocab_size, max_seq_len, pad_token):
        """
        Initialize the dummy sequence dataset.
        
        Args:
            num_samples (int): Number of sequences to generate
            vocab_size (int): Maximum integer value in sequences (exclusive)
            max_seq_len (int): Maximum sequence length
            pad_token (int): Token to use for padding
        """
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        
        # Generate all sequences upfront
        self.sequences = []
        self.sequence_lengths = []
        self.suffix_indices = []
        
        for _ in range(num_samples):
            # Generate random sequence length
            seq_len = random.randint(1, max_seq_len)
            
            # Generate random sequence
            sequence = [random.randint(0, vocab_size-1) for _ in range(seq_len)]
            
            # Pad sequence to max_seq_len
            padded_seq = sequence + [pad_token] * (max_seq_len - seq_len)
            
            # Generate random suffix index (must be less than actual sequence length)
            suffix_idx = random.randint(0, seq_len - 1)
            
            self.sequences.append(padded_seq)
            self.sequence_lengths.append(seq_len)
            self.suffix_indices.append(suffix_idx)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.long),
            'length': self.sequence_lengths[idx],
            'suffix_idx': self.suffix_indices[idx]
        }

def collate_sequences(batch):
    """
    Collate function for the data loader.
    
    Args:
        batch: List of dictionaries containing 'sequence', 'length', and 'suffix_idx'
    
    Returns:
        Dictionary with batched tensors
    """
    sequences = torch.stack([item['sequence'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    suffix_indices = torch.tensor([item['suffix_idx'] for item in batch])
    
    return {
        'sequences': sequences,
        'lengths': lengths,
        'suffix_indices': suffix_indices
    }

def get_dummy_loader(num_samples, vocab_size, max_seq_len, pad_token, 
                    batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for dummy sequence data.
    
    Args:
        num_samples (int): Number of sequences to generate
        vocab_size (int): Maximum integer value in sequences (exclusive)
        max_seq_len (int): Maximum sequence length
        pad_token (int): Token to use for padding
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dummy sequences
    """
    dataset = DummySequenceDataset(num_samples, vocab_size, max_seq_len, pad_token)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sequences
    )

# Example usage:
if __name__ == "__main__":
    # Create a dummy loader
    loader = get_dummy_loader(
        num_samples=1000,
        vocab_size=100,
        max_seq_len=20,
        pad_token=0,
        batch_size=4
    )
    
    # Get a batch
    batch = next(iter(loader))
    
    # Print batch information
    print("Sequences shape:", batch['sequences'].shape)
    print("Lengths:", batch['lengths'])
    print("Suffix indices:", batch['suffix_indices'])
    
    # Print first sequence in batch
    print("\nFirst sequence in batch:")
    print("Sequence:", batch['sequences'][0])
    print("Length:", batch['lengths'][0])
    print("Suffix index:", batch['suffix_indices'][0])