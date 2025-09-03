"""
Data utilities for Flickr30k dataset handling.
This module provides functions for downloading, loading, and preprocessing the Flickr30k dataset.
"""

import os
import json
import requests
import zipfile
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
from collections import Counter
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Flickr30kDataset(Dataset):
    """Custom Dataset class for Flickr30k dataset."""
    
    def __init__(self, 
                 images_dir: str,
                 captions_file: str,
                 vocab_file: str,
                 transform=None,
                 max_length: int = 50,
                 split: str = 'train'):
        """
        Initialize the Flickr30k dataset.
        
        Args:
            images_dir: Directory containing images
            captions_file: Path to captions file (CSV or JSON)
            vocab_file: Path to vocabulary pickle file
            transform: Image transformations
            max_length: Maximum caption length
            split: Dataset split ('train', 'val', 'test')
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_length = max_length
        self.split = split
        
        # Load vocabulary
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.word2idx = self.vocab['word2idx']
        self.idx2word = self.vocab['idx2word']
        self.vocab_size = len(self.word2idx)
        
        # Load captions data
        self.data = self._load_captions(captions_file)
        
        # Special tokens
        self.start_token = self.word2idx.get('<start>', 1)
        self.end_token = self.word2idx.get('<end>', 2)
        self.pad_token = self.word2idx.get('<pad>', 0)
        self.unk_token = self.word2idx.get('<unk>', 3)
    
    def _load_captions(self, captions_file: str) -> List[Dict]:
        """Load captions from file."""
        if captions_file.endswith('.csv'):
            df = pd.read_csv(captions_file)
            # Assuming columns: 'image_name', 'caption'
            data = []
            for _, row in df.iterrows():
                data.append({
                    'image_name': row['image_name'],
                    'caption': row['caption']
                })
            return data
        elif captions_file.endswith('.json'):
            with open(captions_file, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("Captions file must be CSV or JSON")
    
    def _tokenize_caption(self, caption: str) -> List[int]:
        """Tokenize caption and convert to indices."""
        # Basic preprocessing
        caption = caption.lower().strip()
        caption = re.sub(r'[^\w\s]', '', caption)  # Remove punctuation
        words = caption.split()
        
        # Convert to indices
        indices = [self.start_token]
        for word in words:
            indices.append(self.word2idx.get(word, self.unk_token))
        indices.append(self.end_token)
        
        # Pad or truncate
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.pad_token] * (self.max_length - len(indices)))
        
        return indices
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = self.images_dir / item['image_name']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        # Process caption
        caption_indices = self._tokenize_caption(item['caption'])
        
        return {
            'image': image,
            'caption': torch.tensor(caption_indices, dtype=torch.long),
            'image_name': item['image_name'],
            'caption_text': item['caption']
        }


def download_flickr30k(data_dir: str = "data/flickr30k") -> None:
    """
    Download Flickr30k dataset.
    Note: This is a placeholder function. In practice, you would need to:
    1. Sign up for Kaggle API
    2. Download the dataset using kaggle API or manually
    3. Extract the files to the appropriate directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("To download Flickr30k dataset:")
    print("1. Visit: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
    print("2. Download the dataset manually or use Kaggle API:")
    print("   kaggle datasets download -d hsankesara/flickr-image-dataset")
    print(f"3. Extract to: {data_dir}")
    print("\nAlternatively, you can use the original Flickr30k dataset from:")
    print("https://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/")


def preprocess_captions(captions_file: str, output_dir: str, min_word_freq: int = 5) -> Dict:
    """
    Preprocess captions and build vocabulary.
    
    Args:
        captions_file: Path to raw captions file
        output_dir: Directory to save processed files
        min_word_freq: Minimum word frequency for vocabulary
    
    Returns:
        Dictionary containing vocabulary information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading captions...")
    if captions_file.endswith('.csv'):
        df = pd.read_csv(captions_file)
        captions = df['caption'].tolist()
    else:
        raise ValueError("Currently only CSV format is supported")
    
    print("Building vocabulary...")
    word_counts = Counter()
    
    for caption in captions:
        # Basic preprocessing
        caption = caption.lower().strip()
        caption = re.sub(r'[^\w\s]', '', caption)  # Remove punctuation
        words = caption.split()
        word_counts.update(words)
    
    # Build vocabulary with special tokens
    vocab_words = ['<pad>', '<start>', '<end>', '<unk>']
    
    # Add words that meet frequency threshold
    for word, count in word_counts.items():
        if count >= min_word_freq:
            vocab_words.append(word)
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    vocab = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(vocab_words),
        'word_counts': dict(word_counts),
        'min_word_freq': min_word_freq
    }
    
    # Save vocabulary
    vocab_file = output_dir / 'vocab.pkl'
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"Vocabulary saved to: {vocab_file}")
    print(f"Vocabulary size: {len(vocab_words)}")
    print(f"Total unique words: {len(word_counts)}")
    
    return vocab


def create_data_splits(captions_file: str, 
                      output_dir: str,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1) -> None:
    """
    Split the dataset into train/validation/test sets.
    
    Args:
        captions_file: Path to captions file
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(captions_file)
    
    # Get unique images to ensure images don't appear in multiple splits
    unique_images = df['image_name'].unique()
    n_images = len(unique_images)
    
    # Shuffle images
    import numpy as np
    np.random.seed(42)  # For reproducibility
    shuffled_images = np.random.permutation(unique_images)
    
    # Calculate split indices
    train_end = int(n_images * train_ratio)
    val_end = int(n_images * (train_ratio + val_ratio))
    
    # Split images
    train_images = set(shuffled_images[:train_end])
    val_images = set(shuffled_images[train_end:val_end])
    test_images = set(shuffled_images[val_end:])
    
    # Create split dataframes
    train_df = df[df['image_name'].isin(train_images)]
    val_df = df[df['image_name'].isin(val_images)]
    test_df = df[df['image_name'].isin(test_images)]
    
    # Save splits
    train_df.to_csv(output_dir / 'train_captions.csv', index=False)
    val_df.to_csv(output_dir / 'val_captions.csv', index=False)
    test_df.to_csv(output_dir / 'test_captions.csv', index=False)
    
    print(f"Data splits saved to: {output_dir}")
    print(f"Train: {len(train_df)} captions from {len(train_images)} images")
    print(f"Val: {len(val_df)} captions from {len(val_images)} images")
    print(f"Test: {len(test_df)} captions from {len(test_images)} images")


def get_image_transforms(is_training: bool = True, 
                        image_size: int = 224) -> transforms.Compose:
    """
    Get image transformation pipeline.
    
    Args:
        is_training: Whether for training (includes augmentation)
        image_size: Target image size
    
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(data_dir: str,
                      vocab_file: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      image_size: int = 224) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing the data
        vocab_file: Path to vocabulary file
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size for transforms
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    
    # Create datasets
    train_dataset = Flickr30kDataset(
        images_dir=images_dir,
        captions_file=data_dir / 'train_captions.csv',
        vocab_file=vocab_file,
        transform=get_image_transforms(is_training=True, image_size=image_size),
        split='train'
    )
    
    val_dataset = Flickr30kDataset(
        images_dir=images_dir,
        captions_file=data_dir / 'val_captions.csv',
        vocab_file=vocab_file,
        transform=get_image_transforms(is_training=False, image_size=image_size),
        split='val'
    )
    
    test_dataset = Flickr30kDataset(
        images_dir=images_dir,
        captions_file=data_dir / 'test_captions.csv',
        vocab_file=vocab_file,
        transform=get_image_transforms(is_training=False, image_size=image_size),
        split='test'
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Data utilities for Flickr30k dataset")
    download_flickr30k()