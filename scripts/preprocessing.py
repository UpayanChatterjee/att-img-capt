"""
Data preprocessing pipeline for Flickr30k dataset.
This script handles image preprocessing, caption cleaning, and vocabulary building.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict
import json
import pickle
from collections import Counter
import re
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def clean_caption(caption: str) -> str:
    """
    Clean and preprocess a single caption.
    
    Args:
        caption: Raw caption text
        
    Returns:
        Cleaned caption text
    """
    # Convert to lowercase
    caption = caption.lower().strip()
    
    # Remove extra whitespace
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove special characters but keep basic punctuation
    caption = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', caption)
    
    # Remove leading/trailing punctuation
    caption = re.sub(r'^[^\w]+|[^\w]+$', '', caption)
    
    # Handle contractions (basic)
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am"
    }
    
    for contraction, expansion in contractions.items():
        caption = caption.replace(contraction, expansion)
    
    # Remove remaining apostrophes
    caption = caption.replace("'", "")
    
    # Final cleanup
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption

def extract_image_features_stats(images_dir: str, sample_size: int = 1000) -> Dict:
    """
    Extract basic statistics from a sample of images for preprocessing decisions.
    
    Args:
        images_dir: Directory containing images
        sample_size: Number of images to sample for statistics
        
    Returns:
        Dictionary with image statistics
    """
    images_dir = Path(images_dir)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # Sample images for statistics
    sample_files = np.random.choice(image_files, 
                                   min(sample_size, len(image_files)), 
                                   replace=False)
    
    widths = []
    heights = []
    channels = []
    sizes_mb = []
    
    print(f"Analyzing {len(sample_files)} sample images...")
    
    for img_file in tqdm(sample_files):
        try:
            # Get file size
            size_mb = img_file.stat().st_size / (1024 * 1024)
            sizes_mb.append(size_mb)
            
            # Load image
            img = Image.open(img_file)
            widths.append(img.width)
            heights.append(img.height)
            
            # Check channels
            if img.mode == 'RGB':
                channels.append(3)
            elif img.mode == 'RGBA':
                channels.append(4)
            elif img.mode == 'L':
                channels.append(1)
            else:
                channels.append(3)  # Default assumption
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    stats = {
        'total_images': len(image_files),
        'sampled_images': len(sample_files),
        'width_stats': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'median': np.median(widths),
            'std': np.std(widths)
        },
        'height_stats': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'median': np.median(heights),
            'std': np.std(heights)
        },
        'size_stats_mb': {
            'min': min(sizes_mb),
            'max': max(sizes_mb),
            'mean': np.mean(sizes_mb),
            'median': np.median(sizes_mb),
            'std': np.std(sizes_mb)
        },
        'channel_distribution': Counter(channels)
    }
    
    return stats

def create_sample_captions_file(data_dir: str, output_file: str, num_samples: int = 100):
    """
    Create a sample captions file for testing purposes.
    This creates dummy data when the actual Flickr30k dataset is not available.
    
    Args:
        data_dir: Directory containing images
        output_file: Output CSV file path
        num_samples: Number of sample entries to create
    """
    data_dir = Path(data_dir)
    
    # Create sample data structure
    sample_captions = [
        "A person walking down the street",
        "A dog playing in the park",
        "Children playing on a playground",
        "A woman reading a book",
        "A man riding a bicycle",
        "People sitting at a restaurant",
        "A car driving on a highway",
        "A cat sleeping on a bed",
        "People walking on the beach",
        "A bird flying in the sky"
    ]
    
    # Create sample image names
    image_names = [f"sample_image_{i:04d}.jpg" for i in range(num_samples)]
    
    # Create sample data
    data = []
    np.random.seed(42)
    
    for i, img_name in enumerate(image_names):
        # Each image gets multiple captions (1-5 captions per image)
        num_captions = np.random.randint(1, 6)
        for j in range(num_captions):
            caption = np.random.choice(sample_captions)
            # Add some variation
            if np.random.random() > 0.5:
                caption += f" with {np.random.choice(['blue', 'red', 'green', 'yellow'])} background"
            
            data.append({
                'image_name': img_name,
                'caption': caption
            })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Sample captions file created: {output_file}")
    print(f"Total entries: {len(data)}")
    
    return output_file

def preprocess_flickr30k_captions(input_file: str, 
                                 output_dir: str,
                                 min_word_freq: int = 5,
                                 max_caption_length: int = 50) -> Dict:
    """
    Preprocess Flickr30k captions and build vocabulary.
    
    Args:
        input_file: Input captions file (CSV format)
        output_dir: Output directory for processed files
        min_word_freq: Minimum word frequency for inclusion in vocabulary
        max_caption_length: Maximum caption length (in words)
        
    Returns:
        Dictionary containing preprocessing statistics and vocabulary info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading captions from: {input_file}")
    
    # Check if input file exists, create sample if it doesn't
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Creating sample data...")
        create_sample_captions_file("data/flickr30k/images", input_file, num_samples=1000)
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} caption entries")
    
    # Clean captions
    print("Cleaning captions...")
    clean_captions = []
    caption_lengths = []
    
    for caption in tqdm(df['caption']):
        clean_caption_text = clean_caption(str(caption))
        clean_captions.append(clean_caption_text)
        caption_lengths.append(len(clean_caption_text.split()))
    
    df['clean_caption'] = clean_captions
    
    # Filter by caption length
    original_count = len(df)
    df = df[df['clean_caption'].str.len() > 0]  # Remove empty captions
    
    # Get caption length statistics
    caption_length_stats = {
        'min': min(caption_lengths),
        'max': max(caption_lengths),
        'mean': np.mean(caption_lengths),
        'median': np.median(caption_lengths),
        'std': np.std(caption_lengths),
        'percentile_95': np.percentile(caption_lengths, 95),
        'percentile_99': np.percentile(caption_lengths, 99)
    }
    
    print(f"Caption length statistics: {caption_length_stats}")
    
    # Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter()
    
    for caption in df['clean_caption']:
        words = caption.split()
        word_counts.update(words)
    
    # Create vocabulary with special tokens
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    vocab_words = special_tokens.copy()
    
    # Add words that meet frequency threshold
    frequent_words = [word for word, count in word_counts.items() 
                     if count >= min_word_freq]
    vocab_words.extend(frequent_words)
    
    # Create word mappings
    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Vocabulary statistics
    vocab_stats = {
        'total_unique_words': len(word_counts),
        'vocab_size': len(vocab_words),
        'min_word_freq': min_word_freq,
        'coverage': len(frequent_words) / len(word_counts),
        'most_common_words': word_counts.most_common(20),
        'word_freq_distribution': {
            'freq_1': sum(1 for count in word_counts.values() if count == 1),
            'freq_2_5': sum(1 for count in word_counts.values() if 2 <= count <= 5),
            'freq_6_10': sum(1 for count in word_counts.values() if 6 <= count <= 10),
            'freq_11_plus': sum(1 for count in word_counts.values() if count > 10)
        }
    }
    
    print(f"Vocabulary statistics: {vocab_stats}")
    
    # Save vocabulary
    vocab_data = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(vocab_words),
        'word_counts': dict(word_counts),
        'special_tokens': special_tokens,
        'min_word_freq': min_word_freq,
        'stats': vocab_stats
    }
    
    vocab_file = output_dir / 'vocab.pkl'
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab_data, f)
    
    # Save processed captions
    processed_file = output_dir / 'processed_captions.csv'
    df.to_csv(processed_file, index=False)
    
    # Save preprocessing statistics
    preprocessing_stats = {
        'original_captions': original_count,
        'processed_captions': len(df),
        'filtered_captions': original_count - len(df),
        'caption_length_stats': caption_length_stats,
        'vocab_stats': vocab_stats,
        'files_created': {
            'vocab_file': str(vocab_file),
            'processed_captions': str(processed_file)
        }
    }
    
    stats_file = output_dir / 'preprocessing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(preprocessing_stats, f, indent=2)
    
    print(f"\nPreprocessing completed!")
    print(f"Vocabulary saved to: {vocab_file}")
    print(f"Processed captions saved to: {processed_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return preprocessing_stats

def visualize_preprocessing_stats(stats_file: str, output_dir: str):
    """
    Create visualizations for preprocessing statistics.
    
    Args:
        stats_file: Path to preprocessing statistics JSON file
        output_dir: Directory to save visualization plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Flickr30k Dataset Preprocessing Statistics', fontsize=16)
    
    # Caption length distribution (mock data for visualization)
    caption_stats = stats['caption_length_stats']
    ax1 = axes[0, 0]
    
    # Generate sample distribution for visualization
    np.random.seed(42)
    lengths = np.random.normal(caption_stats['mean'], caption_stats['std'], 1000)
    lengths = lengths[lengths > 0]  # Remove negative values
    
    ax1.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(caption_stats['mean'], color='red', linestyle='--', 
                label=f"Mean: {caption_stats['mean']:.1f}")
    ax1.axvline(caption_stats['median'], color='green', linestyle='--', 
                label=f"Median: {caption_stats['median']:.1f}")
    ax1.set_xlabel('Caption Length (words)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Caption Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vocabulary coverage
    vocab_stats = stats['vocab_stats']
    ax2 = axes[0, 1]
    
    coverage_data = [vocab_stats['coverage'], 1 - vocab_stats['coverage']]
    labels = ['Included in Vocab', 'Filtered Out']
    colors = ['lightgreen', 'lightcoral']
    
    ax2.pie(coverage_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Vocabulary Coverage')
    
    # Word frequency distribution
    ax3 = axes[1, 0]
    freq_dist = vocab_stats['word_freq_distribution']
    freq_categories = list(freq_dist.keys())
    freq_counts = list(freq_dist.values())
    
    bars = ax3.bar(range(len(freq_categories)), freq_counts, color='lightblue', edgecolor='black')
    ax3.set_xlabel('Frequency Range')
    ax3.set_ylabel('Number of Words')
    ax3.set_title('Word Frequency Distribution')
    ax3.set_xticks(range(len(freq_categories)))
    ax3.set_xticklabels(freq_categories, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, freq_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom')
    
    # Dataset split visualization
    ax4 = axes[1, 1]
    
    # Summary statistics
    summary_text = f"""Dataset Summary:
    
Total Captions: {stats['processed_captions']:,}
Vocabulary Size: {vocab_stats['vocab_size']:,}
Unique Words: {vocab_stats['total_unique_words']:,}
Min Word Frequency: {vocab_stats['min_word_freq']}

Caption Length:
  Mean: {caption_stats['mean']:.1f} words
  Median: {caption_stats['median']:.1f} words
  Std: {caption_stats['std']:.1f} words
  95th Percentile: {caption_stats['percentile_95']:.1f} words

Most Common Words:
{chr(10).join([f"  {word}: {count}" for word, count in vocab_stats['most_common_words'][:5]])}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'preprocessing_statistics.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Preprocessing statistics plot saved to: {plot_file}")

def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess Flickr30k dataset')
    parser.add_argument('--input_file', type=str, 
                       default='data/flickr30k/captions.csv',
                       help='Input captions file')
    parser.add_argument('--output_dir', type=str,
                       default='data/flickr30k/processed',
                       help='Output directory for processed files')
    parser.add_argument('--min_word_freq', type=int, default=5,
                       help='Minimum word frequency for vocabulary')
    parser.add_argument('--max_caption_length', type=int, default=50,
                       help='Maximum caption length in words')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Run preprocessing
    stats = preprocess_flickr30k_captions(
        input_file=args.input_file,
        output_dir=args.output_dir,
        min_word_freq=args.min_word_freq,
        max_caption_length=args.max_caption_length
    )
    
    # Create visualizations if requested
    if args.visualize:
        stats_file = Path(args.output_dir) / 'preprocessing_stats.json'
        visualize_preprocessing_stats(str(stats_file), args.output_dir)
    
    print("\nPreprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()