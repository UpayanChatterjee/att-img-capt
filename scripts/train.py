"""
Training pipeline for image captioning with attention mechanisms.
This module handles the complete training loop, model checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our models
from models.encoders import ResNet50Encoder, ViTEncoder
from models.decoder import AttentionDecoder
from scripts.data_utils import Flickr30kDataset, get_image_transforms


class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model combining encoder and decoder.
    """
    
    def __init__(self,
                 encoder_type: str,
                 vocab_size: int,
                 embed_dim: int = 512,
                 encoder_dim: int = 512,
                 decoder_dim: int = 512,
                 attention_type: str = 'bahdanau',
                 attention_dim: int = 256,
                 dropout: float = 0.5,
                 fine_tune_encoder: bool = True,
                 max_length: int = 50):
        """
        Initialize the complete image captioning model.
        
        Args:
            encoder_type: Type of encoder ('resnet50' or 'vit')
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            encoder_dim: Dimension of encoder features
            decoder_dim: Dimension of decoder LSTM
            attention_type: Type of attention mechanism
            attention_dim: Dimension of attention layer
            dropout: Dropout probability
            fine_tune_encoder: Whether to fine-tune encoder
            max_length: Maximum sequence length
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        
        # Create encoder
        if encoder_type.lower() == 'resnet50':
            self.encoder = ResNet50Encoder(
                embed_dim=encoder_dim,
                pretrained=True,
                fine_tune=fine_tune_encoder,
                dropout=dropout
            )
        elif encoder_type.lower() in ['vit', 'vision_transformer']:
            self.encoder = ViTEncoder(
                embed_dim=encoder_dim,
                fine_tune=fine_tune_encoder,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Create decoder
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_type=attention_type,
            attention_dim=attention_dim,
            dropout=dropout,
            max_length=max_length
        )
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete model.
        
        Args:
            images: Input images [batch_size, 3, H, W]
            captions: Caption tokens [batch_size, max_length]
            
        Returns:
            Dictionary containing outputs
        """
        # Encode images
        encoder_outputs = self.encoder(images)
        encoder_features = encoder_outputs['features']
        
        # Decode captions
        decoder_outputs = self.decoder(encoder_features, captions)
        
        return {
            'outputs': decoder_outputs['outputs'],
            'attention_weights': decoder_outputs['attention_weights'],
            'encoder_features': encoder_features,
            'global_features': encoder_outputs.get('global_features', None)
        }
    
    def generate_caption(self,
                        images: torch.Tensor,
                        start_token: int,
                        end_token: int,
                        max_length: Optional[int] = None,
                        temperature: float = 1.0,
                        deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate captions for images.
        
        Args:
            images: Input images [batch_size, 3, H, W]
            start_token: Start token ID
            end_token: End token ID
            max_length: Maximum generation length
            temperature: Temperature for sampling
            deterministic: Whether to use greedy decoding
            
        Returns:
            Dictionary containing generated sequences and attention weights
        """
        # Encode images
        encoder_outputs = self.encoder(images)
        encoder_features = encoder_outputs['features']
        
        # Generate captions
        generation_outputs = self.decoder.generate(
            encoder_features,
            start_token=start_token,
            end_token=end_token,
            max_length=max_length,
            temperature=temperature,
            deterministic=deterministic
        )
        
        return {
            'sequences': generation_outputs['sequences'],
            'attention_weights': generation_outputs['attention_weights'],
            'encoder_features': encoder_features
        }


class CaptioningTrainer:
    """
    Trainer class for image captioning models.
    """
    
    def __init__(self,
                 model: ImageCaptioningModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 vocab: Dict,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 grad_clip: float = 5.0,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model: Image captioning model
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab: Vocabulary dictionary
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            grad_clip: Gradient clipping value
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.grad_clip = grad_clip
        
        # Create directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and criterion
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Use CrossEntropyLoss with ignore_index for padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['word2idx'].get('<pad>', 0))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # TensorBoard writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(
            self.log_dir / f'{model.encoder_type}_{timestamp}'
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Special tokens
        self.start_token = vocab['word2idx'].get('<start>', 1)
        self.end_token = vocab['word2idx'].get('<end>', 2)
        self.pad_token = vocab['word2idx'].get('<pad>', 0)
        self.unk_token = vocab['word2idx'].get('<unk>', 3)
        
        print(f"Trainer initialized:")
        print(f"  Model: {model.encoder_type}")
        print(f"  Device: {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            captions = batch['caption'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, captions)
            
            # Compute loss
            # Exclude the last token from input and first token from target
            predictions = outputs['outputs'][:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
            targets = captions[:, 1:].contiguous()  # [batch, seq_len-1]
            
            # Reshape for CrossEntropyLoss
            predictions = predictions.view(-1, predictions.size(2))  # [batch*(seq_len-1), vocab_size]
            targets = targets.view(-1)  # [batch*(seq_len-1)]
            
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Generate sample captions periodically
            if self.global_step % 1000 == 0:
                self._log_sample_captions(images[:2], batch['caption_text'][:2])
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = batch['image'].to(self.device)
                captions = batch['caption'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, captions)
                
                # Compute loss
                predictions = outputs['outputs'][:, :-1, :].contiguous()
                targets = captions[:, 1:].contiguous()
                
                predictions = predictions.view(-1, predictions.size(2))
                targets = targets.view(-1)
                
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, self.epoch)
        
        return avg_loss
    
    def _log_sample_captions(self, images: torch.Tensor, ground_truth: List[str]):
        """
        Generate and log sample captions.
        
        Args:
            images: Sample images
            ground_truth: Ground truth captions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate captions
            generation_outputs = self.model.generate_caption(
                images,
                start_token=self.start_token,
                end_token=self.end_token,
                max_length=20,
                deterministic=True
            )
            
            sequences = generation_outputs['sequences']
            
            # Convert sequences to text
            generated_captions = []
            for seq in sequences:
                caption_words = []
                for token_id in seq:
                    token_id = token_id.item()
                    if token_id == self.end_token:
                        break
                    if token_id not in [self.start_token, self.pad_token]:
                        word = self.vocab['idx2word'].get(token_id, '<unk>')
                        caption_words.append(word)
                
                generated_captions.append(' '.join(caption_words))
            
            # Log samples
            for i, (gt, gen) in enumerate(zip(ground_truth, generated_captions)):
                self.writer.add_text(
                    f'samples/image_{i}',
                    f'Ground Truth: {gt}\nGenerated: {gen}',
                    self.global_step
                )
        
        self.model.train()
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab': self.vocab,
            'model_config': {
                'encoder_type': self.model.encoder_type,
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'encoder_dim': self.model.encoder_dim
            }
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'{self.model.encoder_type}_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model.encoder_type}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch + 1}")
    
    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Print epoch results
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            print(f"  Time: {elapsed_time:.2f}s")
            print("-" * 50)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training and validation curves.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.title(f'Training Curves - {self.model.encoder_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(self.val_losses) + 1
        plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                   label=f'Best Epoch ({best_epoch})')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_model_and_trainer(config: Dict[str, Any],
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           vocab: Dict,
                           device: torch.device) -> Tuple[ImageCaptioningModel, CaptioningTrainer]:
    """
    Create model and trainer from configuration.
    
    Args:
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        vocab: Vocabulary dictionary
        device: Training device
        
    Returns:
        Tuple of (model, trainer)
    """
    # Create model
    model = ImageCaptioningModel(
        encoder_type=config['encoder_type'],
        vocab_size=config['vocab_size'],
        embed_dim=config.get('embed_dim', 512),
        encoder_dim=config.get('encoder_dim', 512),
        decoder_dim=config.get('decoder_dim', 512),
        attention_type=config.get('attention_type', 'bahdanau'),
        attention_dim=config.get('attention_dim', 256),
        dropout=config.get('dropout', 0.5),
        fine_tune_encoder=config.get('fine_tune_encoder', True),
        max_length=config.get('max_length', 50)
    )
    
    # Create trainer
    trainer = CaptioningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=device,
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5),
        grad_clip=config.get('grad_clip', 5.0),
        log_dir=config.get('log_dir', 'logs'),
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints')
    )
    
    return model, trainer


if __name__ == "__main__":
    # Test the training pipeline
    print("Testing training pipeline...")
    
    # This is a basic test - the actual training would be done in the notebook
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create a simple test configuration
    config = {
        'encoder_type': 'resnet50',
        'vocab_size': 1000,
        'embed_dim': 256,
        'encoder_dim': 256,
        'decoder_dim': 256,
        'attention_type': 'bahdanau',
        'max_length': 20
    }
    
    # Create a dummy model to test
    model = ImageCaptioningModel(**config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, config['vocab_size'], (batch_size, config['max_length']))
    
    with torch.no_grad():
        outputs = model(images, captions)
    
    print(f"Forward pass successful!")
    print(f"Output shape: {outputs['outputs'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    
    print("Training pipeline test completed!")