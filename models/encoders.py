"""
Encoder models for image captioning.
This module implements ResNet50 and Vision Transformer (ViT) encoders for feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import ViTModel, ViTConfig
from typing import Dict, Any, Optional
import math


class ResNet50Encoder(nn.Module):
    """
    ResNet50-based encoder for image feature extraction.
    Uses pre-trained ResNet50 and removes the classification head.
    """
    
    def __init__(self, 
                 embed_dim: int = 512,
                 pretrained: bool = True,
                 fine_tune: bool = True,
                 dropout: float = 0.1):
        """
        Initialize ResNet50 encoder.
        
        Args:
            embed_dim: Dimension of output embeddings
            pretrained: Whether to use pre-trained weights
            fine_tune: Whether to allow fine-tuning of ResNet parameters
            dropout: Dropout probability
        """
        super(ResNet50Encoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.fine_tune = fine_tune
        
        # Load pre-trained ResNet50
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Remove the final fully connected layer and average pooling
        modules = list(self.resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        # ResNet50 outputs 2048 features from the last conv layer
        self.resnet_dim = 2048
        
        # Adaptive pooling to ensure consistent spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 7x7 spatial dimension
        
        # Project ResNet features to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.resnet_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize projection layers
        self._initialize_weights()
        
        # Freeze ResNet parameters if not fine-tuning
        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def _initialize_weights(self):
        """Initialize projection layer weights."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ResNet50 encoder.
        
        Args:
            images: Input images [batch_size, 3, H, W]
            
        Returns:
            Dictionary containing:
                - features: Encoded features [batch_size, spatial_dim, embed_dim]
                - spatial_features: Original spatial features [batch_size, 2048, 7, 7]
                - global_features: Global average pooled features [batch_size, embed_dim]
        """
        batch_size = images.size(0)
        
        # Extract features using ResNet50
        with torch.set_grad_enabled(self.fine_tune):
            features = self.resnet(images)  # [batch_size, 2048, H/32, W/32]
        
        # Apply adaptive pooling to get consistent spatial dimensions
        spatial_features = self.adaptive_pool(features)  # [batch_size, 2048, 7, 7]
        
        # Reshape for attention: [batch_size, spatial_dim, feature_dim]
        spatial_dim = spatial_features.size(2) * spatial_features.size(3)  # 49
        spatial_features_flat = spatial_features.view(batch_size, self.resnet_dim, spatial_dim)
        spatial_features_flat = spatial_features_flat.permute(0, 2, 1)  # [batch_size, 49, 2048]
        
        # Project to embedding dimension
        projected_features = self.projection(spatial_features_flat)  # [batch_size, 49, embed_dim]
        projected_features = self.layer_norm(projected_features)
        
        # Global features for initialization
        global_features = torch.mean(projected_features, dim=1)  # [batch_size, embed_dim]
        
        return {
            'features': projected_features,  # [batch_size, 49, embed_dim]
            'spatial_features': spatial_features,  # [batch_size, 2048, 7, 7]
            'global_features': global_features  # [batch_size, embed_dim]
        }
    
    def get_attention_weights(self, features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for features given a query.
        This is a simple implementation for visualization purposes.
        
        Args:
            features: Spatial features [batch_size, spatial_dim, embed_dim]
            query: Query vector [batch_size, embed_dim]
            
        Returns:
            Attention weights [batch_size, spatial_dim]
        """
        # Simple dot-product attention
        query = query.unsqueeze(1)  # [batch_size, 1, embed_dim]
        scores = torch.bmm(query, features.transpose(1, 2))  # [batch_size, 1, spatial_dim]
        weights = F.softmax(scores.squeeze(1), dim=1)  # [batch_size, spatial_dim]
        
        return weights


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) based encoder for image feature extraction.
    Uses pre-trained ViT model from HuggingFace Transformers.
    """
    
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 embed_dim: int = 512,
                 fine_tune: bool = True,
                 dropout: float = 0.1,
                 use_cls_token: bool = True):
        """
        Initialize ViT encoder.
        
        Args:
            model_name: HuggingFace model name for ViT
            embed_dim: Dimension of output embeddings
            fine_tune: Whether to allow fine-tuning of ViT parameters
            dropout: Dropout probability
            use_cls_token: Whether to use CLS token for global features
        """
        super(ViTEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.fine_tune = fine_tune
        self.use_cls_token = use_cls_token
        
        # Load pre-trained ViT model
        try:
            self.vit = ViTModel.from_pretrained(model_name)
            self.vit_dim = self.vit.config.hidden_size
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Using default ViT configuration...")
            config = ViTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16
            )
            self.vit = ViTModel(config)
            self.vit_dim = config.hidden_size
        
        # Number of patches (for 224x224 image with 16x16 patches: 14x14 = 196)
        self.num_patches = (self.vit.config.image_size // self.vit.config.patch_size) ** 2
        
        # Project ViT features to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.vit_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize projection layers
        self._initialize_weights()
        
        # Freeze ViT parameters if not fine-tuning
        if not fine_tune:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def _initialize_weights(self):
        """Initialize projection layer weights."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ViT encoder.
        
        Args:
            images: Input images [batch_size, 3, H, W]
            
        Returns:
            Dictionary containing:
                - features: Encoded patch features [batch_size, num_patches, embed_dim]
                - global_features: Global CLS token features [batch_size, embed_dim]
                - attention_weights: Self-attention weights from last layer [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = images.size(0)
        
        # Forward pass through ViT
        with torch.set_grad_enabled(self.fine_tune):
            outputs = self.vit(images, output_attentions=True)
        
        # Get hidden states (includes CLS token + patch tokens)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        if self.use_cls_token:
            # Separate CLS token and patch tokens
            cls_token = hidden_states[:, 0, :]  # [batch_size, hidden_size]
            patch_tokens = hidden_states[:, 1:, :]  # [batch_size, num_patches, hidden_size]
        else:
            # Use mean of patch tokens as global feature
            patch_tokens = hidden_states[:, 1:, :]  # [batch_size, num_patches, hidden_size]
            cls_token = torch.mean(patch_tokens, dim=1)  # [batch_size, hidden_size]
        
        # Project features to embedding dimension
        projected_patches = self.projection(patch_tokens)  # [batch_size, num_patches, embed_dim]
        projected_patches = self.layer_norm(projected_patches)
        
        projected_global = self.projection(cls_token)  # [batch_size, embed_dim]
        projected_global = self.layer_norm(projected_global)
        
        # Get attention weights from the last layer
        attention_weights = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
        
        return {
            'features': projected_patches,  # [batch_size, num_patches, embed_dim]
            'global_features': projected_global,  # [batch_size, embed_dim]
            'attention_weights': attention_weights,  # [batch_size, num_heads, seq_len, seq_len]
            'raw_features': patch_tokens  # [batch_size, num_patches, vit_dim]
        }
    
    def get_patch_attention(self, attention_weights: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract patch-to-patch attention weights for visualization.
        
        Args:
            attention_weights: Attention weights from forward pass
            layer_idx: Which layer's attention to use (-1 for last layer)
            
        Returns:
            Patch attention weights [batch_size, num_patches, num_patches]
        """
        # Average across attention heads
        attn = torch.mean(attention_weights, dim=1)  # [batch_size, seq_len, seq_len]
        
        # Extract patch-to-patch attention (exclude CLS token)
        patch_attention = attn[:, 1:, 1:]  # [batch_size, num_patches, num_patches]
        
        return patch_attention


class EncoderFactory:
    """Factory class for creating encoder models."""
    
    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> nn.Module:
        """
        Create an encoder model based on type.
        
        Args:
            encoder_type: Type of encoder ('resnet50' or 'vit')
            **kwargs: Additional arguments for encoder initialization
            
        Returns:
            Encoder model
        """
        if encoder_type.lower() == 'resnet50':
            return ResNet50Encoder(**kwargs)
        elif encoder_type.lower() in ['vit', 'vision_transformer']:
            return ViTEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    @staticmethod
    def get_encoder_info(encoder_type: str) -> Dict[str, Any]:
        """
        Get information about encoder architecture.
        
        Args:
            encoder_type: Type of encoder
            
        Returns:
            Dictionary with encoder information
        """
        if encoder_type.lower() == 'resnet50':
            return {
                'name': 'ResNet50',
                'type': 'CNN',
                'spatial_features': True,
                'default_spatial_dim': 49,  # 7x7
                'feature_extraction': 'Convolutional layers with residual connections',
                'strengths': ['Strong spatial understanding', 'Robust feature extraction', 'Well-established architecture'],
                'weaknesses': ['Limited long-range dependencies', 'Fixed receptive fields']
            }
        elif encoder_type.lower() in ['vit', 'vision_transformer']:
            return {
                'name': 'Vision Transformer',
                'type': 'Transformer',
                'spatial_features': True,
                'default_spatial_dim': 196,  # 14x14 patches
                'feature_extraction': 'Self-attention over image patches',
                'strengths': ['Global context modeling', 'Flexible attention patterns', 'State-of-the-art performance'],
                'weaknesses': ['Requires large datasets', 'Computationally expensive', 'Less inductive bias']
            }
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


def test_encoders():
    """Test function for encoder models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing encoders on device: {device}")
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Test ResNet50 encoder
    print("\n=== Testing ResNet50 Encoder ===")
    resnet_encoder = ResNet50Encoder(embed_dim=512, pretrained=False).to(device)
    
    with torch.no_grad():
        resnet_outputs = resnet_encoder(images)
    
    print(f"ResNet50 features shape: {resnet_outputs['features'].shape}")
    print(f"ResNet50 global features shape: {resnet_outputs['global_features'].shape}")
    print(f"ResNet50 spatial features shape: {resnet_outputs['spatial_features'].shape}")
    
    # Test ViT encoder
    print("\n=== Testing ViT Encoder ===")
    try:
        vit_encoder = ViTEncoder(
            model_name="google/vit-base-patch16-224",
            embed_dim=512,
            fine_tune=False
        ).to(device)
        
        with torch.no_grad():
            vit_outputs = vit_encoder(images)
        
        print(f"ViT features shape: {vit_outputs['features'].shape}")
        print(f"ViT global features shape: {vit_outputs['global_features'].shape}")
        print(f"ViT attention weights shape: {vit_outputs['attention_weights'].shape}")
        
    except Exception as e:
        print(f"ViT test failed: {e}")
        print("This might be due to missing HuggingFace model. Using local config...")
        
        vit_encoder = ViTEncoder(
            model_name="google/vit-base-patch16-224",
            embed_dim=512,
            fine_tune=False
        ).to(device)
        
        with torch.no_grad():
            vit_outputs = vit_encoder(images)
        
        print(f"ViT features shape: {vit_outputs['features'].shape}")
        print(f"ViT global features shape: {vit_outputs['global_features'].shape}")
    
    # Test encoder factory
    print("\n=== Testing Encoder Factory ===")
    factory_resnet = EncoderFactory.create_encoder('resnet50', embed_dim=256, pretrained=False)
    factory_vit = EncoderFactory.create_encoder('vit', embed_dim=256, fine_tune=False)
    
    print("Factory created encoders successfully!")
    
    # Print encoder info
    print("\n=== Encoder Information ===")
    for encoder_type in ['resnet50', 'vit']:
        info = EncoderFactory.get_encoder_info(encoder_type)
        print(f"\n{info['name']}:")
        print(f"  Type: {info['type']}")
        print(f"  Spatial dimension: {info['default_spatial_dim']}")
        print(f"  Strengths: {', '.join(info['strengths'])}")


if __name__ == "__main__":
    test_encoders()