import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from quantum_layers import QuantumLayer, SelfAttentionLayer
import numpy as np

class HybridQuantumCNN(nn.Module):
    """
    Hybrid Quantum-Classical CNN for Cancer Detection
    Architecture: ResNet50 → Quantum Layer → Attention → Classification
    """
    
    def __init__(self, num_classes=2, freeze_backbone=True, quantum_dim=256):
        super(HybridQuantumCNN, self).__init__()
        
        self.num_classes = num_classes
        self.quantum_dim = quantum_dim
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers for transfer learning
        if freeze_backbone:
            self._freeze_backbone_layers()
        
        # Get the feature dimension from ResNet50
        backbone_out_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Remove the original classification layer
        self.backbone.fc = nn.Identity()
        
        # Feature dimensionality reduction
        self.feature_reducer = nn.Sequential(
            nn.Linear(backbone_out_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Quantum processing layer
        self.quantum_layer = QuantumLayer(
            input_dim=512, 
            output_dim=quantum_dim,
            n_qubits=8
        )
        
        # Self-attention mechanism
        self.attention_layer = SelfAttentionLayer(
            input_dim=quantum_dim,
            hidden_dim=128
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(quantum_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _freeze_backbone_layers(self):
        """Freeze early layers of ResNet50 for transfer learning"""
        
        # Freeze all layers except the last two blocks
        layers_to_freeze = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3[:2]  # Freeze first 2 blocks of layer3
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
                
        print("🔒 Frozen early ResNet50 layers for transfer learning")
        
    def _initialize_weights(self):
        """Initialize custom layer weights"""
        
        for module in [self.feature_reducer, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, x, return_attention=False):
        """
        Forward pass through the hybrid model
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits
            attention_weights: (optional) Attention weights for explainability
        """
        
        batch_size = x.size(0)
        
        # Extract features using ResNet50 backbone
        backbone_features = self.backbone(x)  # (batch_size, 2048)
        
        # Reduce feature dimensionality
        reduced_features = self.feature_reducer(backbone_features)  # (batch_size, 512)
        
        # Apply quantum processing
        quantum_features = self.quantum_layer(reduced_features)  # (batch_size, quantum_dim)
        
        # Apply attention mechanism
        attended_features, attention_weights = self.attention_layer(quantum_features)
        
        # Final classification
        logits = self.classifier(attended_features)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
            
    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization
        """
        
        feature_maps = {}
        
        # Backbone features
        backbone_features = self.backbone(x)
        feature_maps['backbone'] = backbone_features
        
        # Reduced features
        reduced_features = self.feature_reducer(backbone_features)
        feature_maps['reduced'] = reduced_features
        
        # Quantum features
        quantum_features = self.quantum_layer(reduced_features)
        feature_maps['quantum'] = quantum_features
        
        # Attended features
        attended_features, attention_weights = self.attention_layer(quantum_features)
        feature_maps['attended'] = attended_features
        feature_maps['attention_weights'] = attention_weights
        
        return feature_maps

class EnsembleQuantumCNN(nn.Module):
    """
    Ensemble of Hybrid Quantum CNNs for improved accuracy
    """
    
    def __init__(self, num_models=3, num_classes=2):
        super(EnsembleQuantumCNN, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            HybridQuantumCNN(num_classes=num_classes, quantum_dim=256 + i*32)
            for i in range(num_models)
        ])
        
    def forward(self, x):
        """Forward pass through ensemble"""
        
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
            
        # Average ensemble predictions
        ensemble_output = torch.stack(outputs).mean(dim=0)
        
        return ensemble_output

def count_parameters(model):
    """Count total and trainable parameters"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

# Test the hybrid model
if __name__ == "__main__":
    print("🧪 Testing Hybrid Quantum-CNN Model...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridQuantumCNN(num_classes=2).to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"📊 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    # Test forward pass
    test_input = torch.randn(4, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # Test normal forward pass
        logits = model(test_input)
        print(f"\n✅ Logits Shape: {logits.shape}")
        print(f"✅ Sample Logits: {logits[0]}")
        
        # Test with attention weights
        logits_with_attention, attention_weights = model(test_input, return_attention=True)
        print(f"✅ Attention Weights Shape: {attention_weights.shape}")
        
        # Test feature maps extraction
        feature_maps = model.get_feature_maps(test_input)
        print(f"\n🔍 Feature Map Shapes:")
        for key, features in feature_maps.items():
            print(f"  {key}: {features.shape}")
    
    # Test ensemble model
    print(f"\n🚀 Testing Ensemble Model...")
    ensemble_model = EnsembleQuantumCNN(num_models=3).to(device)
    
    with torch.no_grad():
        ensemble_output = ensemble_model(test_input)
        print(f"✅ Ensemble Output Shape: {ensemble_output.shape}")
        print(f"✅ Sample Ensemble Logits: {ensemble_output[0]}")
    
    print("\n✅ Hybrid Quantum-CNN model working perfectly!")