from sbi.neural_nets.estimators.base import ConditionalEstimator
import torch.nn as nn
from sbi.neural_nets.factory import build_maf, build_nsf


# inference = NLE(
#     prior=prior,
#     density_estimator=build_custom_nle,
#     device=training_device
# )
#
# def build_custom_nle(batch_theta, batch_x):
#                 # Get CNN feature dimensions
#                 with torch.no_grad():
#                     single_x_features = embedding_net(batch_x[:1])
#                 print(f"CNN output shape (single): {single_x_features.shape}")
#                 print(f"Theta shape: {batch_theta.shape}")
#                 print(f"Original X shape: {batch_x.shape}")
#
#                 return SBICompatibleWrapper(
#                     embedding_net,
#                     batch_x.shape,          # Raw x shape
#                     batch_theta.shape,      # θ shape  
#                     single_x_features.shape # CNN feature shape
#                 )

class SBICompatibleWrapper(ConditionalEstimator):
    def __init__(self, cnn, x_shape, theta_shape, cnn_feature_shape):
        super().__init__(x_shape[1:], theta_shape[1:])
        self.cnn = cnn
        self.flow = None
        # Tell SBI what it expects to hear
        self._input_shape = x_shape[1:]      # Raw HEALPix shape for SBI
        self._condition_shape = theta_shape[1:]  # θ shape for SBI
        self.cnn_feature_shape = cnn_feature_shape[1:]
        print(
            f"SBI-compatible wrapper: input_shape={self._input_shape} ",
            f"condition_shape={self._condition_shape}"
        )
        
        # Z-scoring parameters (will be computed on first batch)
        self.x_raw_mean = None
        self.x_raw_std = None
        self.condition_mean = None
        self.condition_std = None

    def _compute_zscore_params(self, x_raw, condition):
        """Compute z-scoring parameters from raw inputs (before CNN)."""
        if self.x_raw_mean is None:
            # STRUCTURED z-scoring for raw x (before CNN embedding)
            # Flatten all dimensions except batch, compute single mean/std
            x_raw_flat = x_raw.flatten(start_dim=1)  # (batch, all_pixels)
            self.x_raw_mean = x_raw_flat.mean()  # Single scalar
            self.x_raw_std = x_raw_flat.std() + 1e-8  # Single scalar
            
            # INDEPENDENT z-scoring for condition (per parameter dimension)
            self.condition_mean = condition.mean(dim=0, keepdim=True)  # (1, n_params)
            self.condition_std = condition.std(dim=0, keepdim=True) + 1e-8  # (1, n_params)
            
            print(f"Z-score params computed - x_raw: mean={self.x_raw_mean:.4f}, std={self.x_raw_std:.4f}")
            print(f"Z-score params computed - condition: mean={self.condition_mean.shape}, std={self.condition_std.shape}")

    def _zscore_tensors(self, x_raw, condition):
        """Apply z-scoring manually to match SBI's behavior."""
        # STRUCTURED z-scoring for raw x - same normalization for all pixels
        x_raw_normed = (x_raw - self.x_raw_mean) / self.x_raw_std
        
        # INDEPENDENT z-scoring for condition - per-parameter normalization
        condition_normed = (condition - self.condition_mean) / self.condition_std
        
        return x_raw_normed, condition_normed

    def _build_flow_if_needed(self, x_raw, condition):
        if self.flow is None:
            print(
                f"Building internal flow with raw x: {x_raw.shape}, "
                f"condition: {condition.shape}"
            )

            # Compute z-scoring parameters
            self._compute_zscore_params(x_raw, condition)
            
            # Apply z-scoring to raw inputs for flow construction
            x_raw_normed, condition_normed = self._zscore_tensors(x_raw, condition)
            
            # Apply CNN to normalized raw data
            x_features_normed = self.cnn(x_raw_normed)

            self.flow = build_nsf(
                batch_x=x_features_normed,      # CNN features
                batch_y=condition_normed, # θ parameters
                z_score_x=None, # 'structured',
                z_score_y=None, # 'independent',
                # hidden_features=64,   # More reasonable for 20D space
                # num_transforms=4,     # Fewer transforms
                # num_blocks=3,         # More blocks per transform
                dropout_probability=0.0,
                use_batch_norm=False
            )
            print("Internal flow built successfully!")

            # Move the entire flow to the correct device
            self.flow = self.flow.to('cuda')
            print(f"Flow built and moved to device: {'cuda'}")

    def log_prob(self, x, condition):
        # Debug: print shapes and statistics
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
            
        if self._debug_counter <= 5:  # Only print first 5 calls
            print(f"log_prob call {self._debug_counter}: x.shape={x.shape}, condition.shape={condition.shape}")
        
        # Initialize z-scoring parameters if needed
        self._build_flow_if_needed(x, condition)
        
        # Apply z-scoring to raw x BEFORE CNN (like original SBI)
        x_raw_normed, condition_normed = self._zscore_tensors(x, condition)
        
        if self._debug_counter <= 5:
            print(f"  Raw x stats: mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"  Normed x stats: mean={x_raw_normed.mean():.4f}, std={x_raw_normed.std():.4f}")
            print(f"  Condition stats (global): mean={condition.mean():.4f}, std={condition.std():.4f}")
            print(f"  Condition stats (per-dim): mean={condition.mean(dim=0)}, std={condition.std(dim=0)}")
            print(f"  Condition_normed stats (per-dim): mean={condition_normed.mean(dim=0)}, std={condition_normed.std(dim=0)}")
        
        # Now apply CNN to normalized raw data
        x_features = self.cnn(x_raw_normed)
        
        if self._debug_counter <= 5:
            print(f"  CNN features stats: mean={x_features.mean():.4f}, std={x_features.std():.4f}")

        # Add sample dimension: (batch_dim, feature_dim) -> (1, batch_dim, feature_dim)
        x_features_with_sample_dim = x_features.unsqueeze(0)

        result = self.flow.log_prob(x_features_with_sample_dim, condition_normed)
        
        if self._debug_counter <= 5:
            print(f"  Flow result stats: mean={result.mean():.4f}, std={result.std():.4f}, range=[{result.min():.2f}, {result.max():.2f}]")
            
        return result
    
    def loss(self, x, condition):
        return -self.log_prob(x, condition)
    
    def sample(self, num_samples, context):
        # This samples CNN features, not raw x
        if self.flow is None:
            raise RuntimeError("Flow not built yet")
        return self.flow.sample(num_samples, context)
        
    @property
    def input_shape(self):
        return self._input_shape  # Raw HEALPix shape
        
    @property  
    def condition_shape(self):
        return self._condition_shape  # θ shape
