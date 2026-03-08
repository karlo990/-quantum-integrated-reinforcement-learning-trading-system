"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUASAR PLASTICITY PRESERVATION MODULE                                       ║
║  Based on: Sokar et al. 2023, Haarnoja et al. 2018, Dohare et al. 2024      ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module implements research-backed solutions to feature collapse and 
plasticity loss in deep reinforcement learning.

REFERENCES:
-----------
[1] Sokar, G. et al. (2023). "The Dormant Neuron Phenomenon in Deep RL" - ICML
[2] Haarnoja, T. et al. (2018). "Soft Actor-Critic Algorithms and Applications"
[3] Dohare, S. et al. (2024). "Loss of plasticity in deep continual learning" - Nature
[4] Lyle, C. et al. (2023). "Understanding Plasticity in Neural Networks" - ICML
[5] Nikishin, E. et al. (2023). "Deep RL with Plasticity Injection"

ROOT CAUSES ADDRESSED:
----------------------
1. Dormant Neuron Phenomenon: Neurons become inactive, reducing network capacity
2. Feature Rank Collapse: Internal representations lose diversity
3. Fixed Entropy: Static exploration coefficient fails to adapt
4. Plasticity Loss: Network loses ability to fit new data over time

INTEGRATION:
------------
Add to your quasar_main4.py:
    from quasar_plasticity_module import PlasticityPreserver, AutoEntropyTuner, ReDo
    
    # In your training loop:
    plasticity = PlasticityPreserver(model, device)
    entropy_tuner = AutoEntropyTuner(action_dim=2, device=device)
    redo = ReDo(model, tau=0.1, reset_interval=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import math


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ReDo: RECYCLING DORMANT NEURONS (Sokar et al., ICML 2023)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Key insight: Neurons become dormant (activations → 0) during RL training,
# reducing network expressivity. Solution: detect and recycle them.
#
# Dormancy score: s_i = E[|h_i(x)|] / (1/H * Σ E[|h_k(x)|])
# If s_i < τ, neuron i is τ-dormant
# ═══════════════════════════════════════════════════════════════════════════════

class ReDo(nn.Module):
    """
    Recycling Dormant Neurons (ReDo) - Sokar et al., ICML 2023
    
    Periodically detects dormant neurons and reinitializes them:
    - Incoming weights: reinitialized from original distribution
    - Outgoing weights: zeroed (preserves network output initially)
    - Adam moments: reset for reinitialized neurons (critical!)
    
    Args:
        model: The neural network to monitor
        tau: Dormancy threshold (default 0.1, neurons with score < τ are dormant)
        reset_interval: Steps between dormancy checks
        device: torch device
    """
    
    def __init__(self, model: nn.Module, tau: float = 0.1, 
                 reset_interval: int = 1000, device=None):
        super().__init__()
        self.model = model
        self.tau = tau
        self.reset_interval = reset_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track activations for dormancy calculation
        self.activation_sums: Dict[str, torch.Tensor] = {}
        self.activation_counts: Dict[str, int] = {}
        self.hooks = []
        self.step_count = 0
        
        # Statistics
        self.dormant_history = deque(maxlen=100)
        self.recycled_total = 0
        
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to track activations"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._activation_hook(n, out)
                )
                self.hooks.append(hook)
                
    def _activation_hook(self, name: str, output: torch.Tensor):
        """Accumulate activation statistics"""
        with torch.no_grad():
            try:
                # Handle different tensor dimensions safely
                if output.dim() == 4:
                    # Conv: (batch, channels, H, W) -> mean over batch, H, W
                    act = output.abs().mean(dim=(0, 2, 3))
                elif output.dim() == 3:
                    # 3D: (batch, seq, features) -> mean over batch, seq
                    act = output.abs().mean(dim=(0, 1))
                elif output.dim() == 2:
                    # Linear: (batch, features) -> mean over batch
                    act = output.abs().mean(dim=0)
                elif output.dim() == 1:
                    # 1D: just use as-is
                    act = output.abs()
                else:
                    return  # Skip unsupported dimensions
                
                if name not in self.activation_sums:
                    self.activation_sums[name] = torch.zeros_like(act)
                    self.activation_counts[name] = 0
                
                self.activation_sums[name] += act
                self.activation_counts[name] += 1
            except Exception:
                pass  # Skip on any error - don't crash the forward pass
            
    def compute_dormancy_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute dormancy score for each neuron in each layer.
        
        Score = neuron_activation / layer_mean_activation
        Lower score = more dormant
        """
        scores = {}
        for name, act_sum in self.activation_sums.items():
            count = self.activation_counts[name]
            if count > 0:
                mean_act = act_sum / count
                layer_mean = mean_act.mean() + 1e-8
                scores[name] = mean_act / layer_mean
        return scores
    
    def get_dormant_neurons(self) -> Dict[str, torch.Tensor]:
        """Get indices of dormant neurons (score < tau) per layer"""
        scores = self.compute_dormancy_scores()
        dormant = {}
        for name, score in scores.items():
            dormant_mask = score < self.tau
            if dormant_mask.any():
                dormant[name] = torch.where(dormant_mask)[0]
        return dormant
    
    def recycle_dormant_neurons(self, optimizer: torch.optim.Optimizer) -> int:
        """
        Recycle (reinitialize) dormant neurons.
        
        Critical: Must also reset Adam momentum for recycled neurons!
        
        Returns:
            Number of neurons recycled
        """
        dormant = self.get_dormant_neurons()
        total_recycled = 0
        
        for name, module in self.model.named_modules():
            if name not in dormant:
                continue
                
            dormant_indices = dormant[name]
            n_dormant = len(dormant_indices)
            
            if n_dormant == 0:
                continue
            
            if isinstance(module, nn.Linear):
                # Reinitialize incoming weights (rows)
                with torch.no_grad():
                    # Xavier/Kaiming initialization for incoming
                    fan_in = module.weight.size(1)
                    std = 1.0 / math.sqrt(fan_in)
                    module.weight[dormant_indices].normal_(0, std)
                    
                    if module.bias is not None:
                        module.bias[dormant_indices].zero_()
                
                # Zero outgoing weights (in next layer) - find connected layer
                # This preserves network output initially
                self._zero_outgoing_weights(name, dormant_indices)
                
                # Reset Adam moments for these parameters
                self._reset_optimizer_state(optimizer, module, dormant_indices)
                
                total_recycled += n_dormant
        
        # Reset activation tracking
        self.activation_sums.clear()
        self.activation_counts.clear()
        
        self.recycled_total += total_recycled
        self.dormant_history.append(total_recycled)
        
        return total_recycled
    
    def _zero_outgoing_weights(self, layer_name: str, indices: torch.Tensor):
        """Zero the outgoing weights from recycled neurons in the next layer"""
        found_current = False
        for name, module in self.model.named_modules():
            if name == layer_name:
                found_current = True
                continue
            if found_current and isinstance(module, nn.Linear):
                # This is the next linear layer
                with torch.no_grad():
                    module.weight[:, indices] = 0
                break
    
    def _reset_optimizer_state(self, optimizer: torch.optim.Optimizer, 
                               module: nn.Module, indices: torch.Tensor):
        """
        Reset Adam momentum/variance for recycled neurons.
        
        CRITICAL: Without this, Adam will immediately push recycled neurons
        back into dormancy using old momentum!
        """
        for param in [module.weight, module.bias]:
            if param is None:
                continue
            for group in optimizer.param_groups:
                if param in [p for p in group['params']]:
                    state = optimizer.state.get(param, {})
                    if 'exp_avg' in state:
                        state['exp_avg'][indices] = 0
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'][indices] = 0
                        
    def step(self, optimizer: torch.optim.Optimizer) -> Optional[int]:
        """
        Call this after each training step.
        Returns number of recycled neurons if recycling occurred.
        """
        self.step_count += 1
        
        if self.step_count % self.reset_interval == 0:
            n_recycled = self.recycle_dormant_neurons(optimizer)
            if n_recycled > 0:
                print(f"🔄 ReDo: Recycled {n_recycled} dormant neurons "
                      f"(total: {self.recycled_total})")
            return n_recycled
        return None
    
    def get_dormancy_ratio(self) -> float:
        """Get fraction of neurons that are currently dormant"""
        dormant = self.get_dormant_neurons()
        total_dormant = sum(len(d) for d in dormant.values())
        
        total_neurons = sum(
            m.weight.size(0) for m in self.model.modules() 
            if isinstance(m, nn.Linear)
        )
        
        return total_dormant / max(total_neurons, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AUTOMATIC ENTROPY TUNING (Haarnoja et al., 2018 - SAC)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key insight: Fixed entropy coefficient fails because optimal entropy varies
# across states and training phases. Solution: constrained optimization.
#
# Instead of: max_π E[r + α*H(π)]  with fixed α
# We solve:   max_π E[r]  subject to  E[H(π)] ≥ H_target
#
# This naturally increases α when policy is too deterministic,
# and decreases α when policy is exploring enough.
# ═══════════════════════════════════════════════════════════════════════════════

class AutoEntropyTuner(nn.Module):
    """
    Automatic Entropy Temperature Tuning from SAC (Haarnoja et al., 2018)
    
    Learns the entropy coefficient α to maintain target entropy level.
    
    α_loss = -α * (log π(a|s) + H_target)
    
    When entropy < target: α increases → more exploration reward
    When entropy > target: α decreases → focus on task reward
    
    Args:
        action_dim: Number of actions (for discrete) or action dimensions
        target_entropy: Target entropy level (default: -action_dim for continuous,
                        0.5 * log(action_dim) for discrete)
        initial_alpha: Starting value for α
        lr: Learning rate for α optimization
        device: torch device
    """
    
    def __init__(self, action_dim: int, target_entropy: float = None,
                 initial_alpha: float = 0.2, lr: float = 3e-4, 
                 device=None, discrete: bool = True):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discrete = discrete
        
        # Target entropy: for discrete actions, use fraction of max entropy
        if target_entropy is None:
            if discrete:
                # Target = 50% of maximum entropy (log(action_dim))
                self.target_entropy = 0.5 * np.log(action_dim)
            else:
                # For continuous: -action_dim (heuristic from SAC paper)
                self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
        
        # Learnable log(α) - optimize in log space for stability
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(initial_alpha), dtype=torch.float32, device=self.device)
        )
        
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Statistics
        self.alpha_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)
        
    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient"""
        return self.log_alpha.exp()
    
    def compute_entropy(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distribution.
        
        For discrete: H = -Σ p(a) log p(a)
        """
        # Clamp for numerical stability
        probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
    
    def update(self, action_probs: torch.Tensor) -> Tuple[float, float]:
        """
        Update α based on current policy entropy.
        
        Args:
            action_probs: (batch, action_dim) action probabilities
            
        Returns:
            (alpha_loss, current_alpha)
        """
        with torch.no_grad():
            entropy = self.compute_entropy(action_probs).mean()
            self.entropy_history.append(entropy.item())
        
        # α loss: increase α when entropy is below target
        # L(α) = E[-α * (log π(a|s) + H_target)]
        # = α * (H_target - H(π))  [since -log π ≈ H for the sampled action]
        alpha_loss = self.alpha * (self.target_entropy - entropy.detach())
        
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        
        current_alpha = self.alpha.item()
        self.alpha_history.append(current_alpha)
        
        return alpha_loss.item(), current_alpha
    
    def get_entropy_bonus(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus for actor loss.
        
        actor_loss = -Q(s,a) - α * H(π(·|s))
        """
        entropy = self.compute_entropy(action_probs)
        return self.alpha.detach() * entropy
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        return {
            'alpha': self.alpha.item(),
            'target_entropy': self.target_entropy,
            'mean_entropy': np.mean(self.entropy_history) if self.entropy_history else 0,
            'entropy_gap': np.mean(self.entropy_history) - self.target_entropy if self.entropy_history else 0
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTINUAL BACKPROPAGATION (Dohare et al., Nature 2024)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key insight: Standard networks lose plasticity over time. Solution: 
# reinitialize a tiny proportion of least-used units on EACH step.
#
# This is like neurogenesis - continuously injecting fresh capacity.
# ═══════════════════════════════════════════════════════════════════════════════

class ContinualBackprop:
    """
    Continual Backpropagation (Dohare et al., Nature 2024)
    
    On each step, reinitialize a small fraction of neurons with 
    lowest utility (contribution to output).
    
    Utility = |outgoing_weights| * recent_activation_magnitude
    
    Args:
        model: Neural network
        replacement_rate: Fraction of neurons to replace per step (e.g., 0.001)
        maturity_threshold: Steps before a neuron can be replaced
        device: torch device
    """
    
    def __init__(self, model: nn.Module, replacement_rate: float = 0.001,
                 maturity_threshold: int = 1000, device=None):
        self.model = model
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track neuron age (steps since last reinitialization)
        self.neuron_ages: Dict[str, torch.Tensor] = {}
        self.activation_magnitudes: Dict[str, torch.Tensor] = {}
        
        self._initialize_tracking()
        
    def _initialize_tracking(self):
        """Initialize age and activation tracking for all neurons"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                n_neurons = module.weight.size(0)
                self.neuron_ages[name] = torch.zeros(n_neurons, device=self.device)
                self.activation_magnitudes[name] = torch.ones(n_neurons, device=self.device)
    
    def compute_utility(self, name: str, module: nn.Linear) -> torch.Tensor:
        """
        Compute utility score for each neuron.
        
        Utility = ||outgoing_weights|| * activation_magnitude
        """
        # Find outgoing weight magnitude (if this isn't the last layer)
        outgoing_mag = torch.ones(module.weight.size(0), device=self.device)
        
        found = False
        for n, m in self.model.named_modules():
            if found and isinstance(m, nn.Linear):
                # This is the next layer - get column norms
                outgoing_mag = m.weight.abs().sum(dim=0)[:module.weight.size(0)]
                break
            if n == name:
                found = True
        
        activation_mag = self.activation_magnitudes.get(
            name, torch.ones(module.weight.size(0), device=self.device)
        )
        
        return outgoing_mag * activation_mag
    
    def step(self, optimizer: torch.optim.Optimizer) -> int:
        """
        Perform continual backprop step: replace lowest-utility mature neurons.
        
        Returns:
            Number of neurons replaced
        """
        total_replaced = 0
        
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in self.neuron_ages:
                continue
                
            n_neurons = module.weight.size(0)
            ages = self.neuron_ages[name]
            
            # Increment ages
            ages += 1
            
            # Find mature neurons (old enough to be replaced)
            mature_mask = ages >= self.maturity_threshold
            if not mature_mask.any():
                continue
            
            # Compute utility for mature neurons
            utility = self.compute_utility(name, module)
            utility[~mature_mask] = float('inf')  # Don't replace immature neurons
            
            # Number to replace
            n_replace = max(1, int(n_neurons * self.replacement_rate))
            n_replace = min(n_replace, mature_mask.sum().item())
            
            if n_replace == 0:
                continue
            
            # Get indices of lowest-utility neurons
            _, indices = torch.topk(utility, n_replace, largest=False)
            
            # Reinitialize
            with torch.no_grad():
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in)
                module.weight[indices].normal_(0, std)
                if module.bias is not None:
                    module.bias[indices].zero_()
            
            # Reset ages for replaced neurons
            ages[indices] = 0
            
            # Reset optimizer state
            self._reset_optimizer_state(optimizer, module, indices)
            
            total_replaced += n_replace
        
        return total_replaced
    
    def _reset_optimizer_state(self, optimizer, module, indices):
        """Reset Adam state for replaced neurons"""
        for param in [module.weight, module.bias]:
            if param is None:
                continue
            state = optimizer.state.get(param, {})
            if 'exp_avg' in state:
                state['exp_avg'][indices] = 0
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'][indices] = 0
    
    def update_activations(self, name: str, activations: torch.Tensor):
        """Update activation magnitude tracking (call from forward hook)"""
        with torch.no_grad():
            mag = activations.abs().mean(dim=0)
            if name in self.activation_magnitudes:
                # Exponential moving average
                self.activation_magnitudes[name] = (
                    0.99 * self.activation_magnitudes[name] + 0.01 * mag
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE RANK PRESERVATION (Lyle et al., 2023; Kumar et al., 2020)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key insight: Feature rank collapse (representations become low-rank)
# indicates loss of expressivity. Solution: regularize to maintain rank.
#
# Effective rank = exp(entropy of singular values)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureRankRegularizer:
    """
    Feature Rank Regularization to prevent representation collapse.
    
    Monitors effective rank of layer activations and adds regularization
    loss when rank drops below threshold.
    
    Effective rank = exp(H(σ/||σ||_1)) where σ are singular values
    
    Based on: Lyle et al. 2023, Kumar et al. 2020
    """
    
    def __init__(self, model: nn.Module, min_rank_ratio: float = 0.5,
                 reg_weight: float = 0.01, device=None):
        self.model = model
        self.min_rank_ratio = min_rank_ratio  # Minimum fraction of max rank
        self.reg_weight = reg_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.activation_buffer: Dict[str, List[torch.Tensor]] = {}
        self.rank_history: Dict[str, deque] = {}
        
    def compute_effective_rank(self, activations: torch.Tensor) -> float:
        """
        Compute effective rank of activation matrix.
        
        effective_rank = exp(entropy of normalized singular values)
        """
        if activations.dim() > 2:
            activations = activations.flatten(start_dim=1)
        
        # SVD
        try:
            _, s, _ = torch.svd(activations, compute_uv=False)
        except:
            return activations.size(1)  # Return max rank on failure
        
        # Normalize singular values to form probability distribution
        s_norm = s / (s.sum() + 1e-8)
        
        # Shannon entropy
        entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum()
        
        # Effective rank = exp(entropy)
        return torch.exp(entropy).item()
    
    def compute_rank_loss(self, activations: torch.Tensor, 
                          layer_name: str) -> torch.Tensor:
        """
        Compute regularization loss to maintain feature rank.
        
        Loss = max(0, min_rank - current_rank) / max_rank
        """
        if activations.dim() > 2:
            activations = activations.flatten(start_dim=1)
        
        max_rank = min(activations.size(0), activations.size(1))
        min_rank = self.min_rank_ratio * max_rank
        
        current_rank = self.compute_effective_rank(activations)
        
        # Track history
        if layer_name not in self.rank_history:
            self.rank_history[layer_name] = deque(maxlen=100)
        self.rank_history[layer_name].append(current_rank)
        
        # Loss if rank is too low
        if current_rank < min_rank:
            # Encourage higher rank through covariance regularization
            # Penalize correlation between features
            centered = activations - activations.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / (activations.size(0) - 1)
            
            # Off-diagonal elements (correlations) should be small
            off_diag = cov - torch.diag(torch.diag(cov))
            rank_loss = self.reg_weight * off_diag.pow(2).mean()
            
            return rank_loss
        
        return torch.tensor(0.0, device=self.device)
    
    def get_rank_stats(self) -> Dict[str, float]:
        """Get current rank statistics per layer"""
        stats = {}
        for name, history in self.rank_history.items():
            if history:
                stats[f'{name}_effective_rank'] = np.mean(history)
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 5. UNIFIED PLASTICITY PRESERVER (Combines all methods)
# ═══════════════════════════════════════════════════════════════════════════════

class PlasticityPreserver:
    """
    Unified plasticity preservation combining multiple research-backed methods.
    
    Integrates:
    1. ReDo (dormant neuron recycling)
    2. Automatic entropy tuning
    3. Feature rank regularization
    4. Continual backpropagation
    
    Usage:
        preserver = PlasticityPreserver(model, action_dim=2, device=device)
        
        # In training loop:
        loss = critic_loss + actor_loss
        loss += preserver.get_regularization_loss(activations)
        
        # After optimizer step:
        preserver.step(optimizer, action_probs)
    """
    
    def __init__(self, model: nn.Module, action_dim: int = 2,
                 device=None, config: dict = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        
        # Default config
        cfg = {
            'enable_redo': True,
            'redo_tau': 0.1,
            'redo_interval': 1000,
            'enable_auto_entropy': True,
            'target_entropy_ratio': 0.5,  # Fraction of max entropy
            'enable_rank_reg': True,
            'min_rank_ratio': 0.5,
            'rank_reg_weight': 0.01,
            'enable_continual_backprop': False,  # More aggressive
            'replacement_rate': 0.001,
        }
        if config:
            cfg.update(config)
        self.config = cfg
        
        # Initialize components
        if cfg['enable_redo']:
            self.redo = ReDo(model, tau=cfg['redo_tau'], 
                           reset_interval=cfg['redo_interval'], device=device)
        else:
            self.redo = None
            
        if cfg['enable_auto_entropy']:
            target_entropy = cfg['target_entropy_ratio'] * np.log(action_dim)
            self.entropy_tuner = AutoEntropyTuner(
                action_dim=action_dim, target_entropy=target_entropy, device=device
            )
        else:
            self.entropy_tuner = None
            
        if cfg['enable_rank_reg']:
            self.rank_reg = FeatureRankRegularizer(
                model, min_rank_ratio=cfg['min_rank_ratio'],
                reg_weight=cfg['rank_reg_weight'], device=device
            )
        else:
            self.rank_reg = None
            
        if cfg['enable_continual_backprop']:
            self.continual_bp = ContinualBackprop(
                model, replacement_rate=cfg['replacement_rate'], device=device
            )
        else:
            self.continual_bp = None
        
        self.step_count = 0
        
    def get_entropy_bonus(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Get entropy bonus for actor loss with auto-tuned coefficient"""
        if self.entropy_tuner is not None:
            return self.entropy_tuner.get_entropy_bonus(action_probs)
        return torch.tensor(0.0, device=self.device)
    
    def get_alpha(self) -> float:
        """Get current entropy coefficient"""
        if self.entropy_tuner is not None:
            return self.entropy_tuner.alpha.item()
        return 0.1  # Default
    
    def get_regularization_loss(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get combined regularization loss from all components"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        if self.rank_reg is not None:
            for name, act in activations.items():
                total_loss += self.rank_reg.compute_rank_loss(act, name)
        
        return total_loss
    
    def step(self, optimizer: torch.optim.Optimizer, 
             action_probs: torch.Tensor = None) -> Dict[str, any]:
        """
        Call after each training step.
        
        Args:
            optimizer: The optimizer being used
            action_probs: Current action probabilities (for entropy tuning)
            
        Returns:
            Dictionary of metrics
        """
        self.step_count += 1
        metrics = {}
        
        # ReDo: recycle dormant neurons
        if self.redo is not None:
            n_recycled = self.redo.step(optimizer)
            if n_recycled:
                metrics['neurons_recycled'] = n_recycled
            metrics['dormancy_ratio'] = self.redo.get_dormancy_ratio()
        
        # Auto entropy tuning
        if self.entropy_tuner is not None and action_probs is not None:
            alpha_loss, alpha = self.entropy_tuner.update(action_probs)
            metrics['alpha'] = alpha
            metrics['alpha_loss'] = alpha_loss
            metrics.update(self.entropy_tuner.get_stats())
        
        # Continual backprop
        if self.continual_bp is not None:
            n_replaced = self.continual_bp.step(optimizer)
            metrics['neurons_replaced'] = n_replaced
        
        # Rank stats
        if self.rank_reg is not None:
            metrics.update(self.rank_reg.get_rank_stats())
        
        return metrics
    
    def get_diagnostics(self) -> str:
        """Get human-readable diagnostics"""
        lines = ["=" * 60, "PLASTICITY DIAGNOSTICS", "=" * 60]
        
        if self.redo:
            ratio = self.redo.get_dormancy_ratio()
            lines.append(f"Dormancy ratio: {ratio:.2%} {'⚠️ HIGH' if ratio > 0.3 else '✅'}")
            lines.append(f"Total neurons recycled: {self.redo.recycled_total}")
        
        if self.entropy_tuner:
            stats = self.entropy_tuner.get_stats()
            lines.append(f"Entropy α: {stats['alpha']:.4f}")
            lines.append(f"Mean entropy: {stats['mean_entropy']:.4f} "
                        f"(target: {stats['target_entropy']:.4f})")
            gap = stats['entropy_gap']
            if gap < -0.1:
                lines.append(f"⚠️ Entropy below target by {-gap:.4f} - increasing exploration")
            elif gap > 0.1:
                lines.append(f"✅ Entropy above target by {gap:.4f}")
        
        if self.rank_reg:
            stats = self.rank_reg.get_rank_stats()
            for name, rank in stats.items():
                lines.append(f"{name}: {rank:.1f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_with_quasar_training():
    """
    Example of how to integrate this module with QUASAR training loop.
    
    Add this to your _train_on_batch method:
    """
    
    example_code = '''
    # ═══════════════════════════════════════════════════════════════════
    # In your __init__ or setup:
    # ═══════════════════════════════════════════════════════════════════
    from quasar_plasticity_module import PlasticityPreserver
    
    self.plasticity = PlasticityPreserver(
        model=self.model,  # Your neural network
        action_dim=2,      # BUY/SELL
        device=self.device,
        config={
            'enable_redo': True,
            'redo_tau': 0.1,           # Dormancy threshold
            'redo_interval': 1000,      # Check every 1000 steps
            'enable_auto_entropy': True,
            'target_entropy_ratio': 0.5,  # 50% of max entropy
            'enable_rank_reg': True,
            'min_rank_ratio': 0.3,     # Minimum 30% of max rank
        }
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # In your training loop (replace fixed entropy bonus):
    # ═══════════════════════════════════════════════════════════════════
    
    # OLD (fixed entropy):
    # entropy_coef = self._get_entropy_coef()  # Decaying coefficient
    # entropy_bonus = -entropy_coef * mean_entropy
    
    # NEW (auto-tuned):
    entropy_bonus = self.plasticity.get_entropy_bonus(action_probs)
    
    # Add to actor loss
    actor_loss = actor_loss - entropy_bonus.mean()  # Maximize entropy
    
    # ═══════════════════════════════════════════════════════════════════
    # After optimizer.step():
    # ═══════════════════════════════════════════════════════════════════
    metrics = self.plasticity.step(self.optimizer, action_probs)
    
    # Log metrics
    if self.step_count % 1000 == 0:
        print(self.plasticity.get_diagnostics())
    '''
    
    return example_code


if __name__ == "__main__":
    print("QUASAR Plasticity Module")
    print("=" * 60)
    print("Based on research from:")
    print("  [1] Sokar et al. 2023 - ReDo (ICML)")
    print("  [2] Haarnoja et al. 2018 - SAC Auto Entropy")
    print("  [3] Dohare et al. 2024 - Continual Backprop (Nature)")
    print("  [4] Lyle et al. 2023 - Feature Rank (ICML)")
    print("=" * 60)
    print("\nIntegration example:")
    print(integrate_with_quasar_training())
