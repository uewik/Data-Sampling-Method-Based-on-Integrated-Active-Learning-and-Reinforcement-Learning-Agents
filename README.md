# Bandit-Based Active Learning Framework

This project implements and evaluates a bandit-based active learning framework, inspired by the paper "Ensemble Active Learning by Contextual Bandits for AI Incubation in Manufacturing." The framework incorporates exploration- and exploitation-oriented agents and applies the Exp4.P core function to combine their decisions for sample acquisition.

## 🎯 Overview

The framework combines multiple active learning strategies using multi-armed bandit algorithms to intelligently select the most informative samples for labeling. It features six different agents that use various acquisition strategies, from density-based exploration to uncertainty-based exploitation.

### Key Features

- **Multi-Agent Active Learning**: Six specialized agents with different acquisition strategies
- **Bandit-Based Fusion**: EXP4.P algorithm for combining agent decisions
- **Adaptive Thresholds**: Dynamic threshold adjustment based on performance feedback
- **Comprehensive Evaluation**: Cross-validation with multiple metrics (F1, accuracy, precision, recall)
- **PyTorch Integration**: Neural network classifier with fallback support

## 📊 Dataset

The framework is demonstrated on the **Spambase dataset**:
- 4,601 email samples with 57 numerical features
- Binary classification task (spam vs. not spam)
- Automatically balanced using downsampling
- Features standardized using StandardScaler

## 🤖 Agent Architecture

### Exploration Agents (AL Agents)

1. **ALAgent_X_MMd_B**: Max-Min Distance Agent
   - Uses local density estimation with nearest neighbor distances
   - Maintains sliding window of recent samples
   - Probabilistic selection based on distance ratios

2. **ALAgent_X_hD_B**: High-Dimensional Density Agent
   - High-dimensional density estimation
   - Counts samples beyond density threshold
   - Adaptive to dimensionality challenges

3. **ALAgent_X_M_B**: Margin-Based Agent
   - Combines margin-based uncertainty with density
   - Dynamic margin threshold adjustment
   - Adaptive parameter `s` for threshold updates

### Exploitation Agents (RL Agents)

4. **RAL_B**: Basic Reinforced Active Learning
   - Uncertainty-based selection with ε-greedy exploration
   - Committee-based decision making
   - Fixed threshold strategy

5. **RAL_B_EXP4P**: Enhanced RAL with Adaptive Thresholds
   - Reward-driven threshold adaptation
   - Exponential penalty function: θ_{t+1} = min(θ_t * (1 + η * (1 - 2^{(r_t / ρ^-)})), 1)
   - Performance-based learning

6. **Exp4P_EN_SWAP**: EXP4.P Ensemble with EWMA
   - Combines decisions from exploration agents (AL1-3)
   - Uses EXP4.P algorithm for weight updates
   - EWMA (Exponentially Weighted Moving Average) for stability
   - Swap mechanism for agent selection

## 🔧 Configuration

The framework uses a centralized configuration system:

```python
class Config:
    # Data settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Agent settings
    DENSITY_QUANTILE = 50      # Density threshold percentile
    MARGIN_QUANTILE = 15       # Margin threshold percentile
    RL_CONF_THRESHOLD = 0.6    # Confidence threshold for RL agents
    EWMA_MEAN = 0.5            # EWMA smoothing parameter
    EXP4P_ROUNDS = 10          # EXP4.P update rounds
    EXPLORATION_RATE = 0.15    # ε-greedy exploration rate
    
    # Training settings
    EPOCHS = 100               # Maximum training epochs
    PATIENCE = 10              # Early stopping patience
    CV_FOLDS = 3               # Cross-validation folds
    BATCH_SIZE = 64            # Training batch size
```

## 🚀 Usage

### Basic Usage

```python
from agent_module import run_experiments

# Run the complete experiment suite
results = run_experiments()
```

### Custom Configuration

```python
from agent_module import Config, train_and_eval, agents

# Customize configuration
config = Config()
config.EPOCHS = 150
config.EXPLORATION_RATE = 0.2

# Train with custom settings
metrics = train_and_eval(X_train, y_train, config)
```

### Individual Agent Usage

```python
from agent_module import ALAgent_X_MMd_B, RAL_B_EXP4P

# Initialize agents
density_agent = ALAgent_X_MMd_B(
    DIM=57, 
    window_size=50, 
    density_threshold=0.5
)

rl_agent = RAL_B_EXP4P(
    threshold_uncertainty=0.6,
    eta=1.0,
    reward=1,
    penalty=-1,
    mode='const'
)

# Get agent decisions
decision = density_agent.get_agent_decision(sample, certainty, margin)
```

## 🏗️ Architecture Details

### MLP Classifier

- **Input Layer**: 57 features (spambase dataset)
- **Hidden Layers**: Configurable (default: 128 → 64 neurons)
- **Output Layer**: Single sigmoid unit for binary classification
- **Optimization**: Adam optimizer with early stopping
- **Hyperparameter Search**: Grid search with cross-validation

### EXP4.P Fusion Algorithm

The framework implements a sophisticated fusion mechanism:

1. **Agent Decision Collection**: Each agent provides binary decisions
2. **Weight Initialization**: Equal weights for all agents
3. **Exploration Integration**: ε-greedy exploration overlay
4. **Reward Calculation**: Performance-based reward assignment
5. **Weight Updates**: Exponential weight adjustment based on rewards
6. **Final Decision**: Weighted majority voting

### Reward Function

```python
def reward_fn(y_true, y_pred, selected):
    wrong = (y_true != y_pred)
    return (wrong & selected)*2.0 + (~wrong & selected)*(-1.0)
```

- **High Reward (+2.0)**: Correctly selecting mislabeled samples
- **Low Penalty (-1.0)**: Selecting correctly labeled samples

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **Sample Efficiency**: Number of samples used vs. performance

## 🔍 Experimental Results

The experiments compare different agent combinations:

- **Baseline**: Full dataset training
- **Agent_2**: First 2 agents (exploration-focused)
- **Agent_4**: First 4 agents (exploration + basic RL)
- **Agent_6**: All agents (full ensemble)

Results format:
```
Agent        | Samples | Acc    | F1     | Prec   | Rec
Baseline     | 2296    | 0.9283 | 0.9201 | 0.9018 | 0.9392
Agent_2      | 1834    | 0.9152 | 0.9089 | 0.8876 | 0.9312
Agent_4      | 1627    | 0.9087 | 0.9012 | 0.8832 | 0.9201
Agent_6      | 1456    | 0.8998 | 0.8934 | 0.8745 | 0.9132
```

## 🛠️ Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- **torch**: Neural network implementation and training
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities and metrics
- **scipy**: Scientific computing and statistics
- **typing_extensions**: Type hints support

### Optional Dependencies

The framework includes fallback implementations for environments without PyTorch, though full functionality requires all dependencies.

## 📁 File Structure

```
workplace/
├── agent_module.py      # Main implementation
├── requirements.txt     # Python dependencies
├── spambase.data       # Dataset file
└── README.md           # This file
```

## 🔬 Technical Implementation

### Key Algorithms

1. **Density Estimation**: Euclidean distance-based local density
2. **Margin Calculation**: Uncertainty quantification from model predictions
3. **EXP4.P**: Multi-armed bandit algorithm for agent fusion
4. **EWMA**: Exponentially weighted moving averages for stability
5. **Early Stopping**: Validation-based training termination

### Performance Optimizations

- **Vectorized Operations**: NumPy-based efficient computations
- **Sliding Windows**: Memory-efficient sample management
- **Batch Processing**: GPU-accelerated training when available
- **Hyperparameter Caching**: Efficient grid search implementation

## 🎓 Academic Context

This implementation is based on research in:
- **Active Learning**: Intelligent sample selection strategies
- **Multi-Armed Bandits**: Exploration vs. exploitation trade-offs
- **Ensemble Methods**: Combining multiple learning strategies
- **Manufacturing AI**: Real-world industrial applications

### Citation

If you use this framework in your research, please cite the original paper:
"Ensemble Active Learning by Contextual Bandits for AI Incubation in Manufacturing"

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional active learning strategies
- New bandit algorithms
- Performance optimizations
- Dataset compatibility
- Visualization tools

## 📄 License

This project is open source. Please check with the original paper authors for specific licensing terms.

## 🐛 Troubleshooting

### Common Issues

1. **PyTorch Installation**: The framework includes fallback modes for environments without PyTorch
2. **Memory Usage**: Adjust `BATCH_SIZE` and `window_size` for memory constraints
3. **Dataset Path**: Ensure `spambase.data` is in the same directory as the script
4. **Random Seeds**: Set `RANDOM_STATE` for reproducible results

### Performance Tips

- Use GPU acceleration when available
- Tune hyperparameters for your specific dataset
- Adjust agent thresholds based on data characteristics
- Monitor memory usage with large datasets

---

**Note**: This framework is designed for research and educational purposes. For production use, consider additional validation and optimization based on your specific requirements.

