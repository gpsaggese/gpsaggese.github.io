"""
Stock Signal Generation Environment

This is a complete end-to-end ensemble workflow for contextual signal generation using:
1. Model A-L (Forecaster): LSTM with MC Dropout for uncertainty cones
2. Model B-L (Interpreter): LM Dictionary + LDA for news context
3. Model C (RL Agent): PPO/SAC for signal synthesis in SignalTesterEnv

The agent learns to synthesize forecasts + news context into robust signals
that maximize risk-adjusted returns (Sharpe-based reward).

Architecture:
- State: Uncertainty cones (6 paths: 3 price + 3 vol) + sentiment + topics
- Action: [directional_bias, volatility_bias, duration] in [-1, 1]
- Reward: Multiplicative Sharpe-based reward (direction × volatility match)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

logger = logging.getLogger(__name__)


# ==================== MODEL A-L: LSTM FORECASTER ====================

class UncertaintyForecaster(nn.Module):
    """
    Lightweight LSTM/GRU model that forecasts uncertainty cones for price and volatility.
    
    Uses Monte Carlo Dropout at inference to generate percentile-based forecasts:
    - 10th, 50th, 90th percentiles for price (20-day paths)
    - 10th, 50th, 90th percentiles for volatility (20-day paths)
    
    Input: (batch, 60, num_features) - 60 days of OHLCV + indicators
    Output: (batch, 20, 2) - 20-day forecast of [log_return, volatility]
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, forecast_horizon: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder for forecasting
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),  # MC Dropout layer
            nn.Linear(hidden_dim, 2)  # Output: [log_return, volatility]
        )
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: (batch, seq_len, features) - Input sequence
        
        Returns:
            (batch, forecast_horizon, 2) - Forecasted [log_return, vol]
        """
        # Encode sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state to forecast
        h = h_n[-1]  # (batch, hidden_dim)
        
        # Decode to multi-step forecast
        forecasts = []
        for _ in range(self.forecast_horizon):
            pred = self.decoder(h)  # (batch, 2)
            forecasts.append(pred)
            # Update hidden state for next step (autoregressive)
            h = h + pred[:, :self.hidden_dim] if pred.shape[1] >= self.hidden_dim else h
        
        return torch.stack(forecasts, dim=1)  # (batch, horizon, 2)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate uncertainty cone using Monte Carlo Dropout.
        
        Args:
            x: (batch, seq_len, features) - Input sequence
            n_samples: Number of MC samples (default: 100)
        
        Returns:
            Dictionary with percentile forecasts:
            - 'price_p10', 'price_p50', 'price_p90': 20-day price paths
            - 'vol_p10', 'vol_p50', 'vol_p90': 20-day volatility paths
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)  # (batch, horizon, 2)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_samples, batch, horizon, 2)
        
        # Calculate percentiles across MC samples
        p10 = np.percentile(predictions, 10, axis=0)  # (batch, horizon, 2)
        p50 = np.percentile(predictions, 50, axis=0)  # median
        p90 = np.percentile(predictions, 90, axis=0)
        
        # Split into price and volatility
        result = {
            'price_p10': p10[0, :, 0],  # (horizon,)
            'price_p50': p50[0, :, 0],
            'price_p90': p90[0, :, 0],
            'vol_p10': p10[0, :, 1],
            'vol_p50': p50[0, :, 1],
            'vol_p90': p90[0, :, 1],
        }
        
        return result


def train_uncertainty_forecaster(
    data: pd.DataFrame,
    feature_columns: list,
    lookback_window: int = 60,
    forecast_horizon: int = 20,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    existing_model: Optional[UncertaintyForecaster] = None,
    incremental: bool = False
) -> UncertaintyForecaster:
    """
    Train the LSTM uncertainty forecaster on historical data.
    
    Supports incremental learning for online updates without full retraining.
    
    Args:
        data: DataFrame with OHLCV + indicators
        feature_columns: List of feature column names to use
        lookback_window: Past days to use as input (default: 60)
        forecast_horizon: Future days to predict (default: 20)
        hidden_dim: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        epochs: Training epochs (default: 50)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 1e-3)
        device: Device to train on (default: 'cpu')
        existing_model: Existing model to continue training (default: None)
        incremental: If True, use fewer epochs and lower LR for fine-tuning (default: False)
    
    Returns:
        Trained UncertaintyForecaster model
    """
    # Incremental mode: reduce epochs and use smaller LR for fine-tuning
    if incremental:
        epochs = max(5, epochs // 4)  # 1/4 of original epochs, minimum 5
        learning_rate = learning_rate * 0.1  # 10x smaller LR for fine-tuning
        logger.info(f"Incremental training mode: {epochs} epochs, LR={learning_rate:.6f}")
    else:
        logger.info(f"Full training mode (horizon={forecast_horizon}, lookback={lookback_window})")
    
    # Prepare sequences
    X, y = [], []
    for i in range(lookback_window, len(data) - forecast_horizon):
        # Input: past 60 days
        X.append(data.iloc[i-lookback_window:i][feature_columns].values)
        
        # Target: future 20 days of [log_return, volatility]
        future_prices = data.iloc[i:i+forecast_horizon]['Close'].values
        log_returns = np.log(future_prices[1:] / future_prices[:-1])
        log_returns = np.concatenate([[0], log_returns])  # Pad first day
        
        realized_vol = data.iloc[i:i+forecast_horizon]['Close'].pct_change().rolling(5).std().fillna(0).values
        
        targets = np.stack([log_returns, realized_vol], axis=-1)  # (horizon, 2)
        y.append(targets)
    
    # Validate we have enough data
    if len(X) == 0:
        raise ValueError(
            f"Not enough data to train forecaster. Need at least {lookback_window + forecast_horizon} samples, "
            f"but got {len(data)}. Try reducing lookback_window or forecast_horizon, or use more training data."
        )
    
    X = torch.FloatTensor(np.array(X)).to(device)  # (n_samples, lookback, features)
    y = torch.FloatTensor(np.array(y)).to(device)  # (n_samples, horizon, 2)
    
    logger.info(f"Created {len(X)} training samples for forecaster")
    
    # Use existing model or create new one
    if existing_model is not None:
        logger.info("Continuing training from existing model (incremental learning)")
        model = existing_model.to(device)
    else:
        logger.info("Creating new forecaster model")
        model = UncertaintyForecaster(
            input_dim=len(feature_columns),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Log only at 25%, 50%, 75%, and 100% to reduce verbosity
        if epoch == 0 or (epoch + 1) in [epochs//4, epochs//2, 3*epochs//4, epochs]:
            mode_str = "incremental" if incremental else "full"
            logger.info(f"[{mode_str}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    logger.info(f"✓ Forecaster training completed ({'incremental' if incremental else 'full'} mode)")
    return model


# ==================== MODEL B-L: NEWS INTERPRETER ====================

class NewsInterpreter:
    """
    Lightweight news interpreter using LM Dictionary + LDA.
    
    Provides two types of context:
    1. Sentiment: LM dictionary-based scores (positive, negative, uncertainty, litigious)
    2. Topics: LDA-based topic probabilities (e.g., Fed/Rates, Earnings, etc.)
    
    This is pre-fitted on the corpus and used for fast inference.
    """
    
    def __init__(self, n_topics: int = 15, max_features: int = 500):
        """
        Initialize the news interpreter.
        
        Args:
            n_topics: Number of LDA topics (default: 15)
            max_features: Max TF-IDF features (default: 500, lower for shorter docs)
        """
        self.n_topics = n_topics
        self.max_features = max_features
        
        # TF-IDF vectorizer (optimized for shorter news snippets from Polygon API)
        # Polygon API returns title + description (concise, not full articles)
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=1,  # Changed from 2: allow words appearing in just 1 doc (handles small corpora)
            max_df=0.95,  # Changed from 0.8: be more lenient with common words
            ngram_range=(1, 2),  # Add bigrams to capture phrases like "interest rate"
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # At least 2-letter words
        )
        
        # LDA model
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        
        # LM Dictionary word lists (expanded for Polygon API news snippets)
        # These capture common sentiment in financial news titles and descriptions
        self.lm_positive = set([
            'gain', 'profit', 'growth', 'positive', 'increase', 'strong', 'beat', 'exceed',
            'surge', 'rally', 'bullish', 'outperform', 'upgrade', 'record', 'high', 'success',
            'optimistic', 'improvement', 'boost', 'advance', 'win', 'opportunity', 'recovery'
        ])
        self.lm_negative = set([
            'loss', 'decline', 'negative', 'decrease', 'weak', 'miss', 'fall', 'drop',
            'plunge', 'crash', 'bearish', 'underperform', 'downgrade', 'concern', 'warning',
            'pessimistic', 'disappointing', 'cut', 'retreat', 'struggle', 'pressure', 'slump'
        ])
        self.lm_uncertainty = set([
            'uncertain', 'risk', 'volatile', 'may', 'could', 'might', 'unclear', 'expect',
            'forecast', 'outlook', 'guidance', 'estimates', 'potential', 'possible', 'maybe',
            'anticipate', 'speculate', 'predict', 'likely', 'probability', 'cautious'
        ])
        self.lm_litigious = set([
            'lawsuit', 'litigation', 'claim', 'sue', 'allege', 'fraud', 'investigation',
            'regulatory', 'violation', 'penalty', 'settlement', 'complaint', 'probe', 'charges'
        ])
        
        self.is_fitted = False
        self._fitted_documents = []  # Track documents used in fitting
    
    def fit(self, documents: list, incremental: bool = False):
        """
        Fit the interpreter on a corpus of news documents.
        
        Args:
            documents: List of news text strings (one per day from Polygon API)
                      Each document should contain title + description snippets
            incremental: If True and already fitted, merge with existing documents (default: False)
        """
        # Incremental update: merge new documents with existing ones
        if incremental and self.is_fitted and len(self._fitted_documents) > 0:
            logger.info(f"Incremental fit: merging {len(documents)} new docs with {len(self._fitted_documents)} existing docs")
            # Keep a sliding window of documents to prevent unbounded growth
            max_history = 500  # Keep last 500 documents
            all_documents = self._fitted_documents[-max_history:] + documents
            logger.info(f"Total corpus size: {len(all_documents)} documents")
        else:
            all_documents = documents
            logger.info(f"Full fit on {len(documents)} documents...")
        
        # Clean documents: strip whitespace, ensure minimum length
        valid_documents = []
        for doc in all_documents:
            if doc and isinstance(doc, str):
                cleaned = doc.strip()
                # Accept any document with at least a few characters
                # (Polygon API returns concise snippets, not full articles)
                if len(cleaned) > 5:
                    valid_documents.append(cleaned)
        
        filtered_count = len(documents) - len(valid_documents)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} empty/invalid documents")
        
        # Ensure we have at least some documents
        if len(valid_documents) < 3:
            logger.warning(f"Only {len(valid_documents)} valid documents. Adding neutral padding for stability.")
            # Add neutral padding documents to ensure TF-IDF has something to work with
            padding = [
                "Market conditions remain stable.",
                "Trading activity continues normally.",
                "Investors monitor economic indicators."
            ]
            valid_documents.extend(padding[:max(0, 3 - len(valid_documents))])
        
        logger.info(f"Using {len(valid_documents)} documents for fitting")
        
        # Fit TF-IDF with relaxed parameters (handles shorter Polygon API snippets)
        tfidf_matrix = self.vectorizer.fit_transform(valid_documents)
        
        vocab_size = tfidf_matrix.shape[1]
        logger.info(f"TF-IDF vocabulary: {vocab_size} features (from {len(valid_documents)} documents)")
        
        if vocab_size == 0:
            # This should never happen with min_df=1 and padding, but handle defensively
            raise ValueError(
                "Failed to extract any TF-IDF features. "
                "This indicates all documents are empty or contain only stop words. "
                "Check news data quality."
            )
        
        # Fit LDA on TF-IDF matrix
        self.lda.fit(tfidf_matrix)
        
        self.is_fitted = True
        self._fitted_documents = valid_documents  # Store for incremental updates
        logger.info(f"✓ News interpreter fitted successfully with {vocab_size} features and {self.n_topics} topics")
    
    def get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Extract LM dictionary-based sentiment scores.
        
        Args:
            text: News text for a single day
        
        Returns:
            Dictionary with sentiment scores
        """
        # Tokenize and clean
        words = text.lower().split()
        total_words = max(len(words), 1)
        
        # Count LM categories
        pos_count = sum(1 for w in words if w in self.lm_positive)
        neg_count = sum(1 for w in words if w in self.lm_negative)
        unc_count = sum(1 for w in words if w in self.lm_uncertainty)
        lit_count = sum(1 for w in words if w in self.lm_litigious)
        
        return {
            'sentiment_score': (pos_count - neg_count) / total_words,
            'uncertainty_score': unc_count / total_words,
            'litigious_score': lit_count / total_words
        }
    
    def get_topic_distribution(self, text: str) -> np.ndarray:
        """
        Extract LDA topic distribution.
        
        Args:
            text: News text for a single day (Polygon API title + description)
        
        Returns:
            Array of topic probabilities (n_topics,)
        """
        if not self.is_fitted:
            raise ValueError("NewsInterpreter must be fitted before calling get_topic_distribution()")
        
        # Handle empty text gracefully
        if not text or not text.strip():
            text = "No news available"
        
        # Transform to TF-IDF
        tfidf = self.vectorizer.transform([text])
        
        # Get topic distribution
        topic_dist = self.lda.transform(tfidf)[0]
        
        return topic_dist
    
    def get_full_context(self, text: str) -> np.ndarray:
        """
        Get complete news context vector.
        
        Args:
            text: News text for a single day
        
        Returns:
            Concatenated vector: [sentiment(3), topics(n_topics)]
        """
        sentiment = self.get_sentiment_scores(text)
        topics = self.get_topic_distribution(text)
        
        # Concatenate
        context = np.concatenate([
            [sentiment['sentiment_score'], 
             sentiment['uncertainty_score'], 
             sentiment['litigious_score']],
            topics
        ])
        
        return context


# ==================== MODEL C: SIGNAL TESTER ENVIRONMENT ====================

class SignalTesterEnv(gym.Env):
    """
    RL Environment for learning to generate robust trading signals.
    
    The agent synthesizes uncertainty cones + news context into a signal vector:
    - State: Concatenated [uncertainty_cones(120), sentiment(3), topics(15)] = 138-dim
    - Action: [directional_bias, volatility_bias, duration] in [-1, 1]^3
    - Reward: Multiplicative Sharpe-based reward
    
    The agent learns patterns like:
    "Wide uncertainty + high uncertainty sentiment + Fed topic → short duration bearish signal"
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        data: pd.DataFrame,
        uncertainty_forecasts: Dict[int, Dict[str, np.ndarray]],  # Pre-computed forecasts
        news_contexts: Dict[int, np.ndarray],  # Pre-computed news vectors
        episode_length: int = 30,
        min_duration: int = 5,
        max_duration: int = 20,
    ):
        """
        Initialize the Signal Tester Environment.
        
        Args:
            data: DataFrame with OHLCV + indicators (for calculating actual returns)
            uncertainty_forecasts: Dict mapping timestep -> uncertainty cone dict
            news_contexts: Dict mapping timestep -> news context vector
            episode_length: Steps per episode (default: 30)
            min_duration: Minimum signal duration in days (default: 5)
            max_duration: Maximum signal duration in days (default: 20)
        """
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.uncertainty_forecasts = uncertainty_forecasts
        self.news_contexts = news_contexts
        self.episode_length = episode_length
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Validate data alignment
        assert len(uncertainty_forecasts) > 0, "Must have pre-computed forecasts"
        assert len(news_contexts) > 0, "Must have pre-computed news contexts"
        
        # Get state dimension
        sample_forecast = list(uncertainty_forecasts.values())[0]
        forecast_dim = sum(len(v) for v in sample_forecast.values())  # Should be 120 (6 paths × 20 days)
        news_dim = len(list(news_contexts.values())[0])  # Should be 18 (3 sentiment + 15 topics)
        state_dim = forecast_dim + news_dim
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action: [directional_bias, volatility_bias, duration] all in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_step = 0
        self.max_steps = len(data) - max_duration - 1
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation (uncertainty cones + news context)."""
        if self.current_step not in self.uncertainty_forecasts:
            # Return zeros if no forecast available
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get uncertainty cone
        forecast = self.uncertainty_forecasts[self.current_step]
        forecast_vec = np.concatenate([
            forecast['price_p10'],
            forecast['price_p50'],
            forecast['price_p90'],
            forecast['vol_p10'],
            forecast['vol_p50'],
            forecast['vol_p90'],
        ])
        
        # Get news context
        news_vec = self.news_contexts.get(self.current_step, np.zeros(18))
        
        # Concatenate
        obs = np.concatenate([forecast_vec, news_vec]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate multiplicative Sharpe-based reward.
        
        Reward = R_dir × (1 + R_vol)
        Where:
        - R_dir = directional_bias × Actual_Sharpe
        - R_vol = volatility_bias × Actual_Vol_Change
        
        Args:
            action: [directional_bias, volatility_bias, duration]
        
        Returns:
            Tuple of (reward, info_dict)
        """
        # Decode action
        dir_bias = action[0]  # [-1, 1]
        vol_bias = action[1]  # [-1, 1]
        
        # Map duration from [-1, 1] to [min_duration, max_duration]
        duration_normalized = (action[2] + 1) / 2  # [0, 1]
        duration = int(self.min_duration + duration_normalized * (self.max_duration - self.min_duration))
        duration = np.clip(duration, self.min_duration, self.max_duration)
        
        # Calculate actual future returns
        start_idx = self.current_step
        end_idx = min(start_idx + duration, len(self.data) - 1)
        actual_duration = end_idx - start_idx
        
        if actual_duration < self.min_duration:
            # Not enough data for evaluation
            return -1.0, {'duration': actual_duration, 'reason': 'insufficient_data'}
        
        # Get returns slice
        returns = self.data.loc[start_idx:end_idx, 'Close'].pct_change().dropna()
        
        if len(returns) < 2:
            return -1.0, {'duration': actual_duration, 'reason': 'insufficient_returns'}
        
        # Calculate Sharpe ratio (annualized)
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return < 1e-6:
            actual_sharpe = 0.0
        else:
            actual_sharpe = (mean_return / std_return) * np.sqrt(252)  # Annualized
        
        # Calculate volatility change
        # Compare volatility during signal period vs. before
        if start_idx >= duration:
            prev_returns = self.data.loc[start_idx-duration:start_idx, 'Close'].pct_change().dropna()
            prev_vol = prev_returns.std()
            current_vol = std_return
            
            if prev_vol < 1e-6:
                vol_change_sign = 0
            else:
                vol_change_sign = 1 if current_vol > prev_vol else -1
        else:
            vol_change_sign = 0
        
        # Calculate reward components
        R_dir = dir_bias * actual_sharpe
        R_vol = vol_bias * vol_change_sign
        
        # Multiplicative reward
        R_total = R_dir * (1 + R_vol)
        
        # Additional bonuses/penalties
        # Bonus for correct direction + strong Sharpe
        if (dir_bias > 0 and actual_sharpe > 0.5) or (dir_bias < 0 and actual_sharpe < -0.5):
            R_total *= 1.2
        
        # Penalty for wrong direction + weak signal
        if (dir_bias > 0 and actual_sharpe < -0.2) or (dir_bias < 0 and actual_sharpe > 0.2):
            R_total *= 0.5
        
        info = {
            'duration': actual_duration,
            'actual_sharpe': actual_sharpe,
            'vol_change_sign': vol_change_sign,
            'R_dir': R_dir,
            'R_vol': R_vol,
            'mean_return': mean_return,
            'std_return': std_return
        }
        
        return float(R_total), info
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to start a new episode."""
        super().reset(seed=seed)
        
        # Random start position
        max_start = self.max_steps - self.episode_length
        if max_start <= 0:
            self.episode_start_step = 0
        else:
            if seed is not None:
                np.random.seed(seed)
            self.episode_start_step = np.random.randint(0, max_start)
        
        self.current_step = self.episode_start_step
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        NOTE: We advance by 1 day (not by signal duration) to allow the model to:
        - React to intra-signal market changes
        - Learn to adapt signals based on new information
        - Make better use of all available forecasts
        
        The reward is still calculated based on the predicted duration's actual outcome.
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Calculate reward based on predicted signal duration
        reward, info = self._calculate_reward(action)
        
        # Advance time by 1 day (not by duration) to make daily predictions
        # This allows the model to:
        # 1. Adapt to market changes within a signal period
        # 2. Learn from errors and adjust subsequent signals
        # 3. Use all pre-computed forecasts efficiently
        self.current_step += 1
        
        # Check if episode done
        steps_in_episode = self.current_step - self.episode_start_step
        terminated = (
            steps_in_episode >= self.episode_length or 
            self.current_step >= self.max_steps
        )
        truncated = False
        
        # Get next observation
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment (not implemented)."""
        pass


# ==================== HELPER FUNCTIONS ====================

def prepare_forecasts_and_context(
    data: pd.DataFrame,
    news_documents: Dict[int, str],  # Mapping timestep -> news text
    forecaster: Optional[UncertaintyForecaster] = None,
    interpreter: Optional[NewsInterpreter] = None,
    feature_columns: Optional[list] = None,
    forecast_horizon: int = 20,
    device: str = 'cpu'
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, np.ndarray]]:
    """
    Pre-compute all forecasts and news contexts for the entire dataset.
    
    This is done once before training to avoid recomputing during episodes.
    
    Args:
        data: DataFrame with OHLCV + indicators
        news_documents: Mapping of timestep -> news text
        forecaster: Trained UncertaintyForecaster (if None, will train)
        interpreter: Fitted NewsInterpreter (if None, will fit)
        feature_columns: Feature columns to use (if None, will auto-select)
        forecast_horizon: Future days to predict (default: 20)
        device: Device for forecaster (default: 'cpu')
    
    Returns:
        Tuple of (uncertainty_forecasts, news_contexts)
    """
    logger.info("Preparing forecasts and context...")
    
    # Auto-select features if not provided
    if feature_columns is None:
        feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'Close' in feature_columns:
            feature_columns.remove('Close')  # Don't use target as feature
    
    # Train forecaster if not provided
    if forecaster is None:
        logger.info("Training uncertainty forecaster...")
        forecaster = train_uncertainty_forecaster(
            data, feature_columns, forecast_horizon=forecast_horizon, device=device
        )
    
    # Fit interpreter if not provided
    if interpreter is None:
        logger.info("Fitting news interpreter on Polygon API news snippets...")
        interpreter = NewsInterpreter()
        interpreter.fit(list(news_documents.values()))
        # Fitting will always succeed (no dummy mode)
    
    # Generate forecasts for all timesteps
    uncertainty_forecasts = {}
    lookback_window = 60
    
    for t in range(lookback_window, len(data) - forecast_horizon):
        # Get input window
        input_window = data.iloc[t-lookback_window:t][feature_columns].values
        input_tensor = torch.FloatTensor(input_window).unsqueeze(0).to(device)  # (1, 60, features)
        
        # Generate uncertainty cone
        forecast = forecaster.predict_with_uncertainty(input_tensor, n_samples=100)
        uncertainty_forecasts[t] = forecast
    
    # Generate news contexts for all timesteps
    news_contexts = {}
    for t, text in news_documents.items():
        try:
            context = interpreter.get_full_context(text)
            news_contexts[t] = context
        except:
            # Use zeros if text processing fails
            news_contexts[t] = np.zeros(18)
    
    logger.info(f"Prepared {len(uncertainty_forecasts)} forecasts and {len(news_contexts)} news contexts")
    
    return uncertainty_forecasts, news_contexts


def create_signal_tester_env(
    data: pd.DataFrame,
    news_documents: Dict[int, str],
    forecaster: Optional[UncertaintyForecaster] = None,
    interpreter: Optional[NewsInterpreter] = None,
    feature_columns: Optional[list] = None,
    forecast_horizon: int = 20,
    episode_length: int = 30,
    device: str = 'cpu',
    n_envs: int = 4,
    use_vectorized: bool = False,
    seed: Optional[int] = None,
    # cleanRL compatibility hooks
    cleanrl_register: bool = False,
    cleanrl_env_id: str = 'SignalTester-v0'
):
    """
    Create SignalTesterEnv with optional vectorization for faster training.
    
    Args:
        data: DataFrame with OHLCV + indicators
        news_documents: Mapping of timestep -> news text
        forecaster: Trained UncertaintyForecaster (optional)
        interpreter: Fitted NewsInterpreter (optional)
        feature_columns: Feature columns to use (optional)
        forecast_horizon: Future days to predict (default: 20)
        episode_length: Steps per episode (default: 30)
        device: Device for forecaster (default: 'cpu')
        n_envs: Number of parallel environments (default: 1)
        use_vectorized: Use vectorized envs for speedup (default: False)
        seed: Random seed for reproducibility (default: None)
        cleanrl_register: If True, register the env id with gymnasium (default: False)
        cleanrl_env_id: The env id to register (default: 'SignalTester-v0')
    
    Returns:
        SignalTesterEnv (single) or gym.vector.SyncVectorEnv (vectorized)
    """
    # Helper: optionally register a gym id for cleanRL scripts
    # If `cleanrl_register=True`, this function will register `cleanrl_env_id`
    # so `gym.make(cleanrl_env_id)` returns a SignalTesterEnv configured with
    # the precomputed forecasts and contexts.
    # Use gym's built-in wrappers and vectorized env API for compatibility.
    import random
    
    # Pre-compute forecasts and context ONCE (shared across all envs)
    uncertainty_forecasts, news_contexts = prepare_forecasts_and_context(
        data, news_documents, forecaster, interpreter, feature_columns, forecast_horizon, device
    )

    # Register a gym id for cleanRL scripts if requested. The registered
    # entry point closes over the precomputed forecasts/context so that
    # gym.make() remains lightweight during training.
    if cleanrl_register:
        try:
            # Check if env already registered
            gym.spec(cleanrl_env_id)
            logger.info(f"Env id '{cleanrl_env_id}' already registered; skipping registration")
        except Exception:
            logger.info(f"Registering gym id '{cleanrl_env_id}' for cleanRL use")

            def _entry_point():
                # Create a fresh environment instance that references the
                # shared precomputed structures (read-only).
                return SignalTesterEnv(
                    data=data,
                    uncertainty_forecasts=uncertainty_forecasts,
                    news_contexts=news_contexts,
                    episode_length=episode_length,
                )

            # Register the env id with gymnasium. Many cleanRL scripts call
            # gym.make(env_id) so providing this id makes integration seamless.
            gym.register(id=cleanrl_env_id, entry_point=_entry_point)
    
    if use_vectorized and n_envs > 1:
        # Vectorized environment for parallel training
        def make_env(rank: int):
            """Create a single environment with unique seed."""
            def _init():
                    env = SignalTesterEnv(
                        data=data.copy(),  # Each env gets its own copy
                        uncertainty_forecasts=uncertainty_forecasts,  # Shared (read-only)
                        news_contexts=news_contexts,  # Shared (read-only)
                        episode_length=episode_length
                    )
                    
                    env = gym.wrappers.RecordEpisodeStatistics(env)
                    if seed is not None:
                        # ensure reproducible starts per worker
                        np.random.seed(seed + rank)
                        random.seed(seed + rank)
                    return env
            return _init
        
        # Set random seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except Exception:
                pass
        
        logger.info(f"Creating vectorized SignalTesterEnv with {n_envs} parallel environments")

        # Create vectorized environment using gym's SyncVectorEnv (keeps API similar to SB3)
        vec_env = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])
        return vec_env
    else:
        # Single environment (backward compatible)
        logger.info("Creating single SignalTesterEnv")
        env = SignalTesterEnv(
            data=data,
            uncertainty_forecasts=uncertainty_forecasts,
            news_contexts=news_contexts,
            episode_length=episode_length
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return env


# Deprecated: we use gym.register directly now in CleanRL.example.ipynb
def register_cleanrl_env(
    env_id: str,
    data: pd.DataFrame,
    news_documents: Dict[int, str],
    forecaster: Optional[UncertaintyForecaster] = None,
    interpreter: Optional[NewsInterpreter] = None,
    feature_columns: Optional[list] = None,
    forecast_horizon: int = 20,
    episode_length: int = 30,
    device: str = 'cpu'
):
    """
    Convenience function to register SignalTesterEnv as a gym environment id.
    
    This is a thin wrapper around create_signal_tester_env with cleanrl_register=True.
    Use this when you want to register the env id separately from creating env instances.
    
    Args:
        env_id: The gym environment id to register (e.g., 'SignalTester-v0')
        data: DataFrame with OHLCV + indicators
        news_documents: Mapping of timestep -> news text
        forecaster: Trained UncertaintyForecaster (optional)
        interpreter: Fitted NewsInterpreter (optional)
        feature_columns: Feature columns to use (optional)
        forecast_horizon: Future days to predict (default: 20)
        episode_length: Steps per episode (default: 30)
        device: Device for forecaster (default: 'cpu')
    
    Example:
        >>> register_cleanrl_env('SignalTester-v0', data, news_documents)
        >>> env = gym.make('SignalTester-v0')
    """
    # Just call create with cleanrl_register=True and discard the returned env
    # The side effect (registration) is what we want
    _ = create_signal_tester_env(
        data=data,
        news_documents=news_documents,
        forecaster=forecaster,
        interpreter=interpreter,
        feature_columns=feature_columns,
        forecast_horizon=forecast_horizon,
        episode_length=episode_length,
        device=device,
        n_envs=1,
        use_vectorized=False,
        cleanrl_register=True,
        cleanrl_env_id=env_id
    )
    logger.info(f"✓ Registered gym env id '{env_id}' for cleanRL use")
