"""
Example Python module for the causal success analysis tutorial.

This module contains helper functions that can be imported into notebooks
or used standalone for running simulations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


class Agent:
    """
    Agent class representing an individual in the simulation.
    """
    
    def __init__(self, agent_id: int, intensity: float, iq: float, 
                 networking: float, initial_capital: float = 1.0):
        self.id = agent_id
        self.talent = {
            'intensity': intensity,
            'iq': iq,
            'networking': networking,
            'initial_capital': initial_capital
        }
        self.capital = initial_capital
        self.capital_history = [initial_capital]
        self.lucky_events = 0
        self.unlucky_events = 0
        
    @property
    def talent_norm(self) -> float:
        values = [self.talent['intensity'], self.talent['iq'], 
                 self.talent['networking'], self.talent['initial_capital']]
        return np.linalg.norm(values)
    
    def get_event_probability(self) -> float:
        alpha = 2.0
        return 1 / (1 + np.exp(-alpha * (self.talent['intensity'] - 0.5)))
    
    def apply_event(self, event_type: str, impact: float):
        if event_type == 'lucky':
            self.capital *= (1 + impact)
            self.lucky_events += 1
        else:
            self.capital = max(0.01, self.capital * (1 - impact))
            self.unlucky_events += 1
        self.capital_history.append(self.capital)


def create_population(n_agents: int = 100, seed: int = 42) -> List[Agent]:
    """
    Create a population of agents with normally distributed talents.
    
    Args:
        n_agents: Number of agents to create
        seed: Random seed for reproducibility
        
    Returns:
        List of Agent objects
    """
    np.random.seed(seed)
    agents = []
    
    for i in range(n_agents):
        intensity = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        iq = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        networking = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        
        agents.append(Agent(i, intensity, iq, networking))
    
    return agents


def calculate_gini(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient for inequality measurement.
    
    Args:
        values: Array of values (e.g., capital amounts)
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((n - np.arange(n)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n


def get_results_dataframe(agents: List[Agent]) -> pd.DataFrame:
    """
    Convert agent list to pandas DataFrame for analysis.
    
    Args:
        agents: List of Agent objects
        
    Returns:
        DataFrame with agent attributes and outcomes
    """
    results = []
    for agent in agents:
        results.append({
            'id': agent.id,
            'talent_intensity': agent.talent['intensity'],
            'talent_iq': agent.talent['iq'],
            'talent_networking': agent.talent['networking'],
            'talent_norm': agent.talent_norm,
            'capital': agent.capital,
            'lucky_events': agent.lucky_events,
            'unlucky_events': agent.unlucky_events,
            'net_events': agent.lucky_events - agent.unlucky_events
        })
    return pd.DataFrame(results)
