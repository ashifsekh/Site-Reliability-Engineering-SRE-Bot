import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any, Tuple, List

# --- Typed Models (Requirement: Typed Models) ---
class SREState(BaseModel):
    cpu_usage: float
    memory_usage: float
    active_servers: int
    latency_ms: float
    is_crashed: bool
    current_traffic: float
    budget_remaining: float

# --- The Environment ---
class SREEnv(gym.Env):
    """
    A real-world SRE environment where an agent manages a web server cluster.
    Goal: Keep latency low and servers running without going over budget.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        # Configuration for difficulty levels
        self.config = config or {}
        self.max_servers = self.config.get("max_servers", 10)
        self.initial_budget = self.config.get("initial_budget", 100.0)
        self.max_steps = self.config.get("max_steps", 100)
        
        # --- Action Space ---
        # 0: Do Nothing, 1: Scale Up (+1 server), 2: Scale Down (-1), 3: Restart (Fix crash)
        self.action_space = spaces.Discrete(4)

        # --- Observation Space ---
        # [CPU %, Mem %, Active Servers, Latency, IsCrashed(0/1), Traffic Level, Budget Left]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )

        self.state = None
        self.current_step = 0
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.active_servers = 2
        self.budget = self.initial_budget
        self.is_crashed = False
        self.traffic = 0.3 # Initial low traffic
        
        obs = self._get_obs()
        info = {"state": self._get_state_model().dict()}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        reward = 0.0
        truncated = False
        terminated = False
        
        # 1. Apply Action Logic
        cost_per_server = 0.5
        
        if action == 1: # Scale Up
            if self.active_servers < self.max_servers and self.budget >= cost_per_server:
                self.active_servers += 1
                self.budget -= cost_per_server * 2 # Scaling cost
        elif action == 2: # Scale Down
            if self.active_servers > 1:
                self.active_servers -= 1
        elif action == 3: # Restart/Heal
            if self.is_crashed:
                self.is_crashed = False
                self.budget -= 5.0 # Heavy cost to restart
                reward += 0.5 # Small reward for fixing issue
        
        # 2. Simulate World Dynamics (Traffic & Load)
        # Traffic fluctuates over time (Simulating a daily cycle)
        self.traffic = 0.5 + 0.4 * np.sin(self.current_step / 20.0) + np.random.normal(0, 0.05)
        self.traffic = np.clip(self.traffic, 0.1, 1.0)

        # Calculate Load
        load_per_server = self.traffic / (self.active_servers + 0.01)
        
        cpu_usage = np.clip(load_per_server * 1.2 + np.random.normal(0, 0.05), 0, 1)
        memory_usage = np.clip(load_per_server * 0.9 + np.random.normal(0, 0.05), 0, 1)
        
        # 3. Update System State
        if not self.is_crashed:
            # Crash condition: CPU too high for too long
            if cpu_usage > 0.95 and np.random.rand() < 0.2:
                self.is_crashed = True
        
        # Calculate Latency
        if self.is_crashed:
            latency = 5000.0 # ms (Timeout)
        else:
            latency = 50.0 + (cpu_usage * 500.0) # Latency increases with CPU load

        # 4. Calculate Reward (Partial Progress Signals)
        # Reward: Low latency is good. Saving money is good. Crashing is bad.
        
        # Normalize latency reward (0 to 1). 100ms is ideal, 1000ms is terrible.
        latency_score = 1.0 - np.clip((latency - 50) / 1000, 0, 1)
        
        # Penalize crash heavily
        if self.is_crashed:
            latency_score = -1.0
            
        # Budget efficiency (don't just spawn 10 servers for 1 user)
        budget_score = self.budget / self.initial_budget
        
        # Composite Reward
        reward = (latency_score * 0.7) + (budget_score * 0.3)
        
        # Operational cost per step
        self.budget -= (self.active_servers * cost_per_server)
        
        # Check termination
        if self.budget <= 0:
            terminated = True
            reward -= 10 # Bankruptcy penalty
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Update internal state object
        self.state = {
            "cpu": cpu_usage,
            "mem": memory_usage,
            "servers": self.active_servers,
            "latency": latency,
            "crashed": float(self.is_crashed),
            "traffic": self.traffic,
            "budget": self.budget
        }

        return self._get_obs(), reward, terminated, truncated, {"state": self._get_state_model().dict()}

    def _get_obs(self) -> np.ndarray:
        # Returns normalized numpy array
        if self.state is None:
            return np.zeros(7, dtype=np.float32)
        
        # Normalize latency for observation (log scale helps agents learn better)
        norm_latency = np.log1p(self.state["latency"]) / 10.0
        
        return np.array([
            self.state["cpu"],
            self.state["mem"],
            self.state["servers"] / self.max_servers,
            norm_latency,
            self.state["crashed"],
            self.state["traffic"],
            self.state["budget"] / self.initial_budget
        ], dtype=np.float32)

    def _get_state_model(self) -> SREState:
        if self.state is None:
            return SREState(
                cpu_usage=0, memory_usage=0, active_servers=0, 
                latency_ms=0, is_crashed=False, current_traffic=0, budget_remaining=0
            )
        return SREState(
            cpu_usage=self.state["cpu"],
            memory_usage=self.state["mem"],
            active_servers=int(self.state["servers"]),
            latency_ms=self.state["latency"],
            is_crashed=bool(self.state["crashed"]),
            current_traffic=self.state["traffic"],
            budget_remaining=self.state["budget"]
        )

    def render(self):
        if self.state:
            status = "CRASHED" if self.state["crashed"] else "ONLINE"
            print(f"Step {self.current_step}: Servers={self.state['servers']}, CPU={self.state['cpu']:.2f}, Latency={self.state['latency']:.0f}ms, Status={status}")