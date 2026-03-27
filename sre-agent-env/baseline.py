import numpy as np
from env import SREEnv

def heuristic_agent(observation):
    """
    A simple rule-based agent to establish a baseline score.
    Logic:
    - If crashed (obs[4] > 0.5): Restart.
    - If CPU (obs[0]) > 0.8: Scale Up.
    - If CPU (obs[0]) < 0.2: Scale Down.
    - Else: Do nothing.
    """
    cpu = observation[0]
    is_crashed = observation[4]
    
    if is_crashed > 0.5:
        return 3 # Restart
    
    if cpu > 0.8:
        return 1 # Scale Up
    
    if cpu < 0.2:
        return 2 # Scale Down
        
    return 0 # Do nothing

def run_baseline(task_name, config):
    print(f"\n--- Running Task: {task_name} ---")
    env = SREEnv(config=config)
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    done = False
    
    while not done:
        action = heuristic_agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Optional: Render for debugging
        # env.render()
        
    print(f"Task: {task_name} | Final Score: {total_reward:.2f}")
    return total_reward

if __name__ == "__main__":
    # Define the 3 Tasks (Easy -> Hard)
    
    # Task 1: Easy - Stable low traffic, just keep servers running
    task1_config = {"max_steps": 50, "initial_budget": 200}
    
    # Task 2: Medium - Normal traffic fluctuations
    task2_config = {"max_steps": 100, "initial_budget": 150}
    
    # Task 3: Hard - Tight budget, long episode
    task3_config = {"max_steps": 200, "initial_budget": 100}
    
    scores = []
    scores.append(run_baseline("Easy Stability", task1_config))
    scores.append(run_baseline("Medium Fluctuation", task2_config))
    scores.append(run_baseline("Hard Budget Constraint", task3_config))
    
    print("\n--- Baseline Summary ---")
    print(f"Average Score: {np.mean(scores):.2f}")