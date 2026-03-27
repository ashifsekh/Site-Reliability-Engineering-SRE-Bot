from fastapi import FastAPI
import baseline

app = FastAPI()

@app.get("/")
def run_agent_evaluation():
    """
    This endpoint runs the SRE environment baseline when someone visits the Space.
    """
    # Task Configurations
    task1_config = {"max_steps": 50, "initial_budget": 200}
    task2_config = {"max_steps": 100, "initial_budget": 150}
    task3_config = {"max_steps": 200, "initial_budget": 100}

    # Run the agent
    score1 = baseline.run_baseline("Easy Stability", task1_config)
    score2 = baseline.run_baseline("Medium Fluctuation", task2_config)
    score3 = baseline.run_baseline("Hard Budget Constraint", task3_config)

    avg_score = (score1 + score2 + score3) / 3

    # Return the results as a clean JSON response on the webpage
    return {
        "project": "SRE Cloud Agent Auto-Scaler",
        "status": "Evaluation Complete",
        "scores": {
            "Task 1 (Easy Stability)": round(score1, 4),
            "Task 2 (Medium Fluctuation)": round(score2, 4),
            "Task 3 (Hard Budget Constraint)": round(score3, 4),
            "Average Overall Score": round(avg_score, 4)
        },
        "message": "OpenEnv fully compliant and operational."
    }