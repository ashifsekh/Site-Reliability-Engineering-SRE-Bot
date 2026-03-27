---
title: SRE Cloud Agent Auto-Scaler
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SRE Cloud Agent Environment

An OpenEnv-compatible environment where an AI agent acts as a Site Reliability Engineer (SRE).

## Description

The agent manages a cluster of web servers. It must balance performance (low latency) against costs (server count) while handling unexpected traffic spikes and server crashes.

## Action Space

- `0`: Do Nothing
- `1`: Scale Up (Add 1 Server)
- `2`: Scale Down (Remove 1 Server)
- `3`: Restart (Fix a crashed server)

## Observation Space

A vector of 7 normalized values:

`[CPU Usage, Memory Usage, Active Servers, Latency, Is_Crashed, Traffic_Level, Budget_Remaining]`

## Tasks

- **Easy**: Short duration, high budget.
- **Medium**: Standard duration, normal budget.
- **Hard**: Long duration, restricted budget.

## Setup and Run

```bash
pip install -r requirements.txt
python baseline.py
```

## Deployment

This environment is containerized via Docker and deployable to Hugging Face Spaces.

## Next Steps

1. **Copy the files**: Create the folder structure and paste the code.
2. **Test locally**: Run `python baseline.py`. You should see the simulation run and output scores.
3. **Deploy**:
   - Create a Hugging Face Space (select **Docker** as the SDK).
   - Push these files to the repository.
   - The Space will automatically build and run the baseline, showing the scores.

## Coverage

This solution covers every bullet point in the problem statement:

- **Real-world**: Simulates server management.
- **Full Spec**: `openenv.yaml` and typed Pydantic models included.
- **3 Tasks**: Defined in YAML and `baseline.py`.
- **Rewards**: Implemented partial signals (latency + budget).
- **Deployment**: Dockerfile provided.

## Optional Tweaks

You can tune the environment logic further, for example by making crashes happen more often to stress-test resilience behavior.
