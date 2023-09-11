# Barrier Function Inspired Reward Shaping
IssacGym implementations of various environments with Barrier Function inspired reward shaping

## Environment Setup
Download and extract the [Isaac Gym preview release](https://developer.nvidia.com/isaac-gym). Supported Python versions are 3.7 or 3.8. Next create a `conda` or `venv` virtual environment and launch it. 

```
python3 -m venv rl-env
source rl-env/bin/activate
```

In the `python` subdirectory of the extracted folder, run:

```
pip install -e .
```

This will install the `isaacgym` package and all of its dependencies in the active Python environment. Now clone this repo using git.

```
git clone https://github.com/hskalin/multi-rl.git
```

## Running
In the `multi-rl` directory run

```
python run.py --exp-name PPO --env-id Ant
```
 
## Credits
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Isaac Gym Benchmark Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
