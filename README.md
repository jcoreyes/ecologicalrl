# Ecological RL
Under review as a conference paper at NeurIPS 2020.

## Terminology
The following is a mapping from terminology used in the codebase to that used in the paper, where it differs:
 - `axe` in the code corresponds to the "salad-making task" in the paper
 - `deer` in the code corresponds to the "hunting task" in the paper
 - `monster` in the code corresponds to the "scavenging task" in the paper
 - `factory` in the code corresponds to the "factory task" in the paper
 - `waypoint` in the code refers to the "subgoal reward" in the paper

## Experiments by Figure
 - Figures 2 (State Visitation Maps and Performance Curve of Episodic and Non-Episodic Learning)
   - [experiment script directory](experiments/continual/measure/dynamism/entropy/axe/tool_dqn_maps.py)
 - Figures 4 (Effect of Dynamic Environment on Episodic and Non-Episodic Learning) and 6 (Dynamic Ablations)
   - [experiment script directory](experiments/continual/dynamic_static)
 - Figure 5 (Shaping Methods for Episodic and Non-Episodic Learning)
   - [experiment script directory](experiments/continual/env_shaping/distance_increasing)
 - Figure 8 (Shaping Methods for Walled Salad-Making Task)
   - [experiment script](experiments/continual/env_shaping/env_vs_reward/wall/tool_dqn_wall_train.py)
 - Figures 9 and 10 (State Visitation Counts for Walled Salad-Making Task)
   - [Jupyter notebook used to generate figure](data/scripts/gen_heatmaps.ipynb)
 - Figures 11 and 12 (Learned Behavior on Salad-Making and Hunting Tasks)
   - [script used to generate figure](data/scripts/gen_validation_rollout_gifs_heatmaps.py)
 - Table 1 (Hitting Time and Marginal State Entropy for Dynamism and Environment Shaping)
   - [experiment script directory](experiments/continual/measure)
   
## Experiment Workflow
The experiments require a set of validation environments for performance evaluation. Paired with each experiment script
is a script called `gen_validation_envs.py`, which outputs a Python pickle file containing these validation
environments, the path to which can be fed in directly to the experiment script as its `validation_envs_pkl` argument
in the `algorithm_kwargs` dictionary found in each experiment script.
