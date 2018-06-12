# Overcoming-exploration-from-demos
Implementation of the paper "Overcoming Exploration in Reinforcement Learning with Demonstrations" Nair et al. over the HER baselines from OpenAI


> Note: This repository is a modification of her baselines from OpenAI

To know more please visit my blog at https://jangirrishabh.github.io/2018/03/25/Overcoming-exploration-demos.html

## Installation 
- Install Python 3.5 or higher

- Clone the baselines from OpenAI, use the following commit in case of any conflicts - a6b1bc70f156dc45c0da49be8e80941a88021700

- Clone this package in your working directory with `git clone https://github.com/jangirrishabh/Overcoming-exploration-from-demos.git`

- Add this package to your PYTHONPATH or if you are not familiar with that alternatively edit `sys.path.append('/path/to/Overcoming-exploration-from-demos/')`  in **train.py**, **play.py** and **config.py**

## File descriptions and Usage

### experiments/train.py
- Configure the run parameters at the bottom of the file, additional parameters since the her baselines are:
	- '--bc_loss', type=int, default=0, help='whether or not to use the behavior cloning loss as an auxilliary loss'
	- '--q_filter', type=int, default=0, help='whether or not a Q value filter should be used on the Actor outputs'
	- '--num_demo', type=int, default = 0, help='number of expert demo episodes'
- Edit `demoFileName = 'your/demo/file'` to point to the file that contains your recorded demonstrations
- To start the training use `python experiment/train.py`

### experiments/play.py
- The above training paradigm spits out policies as .pkl files after every 5 epochs (can be modified) which we can then replay and evaluate with this file
- To play the policy execute `python experiments/play.py /path/to/saved/policy.pkl`

### experiments/config.py
All the training hyperparameters can be configured through this file, feel free to experiment with different combinations and record results

### ddpg.py
- Contains the main DDPG algorithm with a modified network where the losses are changed based on the whether demonstrations are provided for the task. Basically we maintain a separate demoBuffer and sample from this as well
- Following parameters are to be configured here:
	- self.demo_batch_size : number of demos out of total buffer size (32/256 default)
	- self.lambda1, self.lambda2 : correspond to the weights given for Q loss and Behavior cloning loss respectively

<!-- ###her.py

###normalizer.py

###replay_buffer.py

###rollout.py

###actor_critic.py -->


