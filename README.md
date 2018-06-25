# Overcoming-exploration-from-demos
Implementation of the paper "Overcoming Exploration in Reinforcement Learning with Demonstrations" Nair et al. over the HER baselines from OpenAI


> Note: This repository is a modification of her baselines from OpenAI

To know more please visit my blog at https://jangirrishabh.github.io/2018/03/25/Overcoming-exploration-demos.html


## Installation 
- Install Python 3.5 or higher

- Clone the baselines from OpenAI, use the following commit in case of any conflicts - a6b1bc70f156dc45c0da49be8e80941a88021700

- Clone this package in your working directory with `git clone https://github.com/jangirrishabh/Overcoming-exploration-from-demos.git`

- Add this package to your PYTHONPATH or if you are not familiar with that alternatively edit `sys.path.append('/path/to/Overcoming-exploration-from-demos/')`  in **train.py**, **play.py** and **config.py**


## Branches
TThe **gymEnv** branch deals with code that aims to use demonstrations in Fetch Environment tasks, though currently I lack a framework to generate demonstrations in Fetch environments, work towards using HTC Vibe for generating demonstration data in VR is in progress. The **master** branch is more advanced as I have frameworks for generating demonstartions using the Inverse Kinematics and Forward Kinematics nodes developed at IRI. Also, in [this](https://github.com/jangirrishabh/HER-learn-InverseKinematics) repository I integrated the barret WAM Gazebo simulation with OpenAI gym with the help of [Gym-gazebo](https://github.com/erlerobot/gym-gazebo), thus the simulation environment in gazebo can now be used as a stanalone gym environment with all the functionalities.

<p align="center">
  <img src="https://github.com/jangirrishabh/jangirrishabh.github.io/blob/master/assets/research/wam_single_block_reach.png"/>
</p>

## File descriptions and Usage

### [Training](experiments/train.py)
Configure the run parameters at the bottom of the file, additional parameters since the her baselines are:
- '--bc_loss', type=int, default=0, help='whether or not to use the behavior cloning loss as an auxilliary loss'
- '--q_filter', type=int, default=0, help='whether or not a Q value filter should be used on the Actor outputs'
- '--num_demo', type=int, default = 0, help='number of expert demo episodes'.

Edit `demoFileName = 'your/demo/file'` to point to the file that contains your recorded demonstrations
To start the training use `python experiment/train.py`

### [Playing](experiments/play.py)
The above training paradigm spits out policies as .pkl files after every 5 epochs (can be modified) which we can then replay and evaluate with this file. To play the policy execute `python experiments/play.py /path/to/saved/policy.pkl`

### [Configuration](experiments/config.py)
All the training hyperparameters can be configured through this file, feel free to experiment with different combinations and record results

### [DDPG agent](ddpg.py)
Contains the main DDPG algorithm with a modified network where the losses are changed based on the whether demonstrations are provided for the task. Basically we maintain a separate demoBuffer and sample from this as well. Following parameters are to be configured here:
- self.demo_batch_size : number of demos out of total buffer size (32/256 default)
- self.lambda1, self.lambda2 : correspond to the weights given for Q loss and Behavior cloning loss respectively


Major contributions of the paper include the following aspects which I have tried to implement over the HER baselines:

## Demonstration Buffer used along with the exploration replay buffer
First, we maintain a second replay buffer R<sub>D</sub> where we store our demonstration data in the same format as R. In each minibatch, we draw an extra N<sub>D</sub> examples from R<sub>D</sub> to use as off-policy replay data  for the update step. These examples are included in both the actor and critic update.

```python

self.demo_batch_size = 32 #Number of demo samples

def initDemoBuffer(self, demoDataFile, update_stats=True): 
#To initiaze the demobuffer with the recorded demonstration data. We also normalize the demo data.

def sample_batch(self):
    if self.bc_loss:
        transitions = self.buffer.sample(self.batch_size - self.demo_batch_size)
        global demoBuffer

        transitionsDemo = demoBuffer.sample(self.demo_batch_size)

        for k, values in transitionsDemo.items():
            for v in values:
                rolloutV = transitions[k].tolist()
                rolloutV.append(v.tolist())
                transitions[k] = np.array(rolloutV)
    else:
        transitions = self.buffer.sample(self.batch_size)

```


## Behavior Cloning Loss applied on the actor's actions
Second, we introduce a new loss computed only on the demonstration examples for training the actor. This loss is a standard loss in imitation learning, but we show that using it as an  auxiliary loss for RL improves learning significantly. The gradient applied to the actor parameters is. (Note  that  we  maximize J and  minimize L<sub>BC</sub>. Using this loss directly prevents the learned policy from improving significantly beyond the demonstration policy, as the actor is always tied back to the demonstrations. 

> Please read the paper to go through the meaning of the symbols used in these equations

## Q-value filter to account for imperfect demonstrations
We account for the possibility that demonstrations can be suboptimal by applying the behavior cloning loss only to states  where  the  critic Q(s,a)  determines  that  the  demonstrator action is better than the actor action. In python this looks like:

```python

self.lambda1 = 0.004
self.lambda2 =  0.031

def _create_network(self, reuse=False):

	mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis = 0)

	target_Q_pi_tf = self.target.Q_pi_tf
    clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
    target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range) # y = r + gamma*Q(pi)
    self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf)) #(y-Q(critic))^2

    if self.bc_loss ==1 and self.q_filter == 1 :
        maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf >= self.main.Q_pi_tf, mask), [-1])
        self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask(tf.boolean_mask((self.main.pi_tf / self.max_u), mask), maskMain, axis=0) - tf.boolean_mask(tf.boolean_mask((batch_tf['u']/ self.max_u), mask), maskMain, axis=0)))
        self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.Q_pi_tf)
        #self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf

    elif self.bc_loss == 1 and self.q_filter == 0:
        self.cloning_loss_tf = tf.reduce_sum(tf.square(tf.boolean_mask((self.main.pi_tf / self.max_u), mask) - tf.boolean_mask((batch_tf['u']/ self.max_u), mask)))
        self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.Q_pi_tf)
        #self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf

    else:
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
```
Here, we first mask the samples such as to get the cloning loss only on the demonstration samples by using the `tf.boolean_mask` function. We have 3 types of losses depending on the chosen run-paramters. When using both behavior cloning loss with Q_Filter we create another mask that enables us to apply the behavior cloning loss to only those states where the critic Q(s,a) determines that the demonstrator action is better than the actor action.

## Experimentation
The work is in progress and most of the experimentation is being carried out on a Barret WAM simulator, that is because I have access to a Barret WAM robot through the Perception and Manipulation Lab, IRI. I have frameworks for generating demonstartions using the Inverse Kinematics and Forward Kinematics nodes developed at IRI. Also, in [this](https://github.com/jangirrishabh/HER-learn-InverseKinematics) repository I integrated the barret WAM Gazebo simulation with OpenAI gym with the help of [Gym-gazebo](https://github.com/erlerobot/gym-gazebo), thus the simulation environment in gazebo can now be used as a stanalone gym environment with all the functionalities. The plan is to first learn the initial policy on a simulation and then transfer it to the real robot, exploration in RL can lead to wild actions which are not feasible when working with a physical  platform. 



## Tasks
The types of tasks I am considering for now are - 
- [x] Learning Inverse Kinemantics (learning how to reach a particular point inside the workspace)
- [x] Learning to grasp a block and take it to a given goal inside the workspace
- [ ] Learning to stack a block on top of another block
- [ ] Learning to stack 4 blocks on top of each other

> Note: All of these tasks have a sparse reward structure i.e. 0 if the task is complete else a -1.

## Generating demonstrations
Currently using a simple python script to generate demonstrations with the help of Inverse IK and Forward IK functionalities already in place for the robot I am using. Thus not all the generated demonstrations are perfect, which is good as our algorithm uses a Q-filter which accounts for all the bad demonstration data. The video below shows the demonstration generating paradigm for a 2 block stacking case, where one of the blocks is already at its goal position and the task involves stacking the second block on top of this block, the goal positions are shown in red in the rviz window next to gazebo (it is way easier to have markers in rviz than gazebo). When the block reaches its goal position the marker turns green.

> Please visit my [blog](https://jangirrishabh.github.io/2018/03/25/Overcoming-exploration-demos.html) to see the videos.

## Training details and Hyperparameters
We train the robot with the above shown demonstrations in the buffer. We sample a total of 100 demonstrations/rollouts and in every minibatch sample N<sub>D</sub> = 32 samples from the demonstrations in a total of N = 256 samples, the rest of the samples are generated when the arm interacts with the environment. To train the model we use Adam optimizer with learning rate 0.001 for both critic and actor networks. The discount factor is 0.98. To explore during training, we sample random actions uniformly within the action space with a probability of 0.2 at every step, with an additional uniform gaussian noise which is 5% of the maximum value of each action dimension. For details about more hyperparameters, refer to config.py in the source code. 


## Resulting Behaviors
Please visit my [blog](https://jangirrishabh.github.io/2018/03/25/Overcoming-exploration-demos.html) to see the videos. The video shows the agent's learned behavior corresponding to the task of stacking one block on top of the other. It learns to pick up the block, reach to a perfect position to drop the block but still does not learn to take the action that results in dropping the block to the goal. As I said this is a work in progress and I am still working towards improving the performace of the agent in this task. For the other easier tasks of reaching a goal position and picking up the block it shows perfect performance, and I have not reported those results as they are just a subset of the current problem which I am trying to solve.

<!-- ###her.py

###normalizer.py

###replay_buffer.py

###rollout.py

###actor_critic.py -->


