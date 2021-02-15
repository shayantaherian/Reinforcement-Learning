# Reinforcement Learning with pytorch
This repository implements the classic and state-of-the-art deep reinforcement learning algorithms using pytorch. The goal of this repository is to provide an easy to read pytorch reinforcement learning implementation.

### What is included?
Currently this respository includes the agents:
* Deep Q-learning [[1]](https://arxiv.org/abs/1312.5602)
* Deep Deterministic Policy Gradient (DDPG) [[2]](https://arxiv.org/abs/1509.02971)

### Requirements
* Python 3.7
* gym >= 0.10
* pytorch >= 0.4

### Installation

```
pip install -r requirements.txt 
```

In case failing with installation:

* Install gym environment
```
pip install gym 
```

* Install pytorch
```
please go to official webisite to install it: https://pytorch.org/

Recommend use Anaconda Virtual Environment to manage your packages
``` 

### Deep Q-learning
The result of the DQN for training the cart pole (`CartPole-v0`) is upleaded here

![Training_result](https://user-images.githubusercontent.com/51369142/85757437-6081e100-b707-11ea-9ac7-d337937dfa99.png)

First clone the repository

```
git clone https://github.com/shayantaherian/Reinforcement-Learning.git
```

Then move to the directory

```
cd Deep Q-learning
```

To start training, run the following command 

```
python main.py
```

### Deep Deterministic Policy Gradient (DDPG)
The result of the DQN for training the pendulum (`Pendulum-v0`) is upleaded here

![Training_result](https://user-images.githubusercontent.com/51369142/85760923-31b93a00-b70a-11ea-9eb9-6bc944999475.png)

First clone the repository

```
git clone https://github.com/shayantaherian/Reinforcement-Learning.git
```

Then move to the directory

```
cd DDPG
```

To start training, run the following command 

```
python main.py
```

To test the result for generalization purposes run 

```
python Test_result.py
```

### References
1. [DeepQLearning](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning)
2. [Deeplizard](https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv)
3. [Deep Deterministic Policy Gradients](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)
