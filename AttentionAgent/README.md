# Neuroevolution of Self-Interpretable Agents (Simplified)

![attentionagent](https://storage.googleapis.com/quickdraw-models/sketchRNN/attention/assets/card/attentionagent.gif) 
![coinrun](https://media.giphy.com/media/ueNQ2HILL5nnaY0I7J/giphy-downsized.gif)
![breakout](https://media.giphy.com/media/3jU4sxm4Mc7LYWuOA5/giphy.gif)

CarRacing &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Coinrun &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Breakout

![vcartpole](https://media.giphy.com/media/jtGtJq5zWKuyPM3xQS/giphy.gif)

Visual-CartPole

Our agent receives visual input as a stream of RGB images. Each image frame is passed through a self-attention bottleneck module, responsible for selecting K=10 patches. Features from these K patches (such as location) are then routed to a decision-making controller that will produce the agent’s next action. The parameters of the self-attention module and the controller are trained together using neuroevolution.

This repository contains the code to reproduce the results presented in the orignal [paper](https://attentionagent.github.io/).

## Additional informations

This code has been tested on python 3.7.

It includes three new tasks:

    -Visual version of the classic CartPole task (https://gym.openai.com/envs/CartPole-v0/)
    -Atari Breakout (https://gym.openai.com/envs/Breakout-v0/ , you may need to install the Atari ROM to use it, see last line in the requirements)
    -Procgen Coinrun (https://openai.com/blog/procgen-benchmark/)

## Evaluate pre-trained models

You can run the following commands to evaluate the trained agent.
```
# Evaluate for 100 episodes.
python eval_agent.py --log-dir=pretrained/carracing --n-episodes=100

# Evaluate CarRacing with GUI.
python eval_agent.py --log-dir=pretrained/carracing --render

# Evaluate Procgen Coinrun.
python eval_agent.py --log-dir=pretrained/coinrun

# Evaluate Atari Breakout.
python eval_agent.py --log-dir=pretrained/breakout --render

# Evaluate Visual CartPole with GUI.
python eval_agent.py --log-dir=pretrained/visual-cartpole --render
```

## Training

To train on a local machine, run the following command:
```
# Train CarRacing locally.
python train_agent.py --config=configs/carracing.gin --log-dir=log/carracing --reps=3

# Train the Visual CartPole locally.
python train_agent.py --config=configs/visualcartpole.gin --log-dir=log/visualcartpole

# Train Breakout locally.
python train_agent.py --config=configs/breakout.gin --log-dir=log/breakout

# Train Coinrun locally.
python train_agent.py --config=configs/coinrun.gin --log-dir=log/coinrun
```
Please see `train_agent.py` for other command line options.
