# Neuroevolution of Self-Interpretable Agents (Simplified)

![attentionagent](https://storage.googleapis.com/quickdraw-models/sketchRNN/attention/assets/card/attentionagent.gif)  
Our agent receives visual input as a stream of 96x96px RGB images (left). Each image frame is passed through a self-attention bottleneck module, responsible for selecting K=10 patches (highlighted in white, middle). Features from these K patches (such as location) are then routed to a decision-making controller (right) that will produce the agent’s next action. The parameters of the self-attention module and the controller are trained together using neuroevolution.

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

# Evaluate CartPole with GUI.
python eval_agent.py --log-dir=pretrained/cartpole --render
```

## Training

To train on a local machine, run the following command:
```
# Train CarRacing locally.
python train_agent.py --config=configs/carracing.gin --log-dir=log/carracing --reps=3

# Train CartPole locally on GPUs.
python train_agent.py --config=configs/cartpole.gin --log-dir=log/cartpole --num-gpus=8

# Train the Visual Cartpole locally.
python train_agent.py --config=configs/visualcartpole.gin --log-dir=log/visualcartpole

# Train Breakout locally.
python train_agent.py --config=configs/breakout.gin --log-dir=log/breakout

# Train Coinrun locally.
python train_agent.py --config=configs/coinrun.gin --log-dir=log/coinrun
```
Please see `train_agent.py` for other command line options.

## Citation
For attribution in academic contexts, please cite this work as

```
@inproceedings{attentionagent2020,
  author    = {Yujin Tang and Duong Nguyen and David Ha},
  title     = {Neuroevolution of Self-Interpretable Agents},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  url       = {https://attentionagent.github.io},
  note      = "\url{https://attentionagent.github.io}",
  year      = {2020}
}
```

## Disclaimer

This is not an official Google product.
