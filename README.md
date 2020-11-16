# RL-project2

2d robotic arm reinforcement learning project based on Pytorch. The environment for the entire project is based on the Mofan's tutorial plus some of our modifications. We learned to use three different algorithms to train the model(DQN, DDPG, A3C), and compared the results of the algorithms

## Group members

* Li Yifan--A3C
* Li Aijia--DQN
* Jiang Yuedong--DDPG

## Quick start

This project contains trained models (training under CPU) under the Model directory, so you can start testing directly


__TEST:__

Change the code of main.py

```python
ON_TRAIN = False
```

Run

```bash
python main.py
```

__TRAIN:__

Change the code of main.py

```python
ON_TRAIN = True
```

Run

```bash
python main.py
```

<br>
