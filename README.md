# RL-project2

2d robotic arm reinforcement learning project based on Pytorch. The environment for the entire project is based on the Mofan's tutorial plus some of our modifications.

## Requirements

1. python=3.6
2. pytorch=1.4.0
3. pyglet==1.5.0
4. gym=0.12（No need MuJoCo ）

## Quick start

This project contains trained models (training under CPU) under the Model directory, so you can start testing directly


__TEST:__

修改main.py代码

```python
ON_TRAIN = False
```

运行

```bash
python main.py
```

__TRAIN:__

修改main.py代码

```python
ON_TRAIN = True
```

运行

```bash
python main.py
```

<br>
