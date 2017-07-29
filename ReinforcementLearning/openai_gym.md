# openai-gym 介绍

## 安装

> ubuntu16.04 , python3.5



```shell
pip install gym
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
pip install 'gym[all]'
```



## 使用

**核心方法：**

* `gym.make(game_name)`
* `env.reset()`
* `env.render()`
* `env.step(action)`

```python
import gym
env = gym.make('SpaceInvaders-v0') # 创建一个游戏环境
env.reset() # 重置环境
env.render() # 渲染环境，即，将游戏环境可视化出来
# Each timestep, the agent chooses an action, and the environment returns an observation 
# and a reward.
observation, reward, done, info=env.step(action) # 执行一个step
```



## 其它

所有的游戏都在这 [https://github.com/openai/gym/blob/master/gym/envs/__init__.py](https://github.com/openai/gym/blob/master/gym/envs/__init__.py)

游戏的代码 [https://github.com/openai/gym/tree/master/gym/envs](https://github.com/openai/gym/tree/master/gym/envs)

官方文档 [https://gym.openai.com/docs](https://gym.openai.com/docs)



