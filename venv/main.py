import gym
import time


class BespokeAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):
         position, velocity = observation
         lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
         ub = -0.07 * (position + 0.38) ** 2 + 0.06
         if lb < velocity < ub:
             action = 2
         else:
             action = 0
         return action  # 返回动作

    def learn(self, *args):  # 学习
         pass

    def play_ones(self, env, agent, render=False, train=False):
         episode_reward = 0  # 记录回合总奖励，初始值为0
         observation = env.reset()  # 重置游戏环境，开始新回合
         while True:  # 不断循环，直到回合结束
             if render:  # 判断是否显示
                 env.render()  # 显示图形界面，可以用env.close()关闭
             action = agent.decide(observation)
             next_observation, reward, done, _ = env.step(action)  # 执行动作
             episode_reward += reward  # 搜集回合奖励
             if train:  # 判断是否训练智能体
                 break
             observation = next_observation
         return episode_reward  # 返回回合总奖励


if __name__ == '__main__':
     env = gym.make('MountainCar-v0')
     env.seed(0)  # 设置随机数种子，只是为了让结果可以精确复现，一般情况下可以删除

     agent = BespokeAgent(env)
     for _ in range(100):
         episode_reward = agent.play_ones(env, agent, render=True)
         print('回合奖励={}'.format(episode_reward))

     time.sleep(10)  # 停顿10s
     env.close()  # 关闭图形化界面