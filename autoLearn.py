import numpy as np
import matplotlib.pyplot as plt


class KB_Game:
    def __init__(self, *args, **kwargs):
        # 初始化参数
        self.q = np.array([0.0, 0.0, 0.0])  # 每个臂的平均回报
        self.action_counts = np.array([0, 0, 0])  # 摇动每个臂的次数
        self.current_cumulative_rewards = 0.0  # 当前累积回报总和
        self.actions = [1, 2, 3]  # 3个不同的摇臂
        self.counts = 0  # 玩家玩游戏的次数
        self.counts_history = []  # 玩家玩游戏的次数记录
        self.cumulative_rewards_history = []  # 累积回报的记录
        self.a = 1  # 玩家当前动作，初始化为1，摇第一个摇臂
        self.reward = 0  # 当前回报，初始值为0

    def step(self, a):  # 模拟器
        # 根据选择的摇臂返回奖励
        r = 0
        if a == 1:
            r = np.random.normal(1, 1)  # 正态分布，均值为1，方差为1
        elif a == 2:
            r = np.random.normal(2, 1)
        elif a == 3:
            r = np.random.normal(1.5, 1)
        return r

    def choose_action(self, policy, **kwargs):  # 动作策略
        action = 0
        if policy == 'e_greedy':
            if np.random.random() < kwargs['epsilon']:
                action = np.random.randint(1, 4)
            else:
                action = np.argmax(self.q) + 1
        elif policy == 'ucb':
            c_ratio = kwargs['c_ratio']
            if 0 in self.action_counts:
                action = np.where(self.action_counts == 0)[0][0] + 1
            else:
                value = self.q + c_ratio * np.sqrt(np.log(self.counts + 1) / (self.action_counts + 1))
                action = np.argmax(value) + 1
        elif policy == 'boltzmann':
            tau = kwargs['temperature']
            p = np.exp(self.q / tau) / (np.sum(np.exp(self.q / tau)))
            action = np.random.choice([1, 2, 3], p=p.ravel())
        return action

    def train(self, play_total, policy, **kwargs):
        reward_1 = []
        reward_2 = []
        reward_3 = []
        for i in range(play_total):
            if policy == 'e_greedy':
                action = self.choose_action(policy, epsilon=kwargs['epsilon'])
            elif policy == 'ucb':
                action = self.choose_action(policy, c_ratio=kwargs['c_ratio'])
            elif policy == 'boltzmann':
                action = self.choose_action(policy, temperature=kwargs['temperature'])

            self.a = action
            # 与环境交互一次
            self.r = self.step(self.a)
            self.counts += 1
            # 更新值函数
            self.q[self.a - 1] = (self.q[self.a - 1] * self.action_counts[self.a - 1] + self.r) / (
                        self.action_counts[self.a - 1] + 1)
            self.action_counts[self.a - 1] += 1
            reward_1.append(self.q[0])
            reward_2.append(self.q[1])
            reward_3.append(self.q[2])
            self.current_cumulative_rewards += self.r
            self.cumulative_rewards_history.append(self.current_cumulative_rewards)
            self.counts_history.append(i)

    def reset(self):
        # 重置参数
        self.q = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.current_cumulative_rewards = 0.0
        self.counts = 0
        self.counts_history = []
        self.cumulative_rewards_history = []
        self.a = 1
        self.reward = 0

    def plot(self, colors, policy, style):
        # 绘制累积回报图
        plt.plot(self.counts_history, self.cumulative_rewards_history, style, color=colors, label=policy)
        plt.legend()
        plt.xlabel('n', fontsize=18)
        plt.ylabel('total rewards', fontsize=18)


if __name__ == '__main__':
    np.random.seed(0)
    k_gamble = KB_Game()
    total = 20

    # 使用 e_greedy 策略训练
    k_gamble.train(play_total=total, policy='e_greedy', epsilon=0.05)
    greedy_score = list(map(lambda x: round(x, 2), k_gamble.cumulative_rewards_history))

    k_gamble.reset()

    # 使用 boltzmann 策略训练
    k_gamble.train(play_total=total, policy='boltzmann', temperature=1)
    boltzmann_score = list(map(lambda x: round(x, 2), k_gamble.cumulative_rewards_history))

    k_gamble.reset()

    # 使用 ucb 策略训练
    k_gamble.train(play_total=total, policy='ucb', c_ratio=0.5)
    ucb_score = list(map(lambda x: round(x, 2), k_gamble.cumulative_rewards_history))

    print(sum(greedy_score), sum(boltzmann_score), sum(ucb_score))
    with open('greedy_score.txt', 'w') as f:
        f.write(str(greedy_score))
    with open('boltzmann_score.txt', 'w') as f:
        f.write(str(boltzmann_score))
    with open('ucb_score.txt', 'w') as f:
        f.write(str(ucb_score))
    print('end')

