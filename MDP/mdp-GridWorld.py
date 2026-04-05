import numpy as np
# GridWorld MDP示例
# 这是一个简单的网格世界MDP示例，包含四个状态和四个动作。
# 该示例演示了如何定义MDP的五元组（状态、动作、奖励、折扣因子、终止状态）以及如何评估策略。
# 状态定义：
# S1: 起点，S2: 终点，S3: 陷阱，S4: 终止状态
# 动作定义：
# up: 向上移动，down: 向下移动，left: 向左移动，right: 向右移动
# 奖励定义：
# S1 → S2: +10，S3 → S4: -10，S2和S4为终止状态，无动作奖励为0
# 奖励定义：
# S1 → S2: +10，S3 → S4: -10，  # 终点S2无动作奖励为0，陷阱S4无动作奖励为0
# 折扣因子，表示未来奖励的折扣率
# gamma = 0.9
# 状态转移函数
# 根据当前状态和动作返回下一个状态和奖励
# 策略评估
# 计算固定策略的状态价值函数


# 定义MDP五元组
class GridWorldMDP:
    def __init__(self):
        self.states = [0, 1, 2, 3]  # S1=0, S2=1, S3=2, S4=3
        self.actions = ['up', 'down', 'left', 'right']
        self.rewards = {
            (0, 'right'): 10,   # S1 → S2
            (2, 'right'): -10,   # S3 → S4
            (1, 'any'): 0,      # 终点S2无动作
            (3, 'any'): 0        # 陷阱S4无动作
        }
        self.gamma = 0.9          # 折扣因子
        self.terminal_states = [1, 3]  # 终止状态（S2和S4）

    # 状态转移函数
    def transition(self, state, action):
        if state in self.terminal_states:
            return state, 0  # 终止状态不再转移

        next_state = state
        if action == 'up':
            next_state = state - 2 if state >= 2 else state
        elif action == 'down':
            next_state = state + 2 if state < 2 else state
        elif action == 'left':
            next_state = state - 1 if state % 2 != 0 else state
        elif action == 'right':
            next_state = state + 1 if state % 2 == 0 else state

        # 默认奖励-1，特殊奖励从字典中读取
        reward = self.rewards.get((state, action), -1)
        return next_state, reward

    # 策略评估：计算固定策略的状态价值函数
    def policy_evaluation(self, policy, max_iter=1000, tol=1e-6):
        V = np.zeros(len(self.states))
        for _ in range(max_iter):
            delta = 0
            for s in self.states:
                if s in self.terminal_states:
                    continue
                if isinstance(policy, dict):
                    a = policy.get(s)
                if callable(policy):
                    a = policy(s)
                s_next, r = self.transition(s, a)
                new_v = r + self.gamma * V[s_next]
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            if delta < tol:
                break
        return V

# 定义策略（随机策略）
def random_policy(state):
    return np.random.choice(['up', 'down', 'left', 'right'])

# 定义最优策略（手动设定）
optimal_policy = {
    0: 'right',  # S1 → S2
    2: 'up',     # S3 → S1 (避免掉入S4)
    1: 'any',    # 终止状态
    3: 'any'     # 终止状态
}

# 运行MDP
mdp = GridWorldMDP()

# 评估随机策略
print("随机策略的状态价值函数:")
random_v = mdp.policy_evaluation(random_policy)
print(random_v)

# 评估最优策略
print("\n最优策略的状态价值函数:")
optimal_v = mdp.policy_evaluation(optimal_policy)
print(optimal_v)
