import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import networkx as nx

# 간단한 그래프 정의 (인접 행렬)
graph = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 0]])


distance = np.array([[ 0, 40, 68,  0,  0,  0],
                    [16,  0,  0, 18,  0,  0],
                    [42,  0,  0, 43, 95,  0],
                    [ 0, 59,  5,  0,  0, 28],
                    [ 0,  0, 84,  0,  0, 73],
                    [ 0,  0,  0, 10, 73,  0]])

# 환경 설정
class GraphEnvironment:
    def __init__(self, graph, distance):
        self.graph = graph
        self.distance = distance
        self.num_nodes = graph.shape[0]
        self.current_node = 0
        self.target_node = self.num_nodes - 1

    def reset(self):
        self.current_node = 0
        return self.one_hot_encode(self.current_node)

    def step(self, action):
        if self.graph[self.current_node, action] == 1:
            next_node = action
            reward = -self.distance[self.current_node, action]
            self.current_node = next_node
        else:
            next_node = self.current_node
            reward = -100  # 벽에 부딪힌 경우 큰 음수 보상

        done = (next_node == self.target_node)
        return self.one_hot_encode(next_node), reward, done
    
    def one_hot_encode(self, node):
        state = np.zeros(self.num_nodes)
        state[node] = 1
        return state

# DQN 네트워크
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy()[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float()
            next_state = torch.from_numpy(next_state).float()
            reward = torch.tensor(reward).float()
            action = torch.tensor(action)
            done = torch.tensor(done)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state)
            target_f[0][action] = target

            # 역전파
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 학습 루프
def train_dqn(agent, env, episodes=1000, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.memorize(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            state = next_state
            if done:
                print("에피소드: {}/{}, 탐험율: {:.2}".format(e, episodes, agent.epsilon))
                break

# 환경 및 에이전트 초기화 및 학습
env = GraphEnvironment(graph, distance)
agent = DQNAgent(env.num_nodes, env.num_nodes)

train_dqn(agent, env)

def find_optimal_path(agent, env, start_node, target_node):
    env.current_node = start_node
    env.target_node = target_node
    state = env.one_hot_encode(env.current_node)  # 원-핫 인코딩된 상태 사용
    state = np.reshape(state, [1, agent.state_size])  # 적절한 형태로 변환
    optimal_path = [start_node]

    while True:
        action = agent.act(state)
        next_state, _, done = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])

        optimal_path.append(action)
        state = next_state

        if done or env.current_node == target_node:
            break

    return optimal_path


# 예시: 노드 0에서 노드 5까지의 최적 경로 찾기
start_node = 0
target_node = 5
optimal_path = find_optimal_path(agent, env, start_node, target_node)
print("최적 경로:", optimal_path)
