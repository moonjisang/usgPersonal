import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from tqdm import tqdm

# CUDA 지원 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device : ', device)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, epsilon_decay, epsilon_min, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.policy_net = DQN(state_size, hidden_size, action_size).to(device)  # GPU로 이동
        self.target_net = DQN(state_size, hidden_size, action_size).to(device)  # GPU로 이동
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, visited):
        if random.random() < self.epsilon:
            # 방문하지 않은 노드 중에서 무작위로 선택
            possible_actions = [i for i in range(self.action_size) if visited[i] == 0]
            if not possible_actions:  # 가능한 행동이 없는 경우
                return random.randrange(self.action_size)
            return random.choice(possible_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = self.policy_net(state)
                # 방문하지 않은 노드에 대해서만 q 값을 고려
                q_values = q_values.cpu().numpy().flatten()
                q_values[visited == 1] = -float('inf')
                return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(states, dtype=torch.float32).to(device)  # GPU로 이동
        actions = torch.tensor(actions, dtype=torch.int64).to(device)  # GPU로 이동
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # GPU로 이동
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)  # GPU로 이동
        dones = torch.tensor(dones, dtype=torch.uint8).to(device)  # GPU로 이동

        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1))

        target_q_values = self.target_net(next_states)
        max_target_q_values = target_q_values.max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * self.gamma * max_target_q_values

        loss = nn.functional.smooth_l1_loss(q_value, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 간단한 그래프 정의 (인접 행렬)
graph = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 0]])

# 거리 정보 (1에서 100까지)
distance = np.array([[ 0, 40, 68,  0,  0,  0],
 [16,  0,  0, 18,  0,  0],
 [42,  0,  0, 43, 95,  0],
 [ 0, 59,  5,  0,  0, 28],
 [ 0,  0, 84,  0,  0, 73],
 [ 0,  0,  0, 10, 73,  0]])

# 풍향각 정보 (0에서 360까지)
wind_direction = np.array([[  0,  89,   5,   0,   0,   0],
 [208,   0,   0, 166,   0,   0],
 [  4,   0,   0, 230, 163,   0],
 [  0, 143,  30,   0,   0, 131],
 [  0,   0, 303,   0,   0, 165],
 [  0,   0,   0,  42,  53,   0]])

def take_action(state, action, visited, goal_node):
    current_node = state.argmax()

    # 이미 방문한 노드인지 확인
    if visited[action] == 1:
        reward = -1  # 음의 보상을 부여
        next_state = state  # 상태 변경 없음
        done = False
    else:
        # 거리와 풍향각에 따라 보상 계산
        distance_reward = 0 if distance[current_node, action] == 0 else 1 / (distance[current_node, action] + 1)
        wind_direction_reward = 0 if wind_direction[current_node, action] == 0 else 1 / (wind_direction[current_node, action] + 1)

        # 이동 가능한 경우 보상 계산
        if graph[current_node, action] == 1:
            reward = distance_reward + wind_direction_reward
            next_state = np.zeros_like(state)
            next_state[action] = 1
            visited[action] = 1  # 방문한 노드 표시
        else:
            reward = -1  # 이동 불가능한 경우 음의 보상
            next_state = state

        # 목적지에 도달했는지 확인
        done = (action == goal_node)

    return next_state, reward, done, visited



# DQN 에이전트 초기화
state_size = 6
action_size = 6
hidden_size = 24
lr = 0.001
gamma = 0.95
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32

agent = DQNAgent(state_size, action_size, hidden_size, lr, gamma, epsilon_decay, epsilon_min, batch_size)

# 학습을 진행
num_episodes = 500
goal_node = 5  # 예시 목적지 노드 설정

for episode in range(num_episodes):
    state = np.zeros(state_size)  # 초기 상태 설정
    state[0] = 1  # 첫 번째 노드에서 시작
    visited = np.zeros(state_size)  # 방문한 노드 추적
    visited[0] = 1
    done = False
    total_reward = 0

    max_steps_per_episode = 50

    for step in tqdm(range(max_steps_per_episode)):
        action = agent.select_action(state, visited)  # 방문한 노드 고려
        next_state, reward, done, visited = take_action(state, action, visited, goal_node)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    agent.replay()
    agent.update_target_model()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Goal Reached: {done}")

# 학습된 모델 저장
torch.save(agent.policy_net.state_dict(), "dqn_graph_model.pth")

