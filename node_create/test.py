import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from tqdm import tqdm
from pymongo import MongoClient
import math, requests

# MongoDB Connection
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
db = client['usg']
collectionGps = db['gps']

# MongoDB에서 노드 데이터 가져오기
nodes_data = list(collectionGps.find({}))

# 날씨 데이터 가져오기
response = requests.get('http://localhost:5000/get_weather_data')
wind_data = response.json()
wind_direction = float(wind_data[0]['obsrValue'])
wind_speed = float(wind_data[1]['obsrValue'])

# 노드 수 계산
num_nodes = len(nodes_data)

# 인접 행렬, 거리, 풍향각 배열 초기화
adjacency_matrix = np.zeros((num_nodes, num_nodes))
distance_matrix = np.zeros((num_nodes, num_nodes))
wind_angle_matrix = np.zeros((num_nodes, num_nodes))

def get_distance(lon1, lat1, lon2, lat2):
    # Radius of the Earth in km
    R = 6378.137
    # Converting degrees to radians
    dLon = math.radians(lon2 - lon1)
    dLat = math.radians(lat2 - lat1)
    # Haversine formula
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c

    return d * 1000  # Return distance in meters

#엣지의 방향 계산
def calculate_bearing(lat1, lng1, lat2, lng2):
    # 모든 위도, 경도 값을 라디안으로 변환
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    # 경도 차이 계산
    d_lng = lng2 - lng1

    # 방위각 계산
    bearing = math.atan2(math.sin(d_lng) * math.cos(lat2),
                         math.cos(lat1) * math.sin(lat2) -
                         math.sin(lat1) * math.cos(lat2) * math.cos(d_lng))

    # 라디안에서 도로 변환 후, 양수로 만들기
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing

#엣지와 풍향 사이 각 계산
def calculate_angle_difference(bearing, wind_direction):
    # 두 방향 사이의 각도 차이 계산
    difference = abs(bearing - wind_direction)
    # 각도 차이가 180도를 넘지 않도록 조정
    difference = min(difference, 360 - difference)
    return difference


# 데이터를 배열에 채우기
for node in nodes_data:
    node_index = node['nodeIndex']
    for edge in node['nodeEdge']:
        target_node = collectionGps.find_one({'nodeIndex': edge})
        if target_node:
            distance = get_distance(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
            bearing = calculate_bearing(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
            wind_angle = calculate_angle_difference(bearing, wind_direction)
            
            adjacency_matrix[node_index, edge] = 1
            adjacency_matrix[edge, node_index] = 1
            
            distance_matrix[node_index, edge] = distance
            wind_angle_matrix[node_index, edge] = wind_angle
            reverse_wind_angle = (wind_angle + 180) % 360
            wind_angle_matrix[edge, node_index] = reverse_wind_angle
#####################################################################################################################
# CUDA 지원 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device : ', device)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # 입력 레이어
        self.fc_input = nn.Linear(input_size, hidden_sizes[0])
        
        # 은닉 레이어
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # 출력 레이어
        self.fc_output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # 입력 레이어
        x = torch.relu(self.fc_input(x))
        
        # 은닉 레이어
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        
        # 출력 레이어
        x = self.fc_output(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, epsilon_decay, epsilon_min, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1500)
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
            current_node = np.argmax(state)
            possible_actions = [i for i in range(self.action_size) if adjacency_matrix[current_node, i] == 1 and visited[i] == 0]
            if not possible_actions:
                return None, True  # 가능한 행동이 없을 경우 None 반환
            return random.choice(possible_actions), False
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = self.policy_net(state)
                q_values = q_values.cpu().numpy().flatten()
                q_values[visited == 1] = -float('inf')
                return np.argmax(q_values), False



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(device)

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

# 간단한 그래프, 거리, 풍향각 정보 정의
graph = np.array(adjacency_matrix)
distance = np.array(distance_matrix)
wind_direction = np.array(wind_angle_matrix)
max_distance_threshold = np.sum(distance_matrix) / 2
exponent = 1.5  # 조정 가능

# 보상 계산 최적화
def calculate_reward(current_node, action, visited, goal_node, total_distance):
    if visited[action] or distance_matrix[current_node, action] == 0:
        return -10, False, total_distance  # 잘못된 행동에 대한 큰 음의 보상

    step_distance = distance_matrix[current_node, action]
    total_distance += step_distance

    if action == goal_node:
        # 목적지에 도달했을 때의 큰 보상
        reward = 1000 + calculate_path_efficiency_reward(total_distance)
        done = True
    else:
        # 다른 경로 선택 시 작은 패널티
        reward = -12
        done = False

    return reward, done, total_distance


def calculate_path_efficiency_reward(total_distance):
    # max_distance_threshold보다 total_distance가 작을수록 보상 증가
    distance_diff = (max_distance_threshold - total_distance) * 0.1
    normalized_reward = np.exp(distance_diff * exponent)

    # 보상의 최소값을 0으로 설정
    normalized_reward = max(normalized_reward, 0)
    return normalized_reward


# DQN 에이전트 초기화
state_size = 16
action_size = 16
hidden_size = [32, 64, 32]
lr = 0.0005
gamma = 0.95
epsilon_decay = 0.998
epsilon_min = 0.01
batch_size = 64

agent = DQNAgent(state_size, action_size, hidden_size, lr, gamma, epsilon_decay, epsilon_min, batch_size)

# 학습을 진행
num_episodes = 3000
max_steps_per_episode = 1000

for episode in range(num_episodes):
    # 랜덤으로 시작 노드와 도착 노드 선택
    start_node = random.randint(0, num_nodes - 1)
    goal_node = random.randint(0, num_nodes - 1)
    
    # 시작 노드와 도착 노드가 같으면 다시 선택
    while start_node == goal_node:
        goal_node = random.randint(0, num_nodes - 1)

    state = np.zeros(num_nodes)
    state[start_node] = 1
    visited = np.zeros(num_nodes)
    visited[start_node] = 1
    total_reward= 0
    total_distance = 0

    for step in range(max_steps_per_episode):
        action, done = agent.select_action(state, visited)
        if action is None:
            break

        reward, done, total_distance = calculate_reward(state.argmax(), action, visited, goal_node, total_distance)
        total_reward += reward

        next_state = np.zeros_like(state)
        next_state[action] = 1
        visited[action] = 1

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    agent.replay()
    agent.update_target_model()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Goal Reached: {done}")

# 학습된 모델 저장
torch.save(agent.policy_net.state_dict(), "dqn_graph_model.pth")