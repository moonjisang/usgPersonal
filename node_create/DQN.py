import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from tqdm import tqdm
from pymongo import MongoClient
import math
import matplotlib.pyplot as plt
import networkx as nx
import subprocess
from flask import Blueprint, jsonify

DQN_blueprint = Blueprint('DQN', __name__)

@DQN_blueprint.route('/train_dqn', methods=['GET'])
def train_dqn_route():
    try:
        # CUDA 지원 확인 및 GPU 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:', device)

        # MongoDB Connection
        client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
        db = client['usg']
        collectionGps = db['gps']
        collectionStartGps = db['startGps']
        collectionEndGps = db['endGps']

        # 시작 노드 및 목표 노드 가져오기
        start_node_data = collectionStartGps.find_one({})
        goal_node_data = collectionEndGps.find_one({})

        # MongoDB에서 노드 데이터 가져오기
        nodes_data = list(collectionGps.find({}))

        # 노드 수 및 인접 행렬, 거리 행렬 초기화
        num_nodes = len(nodes_data)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        distance_matrix = np.zeros((num_nodes, num_nodes))

        # 거리 계산 함수
        def get_distance(lon1, lat1, lon2, lat2):
            R = 6378.137  # 지구 반지름(km)
            dLon = math.radians(lon2 - lon1)
            dLat = math.radians(lat2 - lat1)
            a = math.sin(dLat/2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            d = R * c
            return d * 1000  # meters

        # 데이터 초기화
        for node in nodes_data:
            node_index = node['nodeIndex']
            for edge in node['nodeEdge']:
                target_node = collectionGps.find_one({'nodeIndex': edge})
                if target_node:
                    distance = get_distance(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
                    adjacency_matrix[node_index, edge] = 1
                    distance_matrix[node_index, edge] = distance

        # DQN 모델 정의
        class DQN(nn.Module):
            def __init__(self, input_size, output_size):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, output_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        # 경험 리플레이 버퍼 정의
        class ReplayBuffer:
            def __init__(self, capacity):
                self.buffer = deque(maxlen=capacity)
            
            def push(self, state, action, next_state, reward, done):
                self.buffer.append((state, action, next_state, reward, done))
            
            def sample(self, batch_size):
                batch = random.sample(self.buffer, batch_size)
                return batch  # 리스트의 리스트가 아닌, 각 요소가 (state, action, next_state, reward, done) 튜플인 리스트 반환


            def __len__(self):
                return len(self.buffer)

        # DQN 학습 함수
        def train_dqn(dqn, target_dqn, replay_buffer, optimizer, gamma, batch_size):
            if len(replay_buffer) < batch_size:
                return

            samples = replay_buffer.sample(batch_size)
            states, actions, next_states, rewards, dones = zip(*samples)

            states = torch.tensor(np.vstack(states), dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            # DQN 모델을 사용하여 Q 값 계산
            q_values = dqn(states)
            next_q_values = target_dqn(next_states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1).values
            expected_q_value = rewards + gamma * next_q_value * (1 - dones)

            # 손실 계산 및 역전파
            loss = nn.MSELoss()(q_value, expected_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 유클리드 거리 계산 함수
        def euclidean_distance(x1, y1, x2, y2):
            return get_distance(x1, y1, x2, y2) * 0.3

        # 모델 및 최적화기 초기화
        dqn = DQN(num_nodes, num_nodes).to(device)
        target_dqn = DQN(num_nodes, num_nodes).to(device)
        target_dqn.load_state_dict(dqn.state_dict())
        optimizer = optim.Adam(dqn.parameters(), lr=0.001)
        replay_buffer = ReplayBuffer(10000)

        # 학습 파라미터 설정
        gamma = 0.99
        epsilon = 0.4
        batch_size = 64
        target_update_freq = 100
        num_episodes = 200

        # 학습 루프
        for episode in range(num_episodes):
            # 초기 상태 및 목표 노드 설정
            current_node = start_node_data['nodeIndex'] #random.randint(0, num_nodes - 1)
            goal_node = goal_node_data['nodeIndex'] #random.randint(0, num_nodes - 1)
            while goal_node == current_node:  # 시작 노드와 목표 노드가 같지 않도록 합니다.
                goal_node = random.randint(0, num_nodes - 1)

            state = np.zeros(num_nodes)
            state[current_node] = 1
            
            total_reward = 0
            done = False
            max_steps = 300

            for step in range(max_steps):
                connected_nodes = np.where((adjacency_matrix[current_node] == 1) & (np.arange(num_nodes) != current_node))[0]
                if random.random() < epsilon:
                    action = np.random.choice(connected_nodes)
                else:
                    with torch.no_grad():
                        q_values = dqn(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                        q_values_connected = q_values[0, connected_nodes]
                        action = connected_nodes[q_values_connected.argmax().item()]

                next_node = action
                next_state = np.zeros(num_nodes)
                next_state[next_node] = 1

                # 휴리스틱 정보를 사용한 보상 계산
                heuristic_reward = euclidean_distance(
                    nodes_data[next_node]['lng'], nodes_data[next_node]['lat'],
                    nodes_data[goal_node]['lng'], nodes_data[goal_node]['lat']
                )
                reward = - heuristic_reward

                if next_node == goal_node:
                    reward += 100  # 목표 노드에 도달했을 때의 추가 보상
                    done = True

                replay_buffer.push(state, action, next_state, reward, done)
                
                state = next_state
                current_node = next_node
                
                train_dqn(dqn, target_dqn, replay_buffer, optimizer, gamma, batch_size)
                
                total_reward += reward

                if done:
                    break

            if episode % target_update_freq == 0:
                target_dqn.load_state_dict(dqn.state_dict())
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Steps: {step + 1}")
            #print_gpu_usage()

        # 모델 저장
        torch.save(dqn.state_dict(), 'dqn_model.pth')
        return jsonify({"message": "DQN 모델 학습이 완료되었습니다."})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

