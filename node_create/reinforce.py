import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pymongo import MongoClient
import requests

# MongoDB Connection
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
db = client['usg']
collectionGps = db['gps']

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.distances = {}
        self.wind_angles = {}  # 바람 방향과의 각도 차이를 저장

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance, wind_angle):
        self.edges.setdefault(from_node, [])
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance
        self.wind_angles[(from_node, to_node)] = wind_angle
        # 반대 방향에 대한 엣지도 추가
        self.edges.setdefault(to_node, [])
        self.edges[to_node].append(from_node)
        self.distances[(to_node, from_node)] = distance
        # 반대 방향의 바람 각도 차이 계산
        reverse_wind_angle = 180 - wind_angle
        self.wind_angles[(to_node, from_node)] = reverse_wind_angle

def build_graph(wind_direction):
    graph = Graph()
    nodes_data = list(collectionGps.find({}))

    for node in nodes_data:
        graph.add_node(node['nodeIndex'])

        for edge in node['nodeEdge']:
            target_node = collectionGps.find_one({'nodeIndex': edge})
            if target_node:
                distance = get_distance(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
                bearing = calculate_bearing(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
                wind_angle = calculate_angle_difference(bearing, wind_direction)
                # 양방향 엣지와 바람 각도 차이를 그래프에 추가
                graph.add_edge(node['nodeIndex'], edge, distance, wind_angle)

    # 중복된 엣지 및 정보 정리
    for from_node in graph.edges:
        unique_to_nodes = list(set(graph.edges[from_node]))
        graph.edges[from_node] = unique_to_nodes

    return graph

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.ModuleList([nn.Linear(64, dim) for dim in output_dim])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x_list = [fc(x) for fc in self.fc3]
        return x_list

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

# 엣지의 방향 계산
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

# 엣지와 풍향 사이 각 계산
def calculate_angle_difference(bearing, wind_direction):
    # 두 방향 사이의 각도 차이 계산
    difference = abs(bearing - wind_direction)
    # 각도 차이가 180도를 넘지 않도록 조정
    difference = min(difference, 360 - difference)
    return difference

# 환경 클래스 정의
class FlightEnvironment:
    def __init__(self, graph, start_node_index):
        self.graph = graph
        self.current_node_index = start_node_index
        self.done = False

    def reset(self):
        self.current_node_index = 0
        self.done = False

    def step(self, action):
        next_node_index = self.graph.edges[self.current_node_index][action]

        distance = self.graph.distances[(self.current_node_index, next_node_index)]
        wind_angle = self.graph.wind_angles[(self.current_node_index, next_node_index)]
        reward = calculate_reward(distance, wind_angle)

        self.current_node_index = next_node_index

        if self.current_node_index not in self.graph.nodes:
            self.done = True

        return next_node_index, reward, self.done

def calculate_reward(distance, wind_angle):
    # 거리와 풍향 각도에 따른 보상 계산 함수 수정
    max_distance = 2000  # 최대 거리 (조절 필요)
    # 거리에 따른 보상 계산 (70%)
    distance_reward = 1000 * (1 - distance / max_distance) * 0.7

    # 풍향 각도에 따른 보상 계산 (30%)
    wind_angle_reward = 1000 * (1 - wind_angle / 180) * 0.3

    # 총 보상은 거리 보상과 풍향 보상을 합산
    total_reward = distance_reward + wind_angle_reward

    return total_reward

# DQN 학습 함수 정의
def train_dqn(model, num_episodes, environment):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        print('episode ; ', episode)
        environment.reset()
        current_state = np.array([environment.current_node_index], dtype=np.float32)

        done = False
        total_reward = 0

        while not done:
            if environment.current_node_index not in environment.graph.nodes:
                done = True
                break

            # 가능한 행동 중 무작위로 선택
            q_values = model(torch.tensor(current_state))
            print('q_values : ', q_values)

            # q_values를 PyTorch 텐서로 변환
            q_values_tensor = torch.tensor(q_values, dtype=torch.float32)
            
            # 가능한 행동 중 무작위로 선택
            possible_actions = range(len(q_values))
            # 올바른 차원(dim)을 지정하여 softmax를 계산
            softmax_values = torch.softmax(q_values_tensor, dim=0).detach().numpy()
            action = np.random.choice(possible_actions, p=softmax_values)
            print('action : ', action)
            
            # 현재 노드와 연결된 다음 노드들을 가져옴
            next_node_indices = environment.graph.edges[environment.current_node_index]
            
            # 다음 노드들과 관련된 정보 계산
            distances = [environment.graph.distances[(environment.current_node_index, next_node)] for next_node in next_node_indices]
            wind_angles = [environment.graph.wind_angles[(environment.current_node_index, next_node)] for next_node in next_node_indices]
            
            # 보상 계산
            rewards = [calculate_reward(dist, wind_angle) for dist, wind_angle in zip(distances, wind_angles)]
            
            # 다음 노드 선택 (보상이 가장 높은 노드 선택)
            next_node_index = next_node_indices[np.argmax(rewards)]
            
            total_reward += max(rewards)

            # 다음 상태 설정
            next_state = np.array([next_node_index], dtype=np.float32)

            current_state = next_state

        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    response = requests.get('http://localhost:5000/get_weather_data')  # Flask 서버의 URL을 사용합니다.
    wind_direction = 0
    if response.status_code == 200:
        wind_data = response.json()
        wind_direction = float(wind_data[0]['obsrValue'])  # 바람 방향 (예: 북동풍)

    graph = build_graph(wind_direction)
    print(' gr test : ', np.array(graph))
    # 그래프의 노드 정보 출력
    print("Nodes:")
    for node in graph.nodes:
        print(node)

    # 그래프의 엣지 정보 출력
    print("\nEdges:")
    for from_node, to_nodes in graph.edges.items():
        for to_node in to_nodes:
            distance = graph.distances[(from_node, to_node)]
            wind_angle = graph.wind_angles[(from_node, to_node)]
            print(f"From Node {from_node} to Node {to_node}, Distance: {distance}, Wind Angle: {wind_angle}")

    # DQN 모델 및 학습 파라미터 설정
    input_dim = 1  # 현재 상태는 현재 노드 인덱스 하나만 사용
    output_dim = [len(graph.edges[node]) for node in graph.nodes]  # 각 노드에 따른 가능한 행동의 개수로 설정
    num_episodes = 100  # 학습 에피소드 수

    # DQN 모델 초기화
    dqn_model = DQN(input_dim, output_dim)

    # 환경 초기화 (시작 노드 인덱스 설정)
    start_node_index = 0
    environment = FlightEnvironment(graph, start_node_index)

    # DQN 학습 실행
    train_dqn(dqn_model, num_episodes, environment)
    # 학습이 완료된 후에 모델 저장
    torch.save(dqn_model.state_dict(), "trained_dqn_model.pth")