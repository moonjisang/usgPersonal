import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math, requests
from pymongo import MongoClient

# MongoDB Connection
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
db = client['usg']
collectionGps = db['gps']
collectionStartGps = db['startGps']
collectionEndGps = db['endGps']

# MongoDB에서 노드 데이터 가져오기
nodes_data = list(collectionGps.find({}))

response = requests.get('http://localhost:5000/get_weather_data')  # Flask 서버의 URL을 사용합니다.
wind_data = response.json()
wind_direction = float(wind_data[0]['obsrValue'])  # 바람 방향 (예: 북동풍)

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


#adjacency_matrix
#distance_matrix
#wind_angle_matrix
            

# CUDA 지원 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device : ', device)

# 저장된 모델 파일 로드
state_size = 16  # 적절한 state_size 값을 설정합니다.
action_size = 16  # 적절한 action_size 값을 설정합니다.
hidden_size = [32, 64, 32]  # 적절한 hidden_size 값을 설정합니다.

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

model = DQN(state_size, hidden_size, action_size).to(device)
model.load_state_dict(torch.load("dqn_graph_model.pth", map_location=device))
model.eval()  # 모델을 평가 모드로 설정


# 그래프 정의
graph = np.array(adjacency_matrix)

# 거리 정보 (1에서 100까지)
distance = np.array(distance_matrix)

# 풍향각 정보 (0에서 360까지)
wind_direction = np.array(wind_angle_matrix)


# 최적 경로 찾기 함수 정의
def find_optimal_path(start_node, goal_node, max_steps=500):
    current_node = start_node
    path = [current_node]
    visited = np.zeros(state_size)  # 방문한 노드 추적
    visited[current_node] = 1

    step = 0
    while current_node != goal_node and step < max_steps:
        state = np.zeros(state_size)
        state[current_node] = 1  # 현재 노드를 상태로 설정
        state = torch.tensor(state, dtype=torch.float32).to(device)

        # 모델을 사용하여 다음 행동 선택
        with torch.no_grad():
            q_values = model(state)
            q_values = q_values.cpu().numpy().flatten()  # 모델 결과를 1D 배열로 변환
            print('q_values ; ', q_values)
            # 이미 방문한 노드는 제외
            q_values[visited == 1] = -float('inf')
            valid_actions = np.where(graph[current_node] == 1)[0]  # 현재 노드와 연결된 유효한 행동 찾기
            print('valid_actions :  ', valid_actions)
            valid_q_values = q_values[valid_actions]  # 유효한 행동에 대한 Q 값만 선택
            if len(valid_q_values) == 0:  # 더 이상 갈 곳이 없는 경우
                break
            action = valid_actions[np.argmax(valid_q_values)]  # Q 값이 가장 큰 행동 선택

        next_node = action  # 다음 노드 선택
        if visited[next_node] == 1:  # 이미 방문한 노드인 경우
            break
        path.append(next_node)  # 경로에 다음 노드 추가
        current_node = next_node  # 현재 노드 업데이트
        visited[current_node] = 1
        step += 1

    return path


# 예시: 시작 노드와 목표 노드를 설정하여 최적 경로 찾기
start_node = 0  # 시작 노드 설정
goal_node = 15   # 목표 노드 설정

print('시작')
optimal_path = find_optimal_path(start_node, goal_node)
print("최적 경로:", optimal_path)

# networkx 그래프로 변환 및 시각화
G = nx.from_numpy_matrix(graph)

# 각 간선에 대한 거리와 풍향각 정보를 추가 (예를 들어, 'weight'와 'wind' 속성을 사용)
for i, (start, end) in enumerate(G.edges()):
    G.edges[start, end]['distance'] = distance[start, end]
    G.edges[start, end]['wind_direction'] = wind_direction[start, end]

# 그래프 시각화
pos = nx.kamada_kawai_layout(G)  # 레이아웃 정의
node_colors = ['lightblue' if node not in optimal_path else 'red' for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_color='black', font_weight='bold')

# 간선 특성 표시 (거리와 풍향각)
edge_labels = {(start, end): f"D:{G.edges[start, end]['distance']}" for start, end in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# 그래프 표시
plt.show()

