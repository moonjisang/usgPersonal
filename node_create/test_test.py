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

# CUDA 지원 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device : ', device)

# 저장된 모델 파일 로드
state_size = 6  # 적절한 state_size 값을 설정합니다.
action_size = 6  # 적절한 action_size 값을 설정합니다.
hidden_size = 24  # 적절한 hidden_size 값을 설정합니다.

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

model = DQN(state_size, hidden_size, action_size).to(device)
model.load_state_dict(torch.load("dqn_graph_model.pth", map_location=device))
model.eval()  # 모델을 평가 모드로 설정


# 그래프 정의
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


# 최적 경로 찾기 함수 정의
def find_optimal_path(start_node, goal_node, max_steps=100):
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
            # 이미 방문한 노드는 제외
            q_values[visited == 1] = -float('inf')
            valid_actions = np.where(graph[current_node] == 1)[0]  # 현재 노드와 연결된 유효한 행동 찾기
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
start_node = 5  # 시작 노드 설정
goal_node = 0   # 목표 노드 설정

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
pos = nx.spring_layout(G)  # 레이아웃 정의
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold')

# 간선 특성 표시 (거리와 풍향각)
edge_labels = {(start, end): f"D:{G.edges[start, end]['distance']}\nW:{G.edges[start, end]['wind_direction']}" for start, end in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# 그래프 표시
plt.show()