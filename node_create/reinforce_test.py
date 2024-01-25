import torch
import numpy as np

# 학습된 모델 로드
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("trained_dqn_model.pth"))
model.eval()

# 시작 노드와 목적지 노드 설정
start_node_index = 0
destination_node_index = 15

# 최적 경로 추적
current_node_index = start_node_index
optimal_path = [current_node_index]

while current_node_index != destination_node_index:
    current_state = np.array([current_node_index], dtype=np.float32)
    q_values = model(torch.tensor(current_state))
    action = np.argmax(q_values.detach().numpy())
    next_node_index = graph.edges[current_node_index][action]
    optimal_path.append(next_node_index)
    current_node_index = next_node_index

print("최적 경로:", optimal_path)
