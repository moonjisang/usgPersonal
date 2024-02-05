from pymongo import MongoClient
from flask import Blueprint, jsonify, request
import math
from weather import get_weather_data
import requests

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

pathFinding_blueprint = Blueprint('pathFinding', __name__)


# MongoDB Connection
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
db = client['usg']
collectionGps = db['gps']
collectionStartGps = db['startGps']
collectionEndGps = db['endGps']

# 시작 노드 및 목표 노드 가져오기
start_node_data = collectionStartGps.find_one({})
goal_node_data = collectionEndGps.find_one({})

@pathFinding_blueprint.route('/calculate_shortest_path', methods=['GET'])
def calculate_shortest_path():
    try:        
        # MongoDB에서 노드 데이터 가져오기
        nodes_data = list(collectionGps.find({}))

        # 노드 수 계산
        num_nodes = len(nodes_data)

        # 인접 행렬, 거리, 풍향각 배열 초기화
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        distance_matrix = np.zeros((num_nodes, num_nodes))

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

        # 데이터를 배열에 채우기
        for node in nodes_data:
            node_index = node['nodeIndex']
            for edge in node['nodeEdge']:
                target_node = collectionGps.find_one({'nodeIndex': edge})
                if target_node:
                    distance = get_distance(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
                    
                    adjacency_matrix[node_index, edge] = 1
                    adjacency_matrix[edge, node_index] = 1
                    
                    distance_matrix[node_index, edge] = distance

        ##############################################################################################################
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
                x = self.fc3(x)
                return x
            
        def find_shortest_path_dqn(start_node, goal_node, dqn_model, distance_matrix, adjacency_matrix, max_steps=100):
            current_node = start_node
            dqn_model.eval()  # 모델을 평가 모드로 설정
            path = [current_node]  # 경로 초기화
            total_distance = 0
            coordinates = []

            coordinates.append({
                        "lng": nodes_data[current_node]['lng'],
                        "lat": nodes_data[current_node]['lat']
                    })

            with torch.no_grad():  # 그래디언트 계산을 비활성화
                for _ in range(max_steps):
                    state = torch.zeros(num_nodes, dtype=torch.float32)
                    state[current_node] = 1
                    state = state.unsqueeze(0)  # 배치 차원 추가

                    q_values = dqn_model(state)
                    # 현재 노드에서 이동 가능한 노드들을 찾되, 현재 노드 및 이미 경로에 있는 노드는 제외
                    available_actions = np.where((adjacency_matrix[current_node] > 0) & (~np.isin(np.arange(num_nodes), path)))[0]

                    if len(available_actions) == 0:  # 이동 가능한 노드가 없는 경우
                        break  # 더 이상 진행할 수 없으므로 반복 종료

                    # 가능한 행동 중 최대 Q 값을 갖는 행동 선택
                    q_values_filtered = q_values[0, available_actions]
                    next_node = available_actions[q_values_filtered.argmax().item()]

                    path.append(next_node)
                    total_distance += distance_matrix[current_node, next_node]

                    # 다음 노드의 좌표를 coordinates에 추가
                    coordinates.append({
                        "lng": nodes_data[next_node]['lng'],
                        "lat": nodes_data[next_node]['lat']
                    })

                    if next_node == goal_node:
                        print(f"목표 노드 {goal_node}에 도달했습니다.")
                        break

                    current_node = next_node

            return path, total_distance, coordinates

        # 하이퍼파라미터 설정
        input_size = num_nodes  # 입력 크기는 노드 수와 같음
        output_size = num_nodes  # 출력 크기도 노드 수와 같음

        # 학습된 모델 로드
        dqn_loaded = DQN(input_size, output_size)
        dqn_loaded.load_state_dict(torch.load('dqn_model.pth'))

        # 특정 노드에서 노드까지의 최단 경로 검출
        start_node = start_node_data['nodeIndex']
        goal_node = goal_node_data['nodeIndex']

        path, total_distance, coordinates = find_shortest_path_dqn(start_node, goal_node, dqn_loaded, distance_matrix, adjacency_matrix)

        print(f"경로: {path}")
        print(f"총 거리: {total_distance} 미터")
        print(f"좌표 : {coordinates} ")

        # 데이터를 JSON으로 직렬화할 때 int64를 int로 변환
        path = list(map(int, path))
        total_distance = int(total_distance)

        # 최단 경로와 좌표값을 JSON으로 반환
        return jsonify({"shortest_path": path, "coordinates": coordinates, "total_distance": total_distance})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

