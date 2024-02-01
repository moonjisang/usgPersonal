import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple, deque
from tqdm import tqdm
from pymongo import MongoClient
import math
import requests
import json


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

# print("인접 행렬:")
# print(adjacency_matrix)
# print("\n거리 행렬:")
# print(distance_matrix)
# print("\n풍향각 행렬:")
# print(wind_angle_matrix)

# NumPy 배열을 리스트로 변환하여 JSON으로 저장할 수 있게 함
adjacency_list = adjacency_matrix.tolist()
distance_list = distance_matrix.tolist()
wind_angle_list = wind_angle_matrix.tolist()

# 데이터를 JSON 형태로 저장
graph_data = {
    'adjacency_matrix': adjacency_list,
    'distance_matrix': distance_list,
    'wind_angle_matrix': wind_angle_list
}

# 빈 값은 NaN으로 채우기
distance_matrix = np.where(np.array(distance_matrix) == 0.0, np.nan, distance_matrix)
wind_angle_matrix = np.where(np.array(wind_angle_matrix) == 0.0, np.nan, wind_angle_matrix)

# 모든 행렬을 함께 그리기
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# adjacency_matrix 그리기
axes[0].imshow(adjacency_matrix, cmap='binary', interpolation='none')
axes[0].set_title('Adjacency Matrix')
axes[0].set_xlabel('Nodes')
axes[0].set_ylabel('Nodes')

# distance_matrix 그리기
im1 = axes[1].imshow(distance_matrix, cmap='Oranges', interpolation='none')
axes[1].set_title('Distance Matrix')
axes[1].set_xlabel('Nodes')
axes[1].set_ylabel('Nodes')
fig.colorbar(im1, ax=axes[1], label='Distance')

# wind_angle_matrix 그리기
#im2 = axes[2].imshow(wind_angle_matrix, cmap='Blues', interpolation='none')
#axes[2].set_title('Wind Angle Matrix')
#axes[2].set_xlabel('Nodes')
#axes[2].set_ylabel('Nodes')
#fig.colorbar(im2, ax=axes[2], label='Wind Angle')

plt.tight_layout()
plt.show()


# JSON 파일로 저장
with open('graph_data.json', 'w') as json_file:
    json.dump(graph_data, json_file, indent=4)