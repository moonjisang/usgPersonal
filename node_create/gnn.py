import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import math
from pymongo import MongoClient

# MongoDB Connection
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
db = client['usg']
collectionGps = db['gps']
collectionStartGps = db['startGps']
collectionEndGps = db['endGps']

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
                graph.add_edge(node['nodeIndex'], edge, distance, wind_angle)

    return graph


#효율성 계산 함수
def calculate_efficiency_score(distance, angle_difference):
    # 거리와 각도 차이에 따라 점수를 계산합니다.
    score = (180 - angle_difference) - distance
    return score

#GNN 모델 정의
class GNNModel(MessagePassing):
    def __init__(self):
        super(GNNModel, self).__init__(aggr='add')  # "add" aggregation.

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        efficiency_score = calculate_efficiency_score(edge_attr[0], edge_attr[1])
        return x_j + efficiency_score

    def update(self, aggr_out):
        return aggr_out

#데이터 준비
node_features = torch.tensor([[0, 35.1541, 128.0928], [1, 35.1550, 128.0935]], dtype=torch.float)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
edge_attr = torch.tensor([
    [calculate_efficiency_score(10, 30)],  # 예시 거리 및 각도
    [calculate_efficiency_score(10, 150)]
], dtype=torch.float)

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

#모델 학습
model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    # 여기에서 손실 함수를 정의하고, loss.backward() 및 optimizer.step()을 호출해야 합니다.
