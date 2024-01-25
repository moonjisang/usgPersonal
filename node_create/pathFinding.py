from pymongo import MongoClient
from flask import Blueprint, jsonify, request
import math
from weather import get_weather_data
import requests

pathFinding_blueprint = Blueprint('pathFinding', __name__)


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


def dijkstra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges.get(min_node, []):
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path

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

    return graph

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


@pathFinding_blueprint.route('/calculate_shortest_path', methods=['GET'])
def calculate_shortest_path():
    try:        
        response = requests.get('http://localhost:5000/get_weather_data')  # Flask 서버의 URL을 사용합니다.
        if response.status_code == 200:
            wind_data = response.json()
            # 가정된 노드의 좌표와 바람 방향
            node1_lat, node1_lng = 35.1541, 128.0928  # 노드 1의 좌표
            node2_lat, node2_lng = 35.1550, 128.0935  # 노드 2의 좌표
            wind_direction = float(wind_data[0]['obsrValue'])  # 바람 방향 (예: 북동풍)

            # 엣지의 방위각 계산
            bearing = calculate_bearing(node1_lat, node1_lng, node2_lat, node2_lng)

            # 각도 차이 계산
            angle_difference = calculate_angle_difference(bearing, wind_direction)
            print("엣지의 방향:", bearing)
            print("바람 방향과 엣지 방향 사이의 각도 차이:", angle_difference)
            
        else:
            return "Failed to get wind data"

        graph = build_graph(wind_direction)
        # Assuming we want to find the shortest path from node index 0 to node index 5
        start_node_data = collectionStartGps.find_one({})
        end_node_data = collectionEndGps.find_one({})
        if start_node_data:
            start_node = start_node_data['nodeIndex']
        if end_node_data:
            end_node = end_node_data['nodeIndex']

        distances, paths = dijkstra(graph, start_node)
        shortest_path_to_end = []
        shortest_path = []
        current_node = end_node
        coordinates = []  # 좌표값을 저장할 리스트 추가

        while current_node != start_node:
            shortest_path_to_end.append(current_node)
            current_node = paths[current_node]

        shortest_path_to_end.append(start_node)
        shortest_path_to_end.reverse()
        shortest_path_to_start = shortest_path_to_end[-2::-1]
        shortest_path = shortest_path_to_end + shortest_path_to_start


        # 노드 번호에 해당하는 좌표값을 가져와 coordinates 리스트에 추가
        for node_index in shortest_path:
            node_data = collectionGps.find_one({'nodeIndex': node_index})
            if node_data:
                coordinates.append({'lng': node_data['lng'], 'lat': node_data['lat']})

        print("Shortest path:", shortest_path)
        print("Total distance:", distances[end_node] * 2)

        # 최단 경로와 좌표값을 JSON으로 반환
        return jsonify({"shortest_path": shortest_path, "coordinates": coordinates, "total_distance": distances[end_node] * 2})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

