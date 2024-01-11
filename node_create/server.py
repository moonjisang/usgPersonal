from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pymongo import MongoClient
import os
import math

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    filepath = './static/marker/red_marker.jpg'  # Change to your local file path
    return send_file(filepath, mimetype='image/jpeg')


# MongoDB 연결 설정
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')  # MongoDB 서버 주소로 변경해야 합니다.
db = client['usg']  # 여기서 'your_database_name'을 실제 데이터베이스 이름으로 바꿔주세요.
collectionGps = db['gps']  # 좌표를 저장할 컬렉션 이름
collectionStartGps = db['startGps']


def update_nodeindex():
    # 1. gps 컬렉션에서 모든 nodeIndex 값을 가져온다.
    gps_documents = collectionGps.find({}, {'_id': 0, 'nodeIndex': 1})
    gps_nodeindexes = [doc['nodeIndex'] for doc in gps_documents if 'nodeIndex' in doc]

    # 2. nodeIndex 컬렉션의 nodeIndex 배열을 위에서 가져온 배열로 업데이트한다.
    nodeindex_collectionGps = db['nodeIndex']  # nodeIndex 컬렉션 접근
    result = nodeindex_collectionGps.update_one({}, {'$set': {'nodeIndex': gps_nodeindexes}})

    if result.modified_count > 0:
        print("nodeIndex 컬렉션이 성공적으로 업데이트되었습니다.")
    else:
        print("업데이트할 내용이 없거나 오류가 발생했습니다.")



# 정적 파일 제공을 위한 라우트 추가
@app.route('/map/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'), filename)

# 웹 페이지를 렌더링하는 라우트
@app.route('/')
def web_html():
    return render_template('web.html')  # index.html 파일을 렌더링합니다.

# 웹 페이지를 렌더링하는 라우트
@app.route('/simulation')
def simulation_html():
    return render_template('simulation.html')  # index.html 파일을 렌더링합니다.


# flask에서 좌표를 가져오는 라우트
@app.route('/get_coordinates', methods=['GET'])
def get_coordinates():
    try:
        update_nodeindex()
    except Exception as e:
        print(f"실패. 오류: {e}")
    # MongoDB에서 좌표 데이터 가져오기
    coordinates_data = list(collectionGps.find({}, {'_id': 0}))

    # Check if coordinates are in startGps and add a flag
    for coord in coordinates_data:
        coord['isStartPoint'] = collectionStartGps.find_one({'lng': coord['lng'], 'lat': coord['lat']}) is not None

    return jsonify(coordinates_data)

# 좌표를 선택하고 "Select Coordinates" 버튼을 클릭할 때, 해당 좌표를 MongoDB에 저장하는 기능을 추가합니다.
@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    data = request.get_json()

    if data:
        # "nodeIndex" 컬렉션에서 사용되지 않은 가장 작은 번호 찾기
        index_document = db.nodeIndex.find_one({})
        if index_document and 'nodeIndex' in index_document:
            node_index_list = index_document['nodeIndex']
        else:
            node_index_list = []

        # 사용되지 않는 가장 작은 번호 찾기
        next_index = 0
        while next_index in node_index_list:
            next_index += 1
        
        # 좌표 데이터에 번호 추가
        data['nodeIndex'] = next_index

        # 좌표 데이터에 간선항목 추가
        data['nodeEdge'] = []

        # MongoDB에 좌표 저장
        collectionGps.insert_one(data)

        # "nodeIndex" 컬렉션 내의 숫자 배열 업데이트
        node_index_list.append(next_index)
        db.nodeIndex.update_one({}, {'$set': {'nodeIndex': node_index_list}})
        
        return jsonify({"nodeIndex" : next_index, "message": "좌표가 성공적으로 저장되었습니다."})
    else:
        return jsonify({"error": "좌표를 제공하지 않았습니다."})
    
@app.route('/delete_coordinate', methods=['DELETE'])
def delete_coordinate():
    lng = float(request.args.get('lng'))
    lat = float(request.args.get('lat'))

    # DB에서 해당 좌표 찾아서 삭제
    result = collectionGps.delete_one({"lng": lng, "lat": lat})

    try:
        update_nodeindex()
    except Exception as e:
        print(f"실패. 오류: {e}")
        
    if result.deleted_count > 0:
        return jsonify({"message": "좌표가 성공적으로 삭제되었습니다."}), 200
    else:
        return jsonify({"message": "삭제할 좌표를 찾을 수 없습니다."}), 404

@app.route('/update_node_edges', methods=['POST'])
def update_node_edges():
    data = request.json
    nodeIndex1 = data['nodeIndex1']
    nodeIndex2 = data['nodeIndex2']

    try:
        # marker1의 nodeEdge 배열에 nodeIndex2 값을 추가합니다.
        # $push 연산자를 사용하면 지정된 값이 배열에 이미 존재하는지 여부와 상관없이 추가됩니다.
        collectionGps.update_one({"nodeIndex": nodeIndex1}, {"$push": {"nodeEdge": nodeIndex2}})
        collectionGps.update_one({"nodeIndex": nodeIndex2}, {"$push": {"nodeEdge": nodeIndex1}})
        
        return jsonify({"message": "Successfully updated nodeEdge values!"})
    except Exception as e:
        return jsonify({"message": f"Error updating nodeEdge: {str(e)}"})

@app.route('/delete_node_edges', methods=['DELETE'])
def delete_node_edges():
    nodeIndex1 = request.args.get('nodeIndex1')
    nodeIndex2 = request.args.get('nodeIndex2')

    try:
        # 각 노드의 nodeEdge 배열에서 상대 노드 인덱스를 삭제
        collectionGps.update_one({"nodeIndex": int(nodeIndex1)}, {"$pull": {"nodeEdge": int(nodeIndex2)}})
        collectionGps.update_one({"nodeIndex": int(nodeIndex2)}, {"$pull": {"nodeEdge": int(nodeIndex1)}})
        
        return jsonify({"message": "Node edges deleted successfully"})
    except Exception as e:
        return jsonify({"message": f"Error deleting node edges: {str(e)}"}), 500

@app.route('/save_starting_point', methods=['POST'])
def save_starting_point():
    lng = request.args.get('lng')
    lat = request.args.get('lat')
    
    # 'startGps' 컬렉션에서 좌표가 이미 있는지 확인
    if db.startGps.find_one({'lng': float(lng), 'lat': float(lat)}):
        return jsonify(message='이미 출발지로 지정된 마커입니다.'), 400

    # 'gps' 컬렉션에서 좌표 찾기
    existing_data = db.gps.find_one({'lng': float(lng), 'lat': float(lat)})

    if existing_data:
        # 'startGps' 컬렉션으로 데이터 복사
        db.startGps.insert_one(existing_data)
        return jsonify(message='Data copied to startGps successfully')
    else:
        return jsonify(message='No data found with given coordinates'), 404

@app.route('/calculate_shortest_path', methods=['POST'])
def calculate_shortest_path():
    try:
        class Graph:
            def __init__(self):
                self.nodes = set()
                self.edges = {}
                self.distances = {}

            def add_node(self, value):
                self.nodes.add(value)

            def add_edge(self, from_node, to_node, distance):
                self.edges.setdefault(from_node, [])
                self.edges[from_node].append(to_node)
                self.distances[(from_node, to_node)] = distance

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

        def build_graph():
            graph = Graph()
            nodes_data = list(collectionGps.find({}))

            for node in nodes_data:
                graph.add_node(node['nodeIndex'])

                for edge in node['nodeEdge']:
                    target_node = collectionGps.find_one({'nodeIndex': edge})
                    if target_node:
                        distance = get_distance(node['lat'], node['lng'], target_node['lat'], target_node['lng'])
                        graph.add_edge(node['nodeIndex'], edge, distance)

            return graph

        # Example usage
        graph = build_graph()
        # Assuming we want to find the shortest path from node index 0 to node index 5
        start_node = 0
        end_node = 5
        distances, paths = dijkstra(graph, start_node)
        shortest_path = []
        current_node = end_node
        coordinates = []  # 좌표값을 저장할 리스트 추가

        while current_node != start_node:
            shortest_path.append(current_node)
            current_node = paths[current_node]

        shortest_path.append(start_node)
        shortest_path.reverse()

        # 노드 번호에 해당하는 좌표값을 가져와 coordinates 리스트에 추가
        for node_index in shortest_path:
            node_data = collectionGps.find_one({'nodeIndex': node_index})
            if node_data:
                coordinates.append({'lng': node_data['lng'], 'lat': node_data['lat']})

        print("Shortest path:", shortest_path)
        print("Total distance:", distances[end_node])

        # 최단 경로와 좌표값을 JSON으로 반환
        return jsonify({"shortest_path": shortest_path, "coordinates": coordinates, "total_distance": distances[end_node]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    #app.run(debug=True)
