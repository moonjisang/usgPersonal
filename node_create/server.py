from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pymongo import MongoClient
import os
import math
from pathFinding import pathFinding_blueprint  # Import the blueprint

app = Flask(__name__)
CORS(app)

# Register the blueprint
app.register_blueprint(pathFinding_blueprint)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    filepath = './static/marker/red_marker.jpg'  # Change to your local file path
    return send_file(filepath, mimetype='image/jpeg')


# MongoDB 연결 설정
client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')  # MongoDB 서버 주소로 변경해야 합니다.
db = client['usg']  # 여기서 'your_database_name'을 실제 데이터베이스 이름으로 바꿔주세요.
collectionGps = db['gps']  # 좌표를 저장할 컬렉션 이름
collectionStartGps = db['startGps']
collectionEndGps = db['endGps']

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
        coord['isEndPoint'] = collectionEndGps.find_one({'lng': coord['lng'], 'lat': coord['lat']}) is not None

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
    
@app.route('/save_ending_point', methods=['POST'])
def save_ending_point():
    lng = request.args.get('lng')
    lat = request.args.get('lat')
    
    # 'startGps' 컬렉션에서 좌표가 이미 있는지 확인
    if db.endGps.find_one({'lng': float(lng), 'lat': float(lat)}):
        return jsonify(message='이미 도착지로 지정된 마커입니다.'), 400

    # 'gps' 컬렉션에서 좌표 찾기
    existing_data = db.gps.find_one({'lng': float(lng), 'lat': float(lat)})

    if existing_data:
        # 'startGps' 컬렉션으로 데이터 복사
        db.endGps.insert_one(existing_data)
        return jsonify(message='Data copied to startGps successfully')
    else:
        return jsonify(message='No data found with given coordinates'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    #app.run(debug=True)
