import { Mapbox3DModel } from './3d_model.js'; // Adjust the path as necessary

mapboxgl.accessToken = 'pk.eyJ1IjoiYW5zOTM5IiwiYSI6ImNsY3g3dHR4czIwNGszdms2ZDA5eHZtOHIifQ.v1aKMbtU1_vRo4ssSlKCqA';

const map = new mapboxgl.Map({
    style: 'mapbox://styles/mapbox/light-v11',
    center: [128.097628451928, 35.153823441258],
    zoom: 15.5,
    pitch: 45,
    bearing: 7,
    container: 'map',
    antialias: true
});

let model; // 전역 변수로 model 선언
let coordinates
let currentIndex = 1

map.on('load', function () {
    const modelUrl = '/static/gltf/drone/scene.gltf'; // Update with your model path
    model = new Mapbox3DModel(modelUrl, map, [128.09298831977316, 35.1541499657056], 100);

    // Fetch coordinates from your API
    fetch('get_coordinates')
    .then(response => response.json())
    .then(data => {
        // 좌표를 지도에 표시
        data.forEach(coords => {
            var marker = new mapboxgl.Marker()
                .setLngLat([coords.lng, coords.lat])
                .addTo(map);
            marker.nodeIndex = coords.nodeIndex;

            // Add a label for start points
            if (coords.isStartPoint) {
                var label = document.createElement('div');
                label.textContent = 'Start Point';
                label.style.color = 'red';
                label.style.position = 'absolute';
                label.style.top = '-20px';
                marker.getElement().appendChild(label);
            }
        });
    })
    .catch(error => console.error('Error fetching data:', error));
});

// Start Simulation 버튼에 대한 이벤트 리스너 추가
document.getElementById('start_button').addEventListener('click', function() {
    console.log(coordinates)
    // 최단 경로 데이터를 사용하여 3D 모델을 이동합니다.
    moveModelAlongRoute(coordinates);
    
    // model.setTargetPosition(128.09530124636944, 35.1538675805656, 200); // 새로운 좌표로 모델 이동
});

document.getElementById('route_button').addEventListener('click', function() {
    const startNode = 0; // 시작 노드 인덱스를 원하는 값으로 대체
    const endNode = 5;   // 종료 노드 인덱스를 원하는 값으로 대체

    // 서버에 POST 요청을 보냅니다.
    fetch('/calculate_shortest_path', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ start_node: startNode, end_node: endNode }),
    })
    .then(response => response.json())
    .then(data => {
        // 응답 데이터를 처리합니다. 예: 최단 경로로 3D 모델을 이동합니다.
        console.log("최단 경로:", data.shortest_path);
        console.log("좌표값:", data.coordinates);

        // 팝업 창 열기
        const popupWindow = window.open('static/popup.html', 'RoutePopup', 'width=400,height=300');
        popupWindow.onload = function() {
            // 팝업 창의 HTML 엘리먼트에 최단 경로와 총 거리 정보 채우기
            const shortestPathElement = popupWindow.document.getElementById('shortest-path');
            const totalDistanceElement = popupWindow.document.getElementById('total-distance');
            shortestPathElement.textContent = data.shortest_path.join(' -> ');
            totalDistanceElement.textContent = data.total_distance + ' 미터';
        };

        coordinates = data.coordinates
    })
    .catch(error => console.error('데이터 가져오기 오류:', error));
});

// Next Route 버튼에 대한 이벤트 리스너 추가
document.getElementById('route_next').addEventListener('click', function() {

    if (currentIndex < coordinates.length) {
        console.log('index : ', currentIndex)
        const nodeCoordinates = coordinates[currentIndex];
        console.log('coord : ', coordinates[currentIndex])

        // 3D 모델을 다음 좌표로 이동합니다.
        model.setTargetPosition(nodeCoordinates.lng, nodeCoordinates.lat, 200); // 좌표값 및 이동 시간을 조정하십시오.
        currentIndex++; // 다음 좌표로 이동
    }
});
