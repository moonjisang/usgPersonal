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
// 변수를 사용하여 현재 상태를 저장합니다.
let isSimulationPaused = false;

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

            var label = document.createElement('div');
            label.innerHTML = 'Node<br>Index&nbsp:&nbsp' + coords.nodeIndex;
            label.style.color = 'black';
            label.style.position = 'absolute';
            label.style.top = '30px'; // 위 아래 조정
            label.style.left = '0px'; // 좌우 조정
            label.style.fontSize = '12px';
            label.style.fontWeight = 'bold'; // 글자 굵게 설정
            marker.getElement().appendChild(label);

            // Add a label for start points
            if (coords.isStartPoint) {
                var label = document.createElement('div');
                label.textContent = 'Start Point';
                label.style.color = 'red';
                label.style.position = 'absolute';
                label.style.top = '-20px';
                marker.getElement().appendChild(label);
            }
            if (coords.isEndPoint) {
                var label = document.createElement('div');
                label.textContent = 'End Point';
                label.style.color = 'red';
                label.style.position = 'absolute';
                label.style.top = '-20px';
                marker.getElement().appendChild(label);
            }
        });
    })
    .catch(error => console.error('Error fetching data:', error));
});

document.getElementById('route_button').addEventListener('click', function() {

    // 서버에 POST 요청을 보냅니다.
    fetch('/calculate_shortest_path')
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

// moveModelToNextCoordinate 함수를 외부에서 호출할 수 있도록 전역 함수로 정의
function moveModelToNextCoordinate() {
    if (currentIndex < coordinates.length) {
        console.log('index : ', currentIndex)
        const nodeCoordinates = coordinates[currentIndex];
        console.log('coord : ', coordinates[currentIndex])

        // 3D 모델을 다음 좌표로 이동합니다.
        model.setTargetPosition(nodeCoordinates.lng, nodeCoordinates.lat, 100); // 좌표값 및 이동 시간을 조정하십시오.
        
        // 주기적으로 checkMoving 함수 호출하여 움직임 확인
        const checkInterval = setInterval(function() {
            if (model.checkMoving()) {
                // 3D 모델이 이동 완료된 경우 실행할 코드를 여기에 추가
                console.log("3D 모델 이동 완료");

                // 이동 완료 시 clearInterval을 사용하여 주기적인 호출 중지
                clearInterval(checkInterval);

                // currentIndex를 증가시켜 다음 좌표로 이동
                currentIndex++;
                
                // 다음 좌표로 이동하는 함수 호출 (이 부분을 작성해야 함)
                moveModelToNextCoordinate();
            }
        }, 1000); // 1000밀리초(1초)마다 호출
    } else {
        console.log('복귀 완료')
    }
}

// Next Route 버튼에 대한 이벤트 리스너 추가
document.getElementById('start_button').addEventListener('click', moveModelToNextCoordinate);

// Stop Route 버튼에 대한 이벤트 리스너 추가
document.getElementById('stop_button').addEventListener('click', function() {
    if (!isSimulationPaused) {
        // 3D 모델의 이동을 일시 정지하고 버튼 텍스트를 변경합니다.
        model.pauseMoving();
        this.textContent = 'Resume Simulation'; // 버튼 텍스트 변경
    } else {
        // "Resume Simulation" 버튼을 클릭하면 3D 모델의 이동을 다시 시작하고 버튼 텍스트를 변경합니다.
        model.resumeMoving();
        this.textContent = 'Stop Route'; // 버튼 텍스트 변경
    }
    isSimulationPaused = !isSimulationPaused; // 상태를 토글합니다.
});