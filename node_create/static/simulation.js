import { Mapbox3DModel } from './3d_model.js'; // Adjust the path as necessary

mapboxgl.accessToken = 'pk.eyJ1IjoiYW5zOTM5IiwiYSI6ImNsY3g3dHR4czIwNGszdms2ZDA5eHZtOHIifQ.v1aKMbtU1_vRo4ssSlKCqA';

const map = new mapboxgl.Map({
    style: 'mapbox://styles/ans939/clrlws8mz001v01pq6w8396yy',
    center: [128.097628451928, 35.153823441258],
    zoom: 15.5,
    pitch: 45,
    bearing: 7,
    container: 'map',
    antialias: true
});

let modelSpeed = 0.005
const modelSpeed2 = 0.008
let coordinates
// 변수를 사용하여 현재 상태를 저장합니다.
let isSimulationPaused = false;

const Url = '/static/gltf/drone2/scene.gltf'; // Update with your model path
const Url2 = '/static/gltf/drone3/scene.gltf';
let modelUrl = Url
const models = []

map.on('load', function () {
    // 서버에 POST 요청을 보냅니다.
    fetch('/get_drones')
    .then(response => response.json())
    .then(data => {
        // 응답 데이터를 처리합니다. 예: 최단 경로로 3D 모델을 이동합니다.
        data.map((drone) => {
            console.log("index :", drone.index);
            console.log("loc :", drone.loc);
            console.log('url : ', drone.frame)
            
            // drone.frame 값에 따라 modelUrl 설정
            modelUrl = (drone.frame === 'Url2') ? Url2 : Url;
            const model = new Mapbox3DModel(modelUrl, map, drone.loc, 100, drone.index, modelSpeed);
            modelSpeed += 0.003
            models.push(model)
        })
    })
    .catch(error => console.error('데이터 가져오기 오류:', error));

    
    // model = new Mapbox3DModel(modelUrl, map, [128.09298831977316, 35.1541499657056], 100, 1, modelSpeed);
    // model2 = new Mapbox3DModel(modelUrl2, map, [128.09298831977316, 35.1541499657056], 100, 2, modelSpeed2);

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

    // 서버로 GET 요청을 보냅니다.
    fetch('/get_weather_data')
    .then(response => response.json())
    .then(wind_data => {
        // 날씨 정보를 처리하는 코드를 작성합니다.
        console.log('받아온 날씨 정보:', wind_data);
        // 받아온 날씨 정보(weatherData)를 원하는 대로 처리합니다.
        // 바람의 방향 및 풍속 정보
        var windDirection = wind_data[0].obsrValue; // 풍향 (도)
        var windSpeed = wind_data[1].obsrValue; // 풍속 (m/s)

        // 바람의 방향을 나타내는 마커 추가
        var el = document.createElement('div');
        el.className = 'marker';
        el.style.width = '30px';
        el.style.height = '30px';
        el.style.backgroundImage = 'url(http://localhost:5000/upload)'; // 여기에 바람 방향 화살표 이미지 URL을 입력하세요.
        el.style.backgroundSize = '100%';
        el.style.transform = `rotate(${windDirection}deg)`; // 바람 방향에 따라 이미지 회전

        // 마커를 지도에 추가
        new mapboxgl.Marker(el)
            .setLngLat([128.0928, 35.1542]) // 마커 위치 (경도, 위도)
            .addTo(map);

        // 풍속 표시
        var popup = new mapboxgl.Popup({ offset: 25 })
            .setText(`풍속: ${windSpeed} m/s`)
            .setLngLat([128.0928, 35.1542])
            .addTo(map);
    })
    .catch(error => console.error('날씨 데이터 가져오기 오류:', error));

    
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



// Next Route 버튼에 대한 이벤트 리스너 추가
document.getElementById('start_button').addEventListener('click', function() {
    models[0].moveModelToNextCoordinate(coordinates)
    models[1].moveModelToNextCoordinate(coordinates)
});

// Stop Route 버튼에 대한 이벤트 리스너 추가
document.getElementById('stop_button').addEventListener('click', function() {
    if (!isSimulationPaused) {
        // 3D 모델의 이동을 일시 정지하고 버튼 텍스트를 변경합니다.
        models[0].pauseMoving();
        models[1].pauseMoving();
        this.textContent = 'Resume Simulation'; // 버튼 텍스트 변경
    } else {
        // "Resume Simulation" 버튼을 클릭하면 3D 모델의 이동을 다시 시작하고 버튼 텍스트를 변경합니다.
        models[0].resumeMoving();
        models[1].resumeMoving();
        this.textContent = 'Stop Route'; // 버튼 텍스트 변경
    }
    isSimulationPaused = !isSimulationPaused; // 상태를 토글합니다.
});