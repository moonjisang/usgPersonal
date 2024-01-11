// mapbox-script.js 파일 내용
mapboxgl.accessToken = 'pk.eyJ1IjoiYW5zOTM5IiwiYSI6ImNsY3g3dHR4czIwNGszdms2ZDA5eHZtOHIifQ.v1aKMbtU1_vRo4ssSlKCqA';
 const map = new mapboxgl.Map({
     // Choose from Mapbox's core styles, or make your own style with Mapbox Studio
     style: 'mapbox://styles/mapbox/light-v11',
     center: [128.097628451928, 35.153823441258],
     zoom: 15.5,
     pitch: 45,
     bearing: 7,
     container: 'map',
     antialias: true
 });

 map.on('style.load', () => {
     // Insert the layer beneath any symbol layer.
     const layers = map.getStyle().layers;
     const labelLayerId = layers.find(
         (layer) => layer.type === 'symbol' && layer.layout['text-field']
     ).id;

     // The 'building' layer in the Mapbox Streets
     // vector tileset contains building height data
     // from OpenStreetMap.
     map.addLayer(
         {
             'id': 'add-3d-buildings',
             'source': 'composite',
             'source-layer': 'building',
             'filter': ['==', 'extrude', 'true'],
             'type': 'fill-extrusion',
             'minzoom': 15,
             'paint': {
                 'fill-extrusion-color': '#aaa',

                 // Use an 'interpolate' expression to
                 // add a smooth transition effect to
                 // the buildings as the user zooms in.
                 'fill-extrusion-height': [
                     'interpolate',
                     ['linear'],
                     ['zoom'],
                     15,
                     0,
                     15.05,
                     ['get', 'height']
                 ],
                 'fill-extrusion-base': [
                     'interpolate',
                     ['linear'],
                     ['zoom'],
                     15,
                     0,
                     15.05,
                     ['get', 'min_height']
                 ],
                 'fill-extrusion-opacity': 0.6
             }
         },
         labelLayerId
     );
 });

// 기존에 생성된 마커들
var existingMarkers = [];
// 팝업 메뉴를 저장할 배열
var popups = [];
// 연결할 두 마커 좌표 저장할 배열
var selectedCoordinates = [];
// marker 선택시 이벤트 핸들러
function markerClickHandler(marker) {
    return function() {
        // 클릭한 마커의 경도 (lng) 및 위도 (lat) 정보 가져오기
        var lngLat = marker.getLngLat();
        var lng = lngLat.lng;
        var lat = lngLat.lat;

        // 클릭한 마커의 좌표 정보 출력
        console.log('클릭한 마커의 경도, 위도 : ' , lng, lat);

        // 클릭한 마커의 경도 (lng) 및 위도 (lat) 정보를 문자열로 생성
        var lngLatString = '경도 (lng): ' + lng + '<br>위도 (lat): ' + lat;

        // 팝업 메뉴 생성
        const popupContent = `
            <div class="popup">
                <h3>마커 정보</h3>
                <p>${lngLatString}</p>
                <button id="startingPointButton">Starting Point</button>
                <button id="deleteButton">Delete</button>
            </div>`;

        var popup = new mapboxgl.Popup({
            closeButton: true, // 닫기 버튼 표시
            closeOnClick: false // 지도 클릭 시 팝업이 닫히지 않도록 설정
        }).setHTML(popupContent);

        setTimeout(() => {
            // 팝업 메뉴를 배열에 추가
            popups.push(popup);
        }, 500)
        

        // 팝업 메뉴를 클릭한 마커 위치에 표시
        popup.setLngLat([lng, lat]).addTo(map);

        selectedCoordinates.push(marker);

        // setTimeout을 사용하여 DOM 업데이트 후에 Delete 버튼에 이벤트 리스너 추가
        setTimeout(() => {
            var deleteButton = document.getElementById('deleteButton');
            if (deleteButton) {
                deleteButton.addEventListener('click', function() {
                    // 마커 지우기
                    marker.remove();
                    // 팝업 지우기
                    popup.remove();
                    // DB에서 좌표 삭제하기 위한 서버 요청
                    fetch(`/delete_coordinate?lng=${lng}&lat=${lat}`, {
                        method: 'DELETE'
                    }).then(response => response.json())
                    .then(data => {
                        console.log(data.message);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
                    // 배열에 추가된 마커 삭제
                    const index = selectedCoordinates.indexOf(marker);
                    if (index > -1) {
                        selectedCoordinates.splice(index, 1);
                    }
                });
            }

            document.getElementById('startingPointButton').addEventListener('click', function() {
                // 라벨을 추가할 마커 요소 선택
                var markerElement = marker.getElement();

                // 라벨용 HTML 요소 생성
                var label = document.createElement('div');
                label.textContent = 'Start Point';
                label.style.color = 'red'; // 색상 설정
                label.style.position = 'absolute';
                label.style.top = '-20px'; // 라벨 위치 조정

                // 마커에 라벨 추가
                markerElement.appendChild(label);

                fetch(`/save_starting_point?lng=${lng}&lat=${lat}`, {
                    method: 'POST'
                }).then(response => {
                    if (response.status === 400) {
                        return response.json().then(data => {
                            alert(data.message); // '이미 출발지로 지정된 마커입니다.'
                        });
                    } else {
                        return response.json().then(data => {
                            console.log(data.message);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
            
        }, 0);

        // 두 개의 좌표가 선택되었을 경우
        if (selectedCoordinates.length === 2) {
            connectMarkersWithLine(selectedCoordinates);
            selectedCoordinates = [];  // 배열 초기화
        }
    };
}
// 선택 된 두 마커를 연결하는 함수
function connectMarkersWithLine([marker1, marker2]) {
    // 동일한 마커를 두 번 선택한 경우 함수 실행 중단
    if (marker1.nodeIndex === marker2.nodeIndex) {
        console.log('동일한 마커를 두 번 선택할 수 없습니다.');
        alert('동일한 마커를 두 번 선택할 수 없습니다.');
        selectedCoordinates = [];  // 배열 초기화
        return;
    }

    // 서버에 nodeIndex 정보 전송 및 DB 업데이트
    updateNodeEdgesInDB(marker1.nodeIndex, marker2.nodeIndex);

    // 두마커 라인 그리기
    drawLine(marker1, marker2)
}

function drawLine(marker1, marker2) {

    // 두 마커의 nodeIndex 값을 사용하여 고유한 레이어와 소스 ID 생성
    var lineSourceId = 'line-source-' + marker1.nodeIndex + '-' + marker2.nodeIndex;
    var lineLayerId = 'line-layer-' + marker1.nodeIndex + '-' + marker2.nodeIndex;

    // 두 좌표를 연결하는 LineString 피처를 정의합니다.
    var lineFeature = {
        type: 'Feature',
        geometry: {
            type: 'LineString',
            coordinates: [
                [marker1.getLngLat().lng, marker1.getLngLat().lat],
                [marker2.getLngLat().lng, marker2.getLngLat().lat]
            ]
        }
    };

    // 지도에 LineString 피처를 표시하는 소스와 레이어를 추가합니다.
    map.addSource(lineSourceId, {
        type: 'geojson',
        data: lineFeature
    });

    map.addLayer({
        id: lineLayerId,
        type: 'line',
        source: lineSourceId,
        layout: {
            'line-cap' : 'round'
        },
        paint: {
            'line-color': '#0000FF',
            'line-width': 10,
            'line-opacity': 0.5
        }
    });
}

function updateNodeEdgesInDB(nodeIndex1, nodeIndex2) {
    // 서버에게 요청을 전송하여 DB의 nodeEdge 값을 업데이트합니다.
    fetch('/update_node_edges', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            nodeIndex1: nodeIndex1, 
            nodeIndex2: nodeIndex2 
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// 버튼 선택
const reloadButton = document.getElementById('reloadButton');

// 클릭 이벤트 리스너 추가
reloadButton.addEventListener('click', function() {
    // 아래의 함수를 실행합니다.
    fetchCoordinatesAndAddToMap();
});

function fetchCoordinatesAndAddToMap() {
    // 기존의 마커들을 모두 지운다.
    if (existingMarkers.length > 0) {
        existingMarkers.forEach(marker => {
            marker.remove();
        });
        existingMarkers = [];
    }
    // 서버에서 좌표 데이터 가져오기
    fetch('/get_coordinates')
        .then(response => response.json())
        .then(data => {
            // 좌표를 지도에 표시
            data.forEach(coords => {
                var marker = new mapboxgl.Marker()
                    .setLngLat([coords.lng, coords.lat])
                    .addTo(map);
                marker.nodeIndex = coords.nodeIndex;
                existingMarkers.push(marker);

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
            console.log('reload existing markers : ', existingMarkers)
            // 존재하는 marker 클릭이벤트 추가
            existingMarkers.forEach(function(marker) {
                marker.getElement().addEventListener('click', markerClickHandler(marker));
            });

            // nodeEdge 배열 값이 존재하는 데이터만 필터링
            const nodesWithEdges = data.filter(coords => coords.nodeEdge.length > 0);

            nodesWithEdges.forEach(node => {
                const sourceMarker = existingMarkers.find(marker => marker.nodeIndex === node.nodeIndex);
                node.nodeEdge.forEach(edgeIndex => {
                    const targetMarker = existingMarkers.find(marker => marker.nodeIndex === edgeIndex);
                    if (sourceMarker && targetMarker) {
                        drawLine(sourceMarker, targetMarker);
                    }
                });
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// "Select Coordinates" 버튼 클릭 이벤트 처리
let isSelectingCoordinates = false;
const selectButton = document.getElementById('select-button');
selectButton.addEventListener('click', () => {
    isSelectingCoordinates = !isSelectingCoordinates;

    if (isSelectingCoordinates) {
        // Enter coordinate selection mode
        map.getCanvas().style.cursor = 'crosshair';
        selectButton.textContent = 'Release'; // 버튼 텍스트 변경
    } else {
        // Return to the default map interaction (dragging)
        map.getCanvas().style.cursor = '';
        selectButton.textContent = 'Select Coordinates'; // 버튼 텍스트 변경
    }
});

// Handle map click event when in coordinate selection mode
map.on('click', (e) => {

    if (popups.length > 0) {
        // 지도 클릭시 모든 팝업 닫기
        popups.forEach(popup => {
        popup.remove();
        });
        // 모든 팝업 객체를 배열에서 제거
        popups = [];
    }

    // 좌표 추가 버튼 클릭시
    if (isSelectingCoordinates) {
        const coordinates = e.lngLat;

        // 좌표를 서버로 전송
        fetch('/save_coordinates', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ lng: coordinates.lng, lat: coordinates.lat })
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);

            // 새로운 마커를 지도에 추가
            var marker = new mapboxgl.Marker()
            .setLngLat([coordinates.lng, coordinates.lat])
            .addTo(map);
            marker.nodeIndex = data.nodeIndex;
            existingMarkers.push(marker)
            marker.getElement().addEventListener('click', markerClickHandler(marker));
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

// map에 우클릭 시 기능
map.on('contextmenu', function(e) {
    // 라인 삭제 기능
    //라인 레이어 위에서의 클릭을 감지합니다.
    var features = map.queryRenderedFeatures(e.point);

    for (var i = 0; i < features.length; i++) {
        var feature = features[i];

        // 피처가 라인인 경우 해당 라인의 레이어 ID를 출력합니다.
        if (feature.geometry.type === 'LineString') {
            var idValue = parseInt(feature.layer.id.split("-")[2] + feature.layer.id.split("-")[3]);
            console.log("Clicked line's layer ID:", idValue);
            // 클릭한 라인이 있으면 해당 라인을 지웁니다.
            if (features.length && (typeof idValue === 'number' && !isNaN(idValue))) {
                console.log('숫자니깐지운다')
                var feature = features[0];

                // 라인의 소스와 레이어 ID를 가져옵니다.
                var lineSourceId = feature.source;
                var lineLayerId = feature.layer.id;

                // 라인의 소스와 레이어를 지도에서 제거합니다.
                if (map.getLayer(lineLayerId)) {
                    map.removeLayer(lineLayerId);
                }
                if (map.getSource(lineSourceId)) {
                    map.removeSource(lineSourceId);
                }
                // 노드 인덱스를 추출하고 서버에 삭제 요청
                const nodeIndex1 = parseInt(feature.layer.id.split("-")[2]);
                const nodeIndex2 = parseInt(feature.layer.id.split("-")[3]);
                deleteNodeEdges(nodeIndex1, nodeIndex2);
            }
        }
    }    
})

// 라인 삭제 시, 서버에 삭제 요청을 보내는 함수
function deleteNodeEdges(nodeIndex1, nodeIndex2) {
    fetch(`/delete_node_edges?nodeIndex1=${nodeIndex1}&nodeIndex2=${nodeIndex2}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
