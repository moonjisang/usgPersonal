export class Mapbox3DModel {
    constructor(modelUrl, map, modelOrigin, modelAltitude) {
        this.modelUrl = modelUrl;
        this.map = map;
        this.modelOrigin = modelOrigin
        this.modelAltitude = modelAltitude
        this.clock = new THREE.Clock();
        this.modelRotate = [Math.PI / 2, 0, 0]; // 모델 회전 (라디안 단위)
        this.modelScale = 5.41843220338983e-8;
        this.modelTransform
        this.isMoving
        this.targetPosition; // 타겟 좌표 지리적 좌표로 저장
        this.mercatorTargetPosition; // 타겟 좌표 메르카토르 좌표로 저장
        this.modelTranslateLngLat // 현재 좌표 지리적 좌표
        this.modelAsMercatorCoordinate // 현재 좌표 메르카토르 좌표
        this.add3DModel();
        this.paused = false; // 이동 일시 정지 상태를 나타내는 변수
    }

    add3DModel() {
        this.modelAsMercatorCoordinate = mapboxgl.MercatorCoordinate.fromLngLat(
            this.modelOrigin,
            this.modelAltitude
        );

        // 3D 모델 트랜스폼 설정
        this.modelTransform = {
            translateX: this.modelAsMercatorCoordinate.x,
            translateY: this.modelAsMercatorCoordinate.y,
            translateZ: this.modelAsMercatorCoordinate.z,
            rotateX: this.modelRotate[0],
            rotateY: this.modelRotate[1],
            rotateZ: this.modelRotate[2],
            scale: this.modelScale
        };

        this.map.addLayer({
            id: '3d-model',
            type: 'custom',
            renderingMode: '3d',
            onAdd: (map, gl) => this.onAdd(map, gl),
            render: (gl, matrix) => this.render(gl, matrix)
        });
    }

    change3DModelLocation(origin, altitude) {
        this.modelAsMercatorCoordinate = mapboxgl.MercatorCoordinate.fromLngLat(
            origin,
            altitude
        );

        // 3D 모델 트랜스폼 설정
        this.modelTransform = {
            translateX: this.modelAsMercatorCoordinate.x,
            translateY: this.modelAsMercatorCoordinate.y,
            translateZ: this.modelAsMercatorCoordinate.z,
            rotateX: this.modelRotate[0],
            rotateY: this.modelRotate[1],
            rotateZ: this.modelRotate[2],
            scale: this.modelScale
        };
    }

    // Method to set the target position
    setTargetPosition(targetLng, targetLat, targetAltitude) {
        // Store the target position in both geographic and Mercator coordinates
        this.targetPosition = { lng: targetLng, lat: targetLat, altitude: targetAltitude };
        this.mercatorTargetPosition = mapboxgl.MercatorCoordinate.fromLngLat(
            [targetLng, targetLat],
            targetAltitude
        );
        this.isMoving = true;
        console.log('origin : ', this.modelOrigin)
    }

    checkMoving() {
        return !this.isMoving;
    }

    pauseMoving() { // 3D 모델의 이동 일시 정지
        this.paused = true;
    }

    resumeMoving() { // 3D 모델의 이동 재개
        this.paused = false;
    }

    getDistance([lon1, lat1], [lon2, lat2]) { // generally used geo measurement function
        const R = 6378.137 // Radius of earth in KM
        const dLon = lon2 * Math.PI / 180 - lon1 * Math.PI / 180
        const dLat = lat2 * Math.PI / 180 - lat1 * Math.PI / 180
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2)
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
        const d = R * c
    
        return d * 1000 // meters
    }

    getMercatorDistance(coord1, coord2) {
        const R = 6371000; // 지구 반지름 (미터)
    
        const lat1 = coord1[1];
        const lon1 = coord1[0];
        const lat2 = coord2[1];
        const lon2 = coord2[0];
    
        // 라디안 단위로 변환
        const lat1Rad = (lat1 * Math.PI) / 180;
        const lon1Rad = (lon1 * Math.PI) / 180;
        const lat2Rad = (lat2 * Math.PI) / 180;
        const lon2Rad = (lon2 * Math.PI) / 180;
    
        // 중심 각도 계산
        const dLon = lon2Rad - lon1Rad;
    
        // 중심 각도를 이용하여 거리 계산
        const distance = R * Math.acos(Math.sin(lat1Rad) * Math.sin(lat2Rad) + Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dLon));
    
        return distance;
    }

    onAdd(map, gl) {
        this.camera = new THREE.Camera();
        this.scene = new THREE.Scene();

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
        directionalLight.position.set(0, 1, 0); // 조명 위치 조정
        this.scene.add(directionalLight);

        // 3D 모델 로더 생성 및 모델 로드
        const loader = new THREE.GLTFLoader();
        loader.load(this.modelUrl, (gltf) => {
            this.scene.add(gltf.scene);
            this.mixer = new THREE.AnimationMixer(gltf.scene);
            gltf.animations.forEach((clip) => {
                this.mixer.clipAction(clip).play();
            });
        }, undefined, (error) => {
            console.error('An error happened while loading the model:', error);
        })

        this.map = map;

        // 맵의 카메라와 동기화
        this.renderer = new THREE.WebGLRenderer({
            canvas: map.getCanvas(),
            context: gl,
            antialias: true
        });

        this.renderer.autoClear = false;
    }

    render(gl, matrix) {
        if (!this.paused) {
            if (this.isMoving) {
                // Interpolation logic
                const speed = 0.005; // Adjust speed as needed
                // 메르카토르 좌표를 지리적 좌표로 변환
                this.modelTranslateLngLat = new mapboxgl.MercatorCoordinate(this.modelTransform.translateX, this.modelTransform.translateY, this.modelTransform.translateZ).toLngLat();
                const distance = this.getDistance(
                    [this.modelTranslateLngLat.lng, this.modelTranslateLngLat.lat],
                    [this.targetPosition.lng, this.targetPosition.lat]
                );
                // console.log(
                //     [this.modelTranslateLngLat.lng, this.modelTranslateLngLat.lat],
                //     [this.targetPosition.lng, this.targetPosition.lat]
                // );
                if (distance < 1) {
                    this.isMoving = false;
                    this.modelOrigin = [this.targetPosition.lng, this.targetPosition.lat]
                    this.change3DModelLocation(this.modelOrigin, this.targetPosition.altitude)
                    console.log(distance);
                    console.log('false');
                } else {
                    // 아직 이동 중이므로 메르카토르 좌표 간의 거리에 따라 이동
                    this.modelTransform.translateX +=
                        (this.mercatorTargetPosition.x - this.modelAsMercatorCoordinate.x) * speed;
                    this.modelTransform.translateY +=
                        (this.mercatorTargetPosition.y - this.modelAsMercatorCoordinate.y) * speed;
                    // this.modelTransform.translateZ +=
                    //     (this.mercatorTargetPosition.z - this.modelAsMercatorCoordinate.z) * speed;
                }
            }
    
            
        }
        // 애니메이션 믹서 업데이트
        if (this.mixer) {
            var delta = this.clock.getDelta();
            this.mixer.update(delta);
        }

        const rotationX = new THREE.Matrix4().makeRotationAxis(
            new THREE.Vector3(1, 0, 0),
            this.modelTransform.rotateX
        );
        const rotationY = new THREE.Matrix4().makeRotationAxis(
            new THREE.Vector3(0, 1, 0),
            this.modelTransform.rotateY
        );
        const rotationZ = new THREE.Matrix4().makeRotationAxis(
            new THREE.Vector3(0, 0, 1),
            this.modelTransform.rotateZ
        );

        const m = new THREE.Matrix4().fromArray(matrix);
        const l = new THREE.Matrix4()
            .makeTranslation(
                this.modelTransform.translateX,
                this.modelTransform.translateY,
                this.modelTransform.translateZ
            )
            .scale(
                new THREE.Vector3(
                    this.modelTransform.scale,
                    -this.modelTransform.scale,
                    this.modelTransform.scale
                )
            )
            .multiply(rotationX)
            .multiply(rotationY)
            .multiply(rotationZ);

        this.camera.projectionMatrix = m.multiply(l);
        this.renderer.state.reset();
        this.renderer.render(this.scene, this.camera);
        this.map.triggerRepaint()
    }

}