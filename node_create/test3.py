import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 간단한 그래프 정의 (인접 행렬)
graph = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0],
                  [1, 0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 0]])

# 랜덤한 거리와 풍향각 정보를 가진 그래프 생성
num_nodes = 6  # 노드의 수
num_features = 2  # 특성의 수 (거리와 풍향각)

# 랜덤한 거리 정보 (1에서 100까지)
distance = np.array([[  0,  89,   5,   0,   0,   0],
                    [89,   0,   0, 166,   0,   0],
                    [  5,   0,   0, 230, 163,   0],
                    [  0, 166,  230,   0,   0, 131],
                    [  0,   0, 163,   0,   131, 165],
                    [  0,   0,   0,  131,  165,   0]])
print('distance : ', distance)

# 랜덤한 풍향각 정보 (0에서 360까지)
wind_direction = np.array([[  0,  89,   5,   0,   0,   0],
                            [89,   0,   0, 166,   0,   0],
                            [  5,   0,   0, 230, 163,   0],
                            [  0, 166,  230,   0,   0, 131],
                            [  0,   0, 163,   0,   131, 165],
                            [  0,   0,   0,  131,  165,   0]])
print('wind_direction : ', wind_direction)

# 대각선 원소는 0으로 설정 (자기 자신으로의 거리는 0)
np.fill_diagonal(distance, 0)
np.fill_diagonal(wind_direction, 0)

# 인접 텐서 생성 (3D 텐서)
adjacency_tensor = np.zeros((num_nodes, num_nodes, num_features))

# 거리 정보를 첫 번째 특성으로 저장
adjacency_tensor[:, :, 0] = distance

# 풍향각 정보를 두 번째 특성으로 저장
#adjacency_tensor[:, :, 1] = wind_direction

# 생성된 인접 텐서 출력
print(adjacency_tensor)

# 그림 재출력을 위한 준비
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 엣지에 대한 거리와 풍향각 정보를 플롯하고 올바른 수선의 발을 표시
for i in range(num_nodes):
    for j in range(num_nodes):
        if graph[i, j] == 1:
            distance = adjacency_tensor[i, j, 0]
            #wind_direction = adjacency_tensor[i, j, 1]

            # 거리를 파란색(o)으로 표시하고 올바른 수선의 발을 표시
            ax.scatter(i, j, distance, c='b', marker='o')
            ax.plot([i, i], [j, j], [0, distance], 'b', linewidth=0.5, linestyle='--')

            # 풍향각을 빨간색(x)으로 표시하고 올바른 수선의 발을 표시
            #ax.scatter(i, j, wind_direction, c='r', marker='x')
            #ax.plot([i, i], [j, j], [0, wind_direction], 'r', linewidth=0.5, linestyle='--')

            # x, y축의 교점 위에 수선의 발을 표시
            ax.scatter(i, j, 0, c='k', marker='.')

ax.set_xlabel('Start Node')
ax.set_ylabel('End Node')
ax.set_zlabel('Feature Value')
ax.set_title('3D Tensor Visualization with Correct Perpendiculars')

# 범례 추가
ax.legend(['Distance', 'Perpendiculars for Distance', 'X-Y Axis Intersection'])

plt.show()
