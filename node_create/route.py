from pymongo import MongoClient
from flask import Flask, jsonify
import heapq
import math
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


        # MongoDB Connection
        client = MongoClient('mongodb+srv://srlabmongodb:mongodb1234@pathmaker.4frvxqx.mongodb.net/test')
        db = client['usg']
        collectionGps = db['gps']

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

        while current_node != start_node:
            shortest_path.append(current_node)
            current_node = paths[current_node]

        shortest_path.append(start_node)
        shortest_path.reverse()

        print("Shortest path:", shortest_path)
        print("Total distance:", distances[end_node])

        return jsonify({"shortest_path": shortest_path, "total_distance": distances[end_node]})
    except Exception as e:
            return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()