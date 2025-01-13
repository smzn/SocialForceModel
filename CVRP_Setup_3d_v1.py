import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

class CVRP_Setup_3d:
    def __init__(self, num_clients, num_shelters, num_vehicles, demand_options, vehicle_capacity, area_size, min_distance, speed, mean_elevation=50, std_elevation=5):
        """
        CVRPの初期設定を行うクラス。
        """
        self.num_clients = num_clients
        self.num_shelters = num_shelters
        self.num_vehicles = num_vehicles
        self.demand_options = demand_options
        self.vehicle_capacity = vehicle_capacity
        self.area_size = area_size
        self.min_distance = min_distance
        self.speed = speed

        # 標高関連のパラメータ
        self.mean_elevation = mean_elevation
        self.std_elevation = std_elevation

        # 初期化時に nodes と vehicles を None に設定
        self.nodes = None
        self.vehicles = None

    def generate_nodes_and_vehicles(self, node_file="./init/nodes_3d.csv", vehicle_file="./init/vehicles_3d.csv"):
        """
        ノードと車両をランダム生成し、3次元座標対応のCSVに保存。
        """
        def generate_positions(num_points, area_size, min_distance, mean_elevation, std_elevation):
            """
            ランダムな位置 (x, y, z) を生成。
            """
            positions = []
            while len(positions) < num_points:
                x, y = random.uniform(0, area_size), random.uniform(0, area_size)
                z = max(0, random.gauss(mean_elevation, std_elevation))  # 標高は正規分布
                if all(np.sqrt((x - px)**2 + (y - py)**2) >= min_distance for px, py, _ in positions):
                    positions.append((x, y, z))
            return positions

        total_nodes = self.num_clients + self.num_shelters + 1
        positions = generate_positions(
            total_nodes,
            self.area_size,
            self.min_distance,
            self.mean_elevation,
            self.std_elevation,
        )

        # ノードの生成
        nodes = []
        for i, (x, y, z) in enumerate(positions):
            if i == 0:
                nodes.append({"id": i, "type": "city_hall", "x": x, "y": y, "z": z, "demand": 0})
            elif i <= self.num_shelters:
                nodes.append({"id": i, "type": "shelter", "x": x, "y": y, "z": z, "demand": 0})
            else:
                demand = random.choice(self.demand_options)
                nodes.append({"id": i, "type": "client", "x": x, "y": y, "z": z, "demand": demand})

        # 車両の生成
        vehicles = [{"id": i, "capacity": self.vehicle_capacity} for i in range(1, self.num_vehicles + 1)]

        self.nodes = nodes
        self.vehicles = vehicles

        # CSV保存
        self._save_to_csv(nodes, node_file, ["id", "type", "x", "y", "z", "demand"])
        self._save_to_csv(vehicles, vehicle_file, ["id", "capacity"])
        return nodes, vehicles

    def _save_to_csv(self, data, file_name, fieldnames):
        with open(file_name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved to {file_name}")

    def calculate_cost_matrix(self, cost_matrix_file='./init/travel_time_3d.csv'):
        """
        拠点の座標 (x, y, z) から移動時間のコスト行列を計算。
        """
        num_nodes = len(self.nodes)
        cost_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 3次元の距離を計算
                    dx = self.nodes[i]["x"] - self.nodes[j]["x"]
                    dy = self.nodes[i]["y"] - self.nodes[j]["y"]
                    dz = self.nodes[i]["z"] - self.nodes[j]["z"]
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # 距離から移動時間を計算
                    travel_time = distance / self.speed * 60  # 分単位
                    cost_matrix[i, j] = travel_time
        np.savetxt(cost_matrix_file, cost_matrix, delimiter=",", fmt="%.2f")
        print(f"Cost matrix saved to {cost_matrix_file}")
        return cost_matrix
    
    def plot_nodes(self, map_file='./init/node_map_3d_2d.png'):
        """
        ノードを種類別にプロットし、PNG形式で保存。
        """
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.viridis
        norm = plt.Normalize(1, max(node["demand"] for node in self.nodes if node["type"] == "client"))

        for node in self.nodes:
            if node["type"] == "city_hall":
                plt.scatter(node["x"], node["y"], color="blue", marker="s", s=100)
            elif node["type"] == "shelter":
                plt.scatter(node["x"], node["y"], color="green", marker="^", s=100)
            elif node["type"] == "client":
                color = cmap(norm(node["demand"]))
                plt.scatter(node["x"], node["y"], color=color, marker="o", s=100)

        # ノードの種類に対応する凡例
        type_legend_handles = [
            plt.Line2D([0], [0], color="blue", marker="s", linestyle="", markersize=10, label="City Hall"),
            plt.Line2D([0], [0], color="green", marker="^", linestyle="", markersize=10, label="Shelter"),
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", markersize=10, label="Client"),
        ]

        type_legend = plt.legend(handles=type_legend_handles, title="Node Type", loc="upper left", bbox_to_anchor=(1.0, 1))
        plt.gca().add_artist(type_legend)

        # クライアント需要に対応する凡例
        unique_demands = sorted(set(node["demand"] for node in self.nodes if node["type"] == "client"))
        demand_legend_handles = [
            plt.Line2D([0], [0], color=cmap(norm(demand)), marker="o", linestyle="", markersize=10, label=f"{demand}")
            for demand in unique_demands
        ]
        plt.legend(handles=demand_legend_handles, title="Client Demand", loc="upper left", bbox_to_anchor=(1.0, 0.5))

        # 凡例と装飾
        plt.xlabel("X Coordinate (km)")
        plt.ylabel("Y Coordinate (km)")
        plt.title("Node Locations")
        plt.grid(True)
        plt.savefig(map_file, dpi=300)
        print(f"Node map saved to {map_file}")
        plt.close()


    def plot_nodes_3d(self, map_file='./init/node_map_3d.png'):
        """
        ノードを種類別にプロットし、3DマップをPNG形式で保存。
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.cm.viridis
        norm = plt.Normalize(1, max(node["demand"] for node in self.nodes if node["type"] == "client"))

        for node in self.nodes:
            if node["type"] == "city_hall":
                ax.scatter(node["x"], node["y"], node["z"], color="blue", marker="s", s=100, label="City Hall")
            elif node["type"] == "shelter":
                ax.scatter(node["x"], node["y"], node["z"], color="green", marker="^", s=100, label="Shelter")
            elif node["type"] == "client":
                color = cmap(norm(node["demand"]))
                ax.scatter(node["x"], node["y"], node["z"], color=color, marker="o", s=100, label="Client")

        # 軸ラベル
        ax.set_xlabel("X Coordinate (km)")
        ax.set_ylabel("Y Coordinate (km)")
        ax.set_zlabel("Elevation (m)")

        # タイトル
        ax.set_title("3D Node Locations")

        # 凡例
        type_legend_handles = [
            plt.Line2D([0], [0], color="blue", marker="s", linestyle="", markersize=10, label="City Hall"),
            plt.Line2D([0], [0], color="green", marker="^", linestyle="", markersize=10, label="Shelter"),
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", markersize=10, label="Client"),
        ]
        ax.legend(handles=type_legend_handles, title="Node Type", loc="upper left", bbox_to_anchor=(1.0, 1.0))

        # クライアント需要に対応する凡例
        unique_demands = sorted(set(node["demand"] for node in self.nodes if node["type"] == "client"))
        demand_legend_handles = [
            plt.Line2D([0], [0], color=cmap(norm(demand)), marker="o", linestyle="", markersize=10, label=f"{demand}")
            for demand in unique_demands
        ]
        plt.legend(handles=demand_legend_handles, title="Client Demand", loc="upper left", bbox_to_anchor=(1.0, 0.5))

        # グリッド表示
        ax.grid(True)

        # ファイル保存
        plt.savefig(map_file, dpi=300)
        print(f"3D Node map saved to {map_file}")
        plt.close()

    def plot_contour_2d(self, contour_file='./init/node_contour_2d.png', resolution=100):
        """
        等高線を使ったノードの2次元表示をPNG形式で保存。
        等高線の線を追加し、標高の変化をなだらかに調整。
        """
        # グリッドを作成
        x = np.linspace(0, self.area_size, resolution)
        y = np.linspace(0, self.area_size, resolution)
        X, Y = np.meshgrid(x, y)

        # 標高データを補間
        Z = np.zeros_like(X)
        for node in self.nodes:
            Z += np.exp(-((X - node["x"])**2 + (Y - node["y"])**2) / (2 * (self.min_distance / 2)**2)) * node["z"]

        # スムージング（標高データのなだらかさを調整）
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=3)  # sigmaの値を調整すると滑らかさが変化

        # データをリスケーリング（最小値を0、最大値を1にスケール）
        Z_min, Z_max = Z.min(), Z.max()
        Z = (Z - Z_min) / (Z_max - Z_min)

        # カラーマップの設定（白から青系統に変化）
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 1, 1), (0.6, 0.8, 0.4), (0.2, 0.6, 0.8), (0.1, 0.3, 0.5)]
        custom_cmap = LinearSegmentedColormap.from_list("custom_terrain", colors, N=256)

        # 等高線プロット
        plt.figure(figsize=(12, 10))
        plt.style.use('default')  # 背景を白に設定
        contour_filled = plt.contourf(X, Y, Z, levels=20, cmap=custom_cmap, alpha=0.8)
        contour_lines = plt.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5)  # 等高線の線を描画
        colorbar = plt.colorbar(contour_filled, label="Normalized Elevation", location="left", pad=0.1)  # 凡例を左側に配置

        # ノードのプロット
        cmap = plt.cm.viridis
        norm = plt.Normalize(1, max(node["demand"] for node in self.nodes if node["type"] == "client"))
        for node in self.nodes:
            if node["type"] == "city_hall":
                plt.scatter(node["x"], node["y"], color="blue", marker="s", s=100)
            elif node["type"] == "shelter":
                plt.scatter(node["x"], node["y"], color="green", marker="^", s=100)
            elif node["type"] == "client":
                color = cmap(norm(node["demand"]))
                plt.scatter(node["x"], node["y"], color=color, marker="o", s=100)

        # ノードの種類に対応する凡例
        type_legend_handles = [
            plt.Line2D([0], [0], color="blue", marker="s", linestyle="", markersize=10, label="City Hall"),
            plt.Line2D([0], [0], color="green", marker="^", linestyle="", markersize=10, label="Shelter"),
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", markersize=10, label="Client"),
        ]
        type_legend = plt.legend(handles=type_legend_handles, title="Node Type", loc="upper left", bbox_to_anchor=(1.0, 1))
        plt.gca().add_artist(type_legend)

        # クライアント需要に対応する凡例
        unique_demands = sorted(set(node["demand"] for node in self.nodes if node["type"] == "client"))
        demand_legend_handles = [
            plt.Line2D([0], [0], color=cmap(norm(demand)), marker="o", linestyle="", markersize=10, label=f"{demand}")
            for demand in unique_demands
        ]
        plt.legend(handles=demand_legend_handles, title="Client Demand", loc="upper left", bbox_to_anchor=(1.0, 0.5))

        # ラベルと装飾
        plt.xlabel("X Coordinate (km)")
        plt.ylabel("Y Coordinate (km)")
        plt.title("2D Contour Map of Nodes")
        plt.grid(True)
        plt.savefig(contour_file, dpi=300, bbox_inches="tight")
        print(f"2D Contour map saved to {contour_file}")
        plt.close()

    def plot_contour_3d(self, contour_file='./init/node_contour_3d.png', resolution=100):
        """
        等高線の表面を3次元で表示し、各ノードを地形より少し浮かせてプロット。
        """

        # グリッドを作成
        x = np.linspace(0, self.area_size, resolution)
        y = np.linspace(0, self.area_size, resolution)
        X, Y = np.meshgrid(x, y)

        # 標高データを補間
        Z = np.zeros_like(X)
        for node in self.nodes:
            Z += np.exp(-((X - node["x"])**2 + (Y - node["y"])**2) / (2 * (self.min_distance / 2)**2)) * node["z"]

        # スムージング（標高データをなだらかに）
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=3)

        # データをリスケーリング（最小値を0、最大値を1にスケール）
        Z_min, Z_max = Z.min(), Z.max()
        Z = (Z - Z_min) / (Z_max - Z_min)

        # 各ノードの高さを地形データに基づいて計算
        def get_node_elevation(node_x, node_y):
            i = np.abs(x - node_x).argmin()
            j = np.abs(y - node_y).argmin()
            return Z[j, i]

        for node in self.nodes:
            node["elevation"] = get_node_elevation(node["x"], node["y"]) + 0.02  # 地形より少し浮かせる

        # カラーマップの設定（白から青系統に変化）
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(1, 1, 1), (0.6, 0.8, 0.4), (0.2, 0.6, 0.8), (0.1, 0.3, 0.5)]
        custom_cmap = LinearSegmentedColormap.from_list("custom_terrain", colors, N=256)

        # 3Dプロット
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap=custom_cmap, edgecolor='none', alpha=0.5)  # 地形の透明度を調整

        # 等高線の線を追加
        ax.contour(X, Y, Z, levels=20, colors="black", linestyles="solid", linewidths=0.5)

        # ノードのプロット
        cmap = plt.cm.viridis
        norm = plt.Normalize(1, max(node["demand"] for node in self.nodes if node["type"] == "client"))
        for node in self.nodes:
            if node["type"] == "city_hall":
                ax.scatter(node["x"], node["y"], node["elevation"], color="blue", marker="s", s=120, label="City Hall")
            elif node["type"] == "shelter":
                ax.scatter(node["x"], node["y"], node["elevation"], color="green", marker="^", s=120, label="Shelter")
            elif node["type"] == "client":
                color = cmap(norm(node["demand"]))
                ax.scatter(node["x"], node["y"], node["elevation"], color=color, marker="o", s=120, label="Client")

        # 軸ラベルとカラーバー
        ax.set_xlabel("X Coordinate (km)", labelpad=10)
        ax.set_ylabel("Y Coordinate (km)", labelpad=10)
        ax.set_zlabel("Normalized Elevation", labelpad=10)
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label="Normalized Elevation")

        # 凡例を追加
        type_legend_handles = [
            plt.Line2D([0], [0], color="blue", marker="s", linestyle="", markersize=10, label="City Hall"),
            plt.Line2D([0], [0], color="green", marker="^", linestyle="", markersize=10, label="Shelter"),
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", markersize=10, label="Client"),
        ]
        ax.legend(handles=type_legend_handles, title="Node Type", loc="upper left", bbox_to_anchor=(1.0, 1.0))

        # タイトル
        ax.set_title("3D Contour Map of Nodes")
        plt.savefig(contour_file, dpi=300, bbox_inches="tight")
        print(f"3D Contour map saved to {contour_file}")
        plt.close()
