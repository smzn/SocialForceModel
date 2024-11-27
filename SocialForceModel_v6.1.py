import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class SocialForceModel:
    def __init__(self, transition_file_path, destination_file_path, agent_parameter_file_path, time_step, total_time):
        # 推移確率行列をファイルから読み込む
        transition_df = pd.read_csv(transition_file_path, index_col=0)
        self.original_transition_matrix = transition_df.values  # 元の推移確率行列 (num_nodes * num_classes, num_nodes * num_classes)

        # 目的地情報をファイルから読み込む
        destination_df = pd.read_csv(destination_file_path)
        self.destinations = destination_df[['x', 'y']].values  # x, y座標を取得
        self.num_nodes = len(self.destinations)  # 拠点数を取得
        self.service_time_parameters = destination_df['lambda'].values  # サービスレートを取得
        self.num_servers = destination_df['num_servers'].values.astype(int)  # サーバー数を取得

        # クラス数を計算
        num_rows = self.original_transition_matrix.shape[0]
        self.num_classes = num_rows // self.num_nodes  # クラス数 = 行数 / 拠点数

        # 拠点数とクラス数を確認のため表示
        print(f"拠点数: {self.num_nodes}")
        print(f"クラス数: {self.num_classes}")

                # 推移確率行列を拠点単位に集約
        self.transition_matrix = self.aggregate_transition_matrix()

        # エージェントパラメータを読み込み
        self.agent_parameters = pd.read_csv(agent_parameter_file_path)
        # 初期位置、パラメータをCSVから設定
        self.positions = self.agent_parameters[['x', 'y']].values.astype(float)
        self.masses = self.agent_parameters['masses'].values
        self.reaction_times = self.agent_parameters['reaction_times'].values
        self.desired_speeds = self.agent_parameters['desired_speeds'].values
        self.radii = self.agent_parameters['radi'].values
        self.A = self.agent_parameters['a'].values
        self.B = self.agent_parameters['b'].values
        self.current_destinations = self.agent_parameters['initial_destination'].values
        self.num_agents = len(self.agent_parameters)  # CSVファイルの行数をエージェント数として設定
        # ファイルから障害物反発力を取得
        self.obstacle_A = self.agent_parameters['obstacle_a'].values
        self.obstacle_B = self.agent_parameters['obstacle_b'].values

        self.servers_available = self.num_servers.copy()  # 各拠点の利用可能なサーバー数
        self.is_served = np.zeros(self.num_agents, dtype=bool)  # 各エージェントのサービス状態
                
        self.time_step = time_step
        self.total_time = total_time
        self.arrived_agents = np.zeros(self.num_agents, dtype=bool)
        self.arrival_times = np.full(self.num_agents, np.nan)
        
        # 既存のパラメータ初期化
        self.velocities = np.zeros((self.num_agents, 2))
        self.service_times = np.zeros(self.num_agents)  # サービス時間の初期化
        self.agent_states = ["move"] * self.num_agents  # 初期状態は全エージェントが移動中

        self.obstacles = {
            "lines": [  # 直線障害物
                {"a": 1, "b": -1, "c": 5, "start": (10, 5), "end": (10, 10)},
                {"a": 0, "b": 1, "c": -10, "start": (2, 10), "end": (7, 10)}
            ],
            "polygons": [  # 多角形障害物
                {"vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                {"vertices": [(8, 8), (9, 7), (10, 8)]}
            ],
            "circles": [  # 円障害物
                {"center": (15, 15), "radius": 1},
                {"center": (5, 15), "radius": 1}
            ]
        }

        # 障害物用パラメータの初期化
        total_obstacles = (
            len(self.obstacles["lines"]) +
            len(self.obstacles["polygons"]) +
            len(self.obstacles["circles"])
        )
        
        # 履歴データの初期化
        self.history = []
        self.destination_history = []  # 目的地履歴を追加
        self.velocity_history = np.zeros((int(self.total_time / self.time_step), self.num_agents))
        self.desire_force_history = np.zeros((int(self.total_time / self.time_step), self.num_agents))
        self.repulsion_force_history = np.zeros((int(self.total_time / self.time_step), self.num_agents))

    def aggregate_transition_matrix(self):
        """
        推移確率行列をクラス単位から拠点単位に集約します（対角要素のみ使用）。

        Returns:
            np.ndarray: 集約された推移確率行列 (num_nodes, num_nodes)
        """
        aggregated_matrix = np.zeros((self.num_nodes, self.num_nodes))

        # 対角部分行列を抽出して集計
        for i in range(self.num_classes):
            start_idx = i * self.num_nodes
            end_idx = (i + 1) * self.num_nodes
            class_submatrix = self.original_transition_matrix[start_idx:end_idx, start_idx:end_idx]
            aggregated_matrix += class_submatrix

        # 行を正規化して確率行列にする
        row_sums = aggregated_matrix.sum(axis=1, keepdims=True)
        aggregated_matrix = np.divide(
            aggregated_matrix, 
            row_sums,
            out=np.zeros_like(aggregated_matrix),  # ゼロ除算防止
            where=row_sums != 0
        )

        return aggregated_matrix

    def generate_random_position_away_from_target(self, destination, min_distance=10, max_range=20):
        while True:
            position = np.random.rand(2) * max_range
            distance_to_destination = np.linalg.norm(position - destination)
            if distance_to_destination > min_distance:
                return position

    def select_next_destination(self, agent_idx):
        """エージェントの次の目的地を確率的に選択"""
        current_dest = self.current_destinations[agent_idx]
        probabilities = self.transition_matrix[current_dest]
        return np.random.choice(len(self.destinations), p=probabilities)

    def calculate_desire_force(self, agent_idx):
        # 現在の目的地に向かう方向を計算
        current_destination = self.destinations[self.current_destinations[agent_idx]]
        direction = current_destination - self.positions[agent_idx]
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
        desired_velocity = self.desired_speeds[agent_idx] * direction
        return (self.masses[agent_idx] / self.reaction_times[agent_idx]) * (desired_velocity - self.velocities[agent_idx])

    def calculate_repulsion_force(self, agent_idx):
        repulsion_force = np.zeros(2)
        for j in range(self.num_agents):
            if j != agent_idx:
                d_ij = self.positions[agent_idx] - self.positions[j]
                distance = np.linalg.norm(d_ij)
                if distance > 0:
                    n_ij = d_ij / distance
                    r_ij = self.radii[agent_idx] + self.radii[j]
                    repulsion_force += self.A[agent_idx] * np.exp((r_ij - distance) / self.B[agent_idx]) * n_ij
        return np.linalg.norm(repulsion_force)
    
    def calculate_obstacle_repulsion_force(self, agent_idx):
        """エージェントと障害物の反発力を計算"""
        position = self.positions[agent_idx]
        repulsion_force = np.zeros(2)  # 初期化

        # 障害物ごとの反発力を計算
        for idx, line in enumerate(self.obstacles["lines"]):
            a, b, c = line["a"], line["b"], line["c"]
            distance = abs(a * position[0] + b * position[1] + c) / np.sqrt(a**2 + b**2)
            if distance > 0:
                n_w = np.array([a, b]) / np.sqrt(a**2 + b**2)
                r_i = self.radii[agent_idx]
                A_i = self.obstacle_A[idx]
                B_i = self.obstacle_B[idx]
                repulsion_force += A_i * np.exp((r_i - distance) / B_i) * n_w

        for idx, polygon in enumerate(self.obstacles["polygons"]):
            vertices = polygon["vertices"]
            min_distance = float('inf')
            closest_segment_n_w = np.zeros(2)
            for j in range(len(vertices)):
                start, end = vertices[j], vertices[(j + 1) % len(vertices)]
                distance, n_w = self.point_to_segment_repulsion(position, start, end)
                if distance < min_distance:
                    min_distance = distance
                    closest_segment_n_w = n_w
            r_i = self.radii[agent_idx]
            A_i = self.obstacle_A[len(self.obstacles["lines"]) + idx]
            B_i = self.obstacle_B[len(self.obstacles["lines"]) + idx]
            repulsion_force += A_i * np.exp((r_i - min_distance) / B_i) * closest_segment_n_w

        for idx, circle in enumerate(self.obstacles["circles"]):
            center = np.array(circle["center"])
            radius = circle["radius"]
            distance = np.linalg.norm(position - center) - radius
            if distance > 0:
                n_w = (center - position) / np.linalg.norm(center - position)
                r_i = self.radii[agent_idx]
                A_i = self.obstacle_A[len(self.obstacles["lines"]) + len(self.obstacles["polygons"]) + idx]
                B_i = self.obstacle_B[len(self.obstacles["lines"]) + len(self.obstacles["polygons"]) + idx]
                repulsion_force += A_i * np.exp((r_i - distance) / B_i) * n_w

        return repulsion_force

    def point_to_segment_repulsion(self, point, start, end):
        """点と線分間の最短距離と単位方向ベクトルを計算"""
        line_vec = np.array(end) - np.array(start)
        point_vec = np.array(point) - np.array(start)
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            n_w = (start - point) / np.linalg.norm(start - point)
            return np.linalg.norm(point - start), n_w
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
        projection = np.array(start) + t * line_vec
        n_w = (projection - point) / np.linalg.norm(projection - point)
        return np.linalg.norm(point - projection), n_w

    def update_positions(self):
        num_steps = int(self.total_time / self.time_step)
        self.total_forces_history = []
        self.desire_forces_history = []
        self.repulsion_forces_history = []
        self.agent_repulsion_info = []
        self.obstacle_distance_info = []  # 障害物との距離情報を記録するリスト

        for step in range(num_steps):
            forces = np.zeros((self.num_agents, 2))
            
            for i in range(self.num_agents):
                # サービス待機中のエージェントは移動せず、サービス時間を減らす
                if self.service_times[i] > 0:
                    self.service_times[i] -= self.time_step  # サービス時間を減少させる
                    self.agent_states[i] = "served"
                    # サービス中のエージェントに対する履歴にデフォルト値を追加
                    self.desire_forces_history.append([step * self.time_step, i + 1, 0, 0])
                    self.repulsion_forces_history.append([step * self.time_step, i + 1, 0])
                    self.total_forces_history.append([step * self.time_step, i + 1, 0, 0])
                    self.velocity_history[step, i] = np.linalg.norm(self.velocities[i])  # 速度をそのまま保持
                    # agent_repulsion_infoにデフォルト値を追加（例として0やNaNを使用）
                    self.agent_repulsion_info.append([
                        step * self.time_step, i + 1, None, 0, 0, None, 0, 0, 0
                    ])
                    continue  # 移動や力の計算をスキップ

                # サービス終了後にサーバーを解放
                if self.service_times[i] <= 0 and self.is_served[i]:
                    self.servers_available[self.current_destinations[i]] += 1
                    self.is_served[i] = False
                    # 新しい目的地を選択
                    self.current_destinations[i] = self.select_next_destination(i)
                    print(f"Agent {i+1} reached destination and is now heading to destination {self.current_destinations[i]+1}")
                    self.agent_states[i] = "move"

                # 目的地到達チェックと更新
                current_destination = self.destinations[self.current_destinations[i]]
                distance_to_destination = np.linalg.norm(self.positions[i] - current_destination)
                
                if distance_to_destination < 0.1:  # 目的地に到達
                    if self.servers_available[self.current_destinations[i]] > 0:
                        # サーバーが空いている場合にサービス時間を設定
                        self.servers_available[self.current_destinations[i]] -= 1
                        self.service_times[i] = np.random.exponential(
                            self.service_time_parameters[self.current_destinations[i]]
                        )
                        self.is_served[i] = True
                        self.agent_states[i] = "served"
                    else:
                        # サーバーが空いていない場合、待機
                        self.service_times[i] = 0
                        self.is_served[i] = False
                        self.agent_states[i] = "queue"
                
                # 既存の力の計算
                desire_force = self.calculate_desire_force(i)
                repulsion_force = self.calculate_repulsion_force(i)
                # 障害物からの反発力を計算
                obstacle_repulsion_force = self.calculate_obstacle_repulsion_force(i)
                total_force = desire_force + repulsion_force + obstacle_repulsion_force
                forces[i] = total_force

                # 既存の履歴記録
                self.desire_forces_history.append([step * self.time_step, i + 1, *desire_force])
                self.repulsion_forces_history.append([step * self.time_step, i + 1, repulsion_force])
                self.total_forces_history.append([step * self.time_step, i + 1, *total_force])
                
                self.desire_force_history[step, i] = np.linalg.norm(desire_force)
                self.repulsion_force_history[step, i] = repulsion_force

                # 障害物との距離を計算
                distances_to_obstacles = self.calculate_obstacle_distances(self.positions[i])
                
                # 障害物との距離を履歴に記録
                self.obstacle_distance_info.append(
                    [step * self.time_step, i + 1, *distances_to_obstacles]
                )

                # エージェント間の反発力情報の計算（既存のコード）
                max_repulsion_force = -np.inf
                max_repulsion_agent_id = -1
                max_repulsion_distance = -1
                min_distance = np.inf
                min_distance_agent_id = -1
                total_repulsion_force = 0.0
                total_distance = 0.0
                count = 0

                for j in range(self.num_agents):
                    if i != j:
                        distance = np.linalg.norm(self.positions[i] - self.positions[j])
                        total_distance += distance
                        count += 1
                        if distance > 0:
                            n_ij = (self.positions[i] - self.positions[j]) / distance
                            r_ij = self.radii[i] + self.radii[j]
                            temp_repulsion_force = self.A[i] * np.exp((r_ij - distance) / self.B[i]) * np.linalg.norm(n_ij)
                            total_repulsion_force += temp_repulsion_force
                            
                            if temp_repulsion_force > max_repulsion_force:
                                max_repulsion_force = temp_repulsion_force
                                max_repulsion_agent_id = j + 1
                                max_repulsion_distance = distance

                        if distance < min_distance:
                            min_distance = distance
                            min_distance_agent_id = j + 1

                avg_repulsion_force = total_repulsion_force / count if count > 0 else 0
                avg_distance = total_distance / count if count > 0 else 0

                self.agent_repulsion_info.append([
                    step * self.time_step, i + 1, max_repulsion_agent_id, max_repulsion_force,
                    max_repulsion_distance, min_distance_agent_id, min_distance,
                    avg_repulsion_force, avg_distance
                ])

            # 位置と速度の更新
            for i in range(self.num_agents):
                if self.service_times[i] > 0:
                    continue  # サービス待機中のエージェントは更新しない
                acceleration = forces[i] / self.masses[i]
                self.velocities[i] += acceleration * self.time_step
                self.positions[i] += self.velocities[i] * self.time_step
                self.velocity_history[step, i] = np.linalg.norm(self.velocities[i])


            # 履歴の保存
            self.history.append(self.positions.copy())
            self.destination_history.append(self.current_destinations.copy())

            # 進捗表示
            print(f"Step {step+1}: Simulation in progress...")

    def save_to_excel(self, csv_file, filepath):
        # 既存のExcel保存機能
        data = []
        for step, positions in enumerate(self.history):
            for agent_idx in range(self.num_agents):
                time = step * self.time_step
                desire_force = self.desire_forces_history[step * self.num_agents + agent_idx][2:]
                repulsion_force = self.repulsion_forces_history[step * self.num_agents + agent_idx][2]
                total_force = self.total_forces_history[step * self.num_agents + agent_idx][2:]
                agent_info = self.agent_repulsion_info[step * self.num_agents + agent_idx]

                # 現在の目的地情報を追加
                current_dest = self.destination_history[step][agent_idx]
                current_dest_coords = self.destinations[current_dest]

                # サービス時間を追加
                remaining_service_time = self.service_times[agent_idx]

                data.append([
                    time, agent_idx + 1, *positions[agent_idx],
                    *self.velocities[agent_idx], False,  # 到達フラグは常にFalse（複数目的地対応）
                    *desire_force, repulsion_force, *total_force,
                    agent_info[2], agent_info[3], agent_info[4],
                    agent_info[5], agent_info[6], agent_info[7], agent_info[8],
                    current_dest + 1,  # 現在の目的地ID
                    *current_dest_coords,  # 現在の目的地座標
                    remaining_service_time,  # 残りサービス時間
                    self.agent_states[agent_idx]
                ])

        # カラム名を更新（目的地情報を追加）
        columns = [
            "時間", "エージェントID", "位置X", "位置Y", "速度X", "速度Y", "到達フラグ",
            "目的志向力X", "目的志向力Y", "反発力", "合力X", "合力Y",
            "最大反発力エージェントID", "最大反発力", "最大反発力エージェントとの距離",
            "最短距離エージェントID", "最短距離", "平均反発力", "平均距離",
            "現在の目的地ID", "目的地X", "目的地Y", "残りサービス時間", "状態"
        ]
        df_combined = pd.DataFrame(data, columns=columns)

        # 障害物との距離データをDataFrameに変換
        obstacle_columns = [f"障害物{i+1}との距離" for i in range(len(self.obstacles["lines"]) +
                                                        len(self.obstacles["polygons"]) +
                                                        len(self.obstacles["circles"]))]
        df_obstacle_distances = pd.DataFrame(
            self.obstacle_distance_info,
            columns=["時間", "エージェントID"] + obstacle_columns
        )
        # 時間とエージェントIDで結合
        df_combined = df_combined.merge(df_obstacle_distances, on=["時間", "エージェントID"], how="left")

        #csv_file = 'sfm_log.csv'
        df_combined.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}")

        # パラメータ情報を更新
        df_params = pd.DataFrame({
            'エージェントID': np.arange(1, self.num_agents + 1),
            '初期位置X': self.positions[:, 0],
            '初期位置Y': self.positions[:, 1],
            '質量': self.masses,
            '反応時間': self.reaction_times,
            '希望速度': self.desired_speeds,
            '初期目的地ID': self.current_destinations + 1
        })

        # 目的地情報を追加
        df_destinations = pd.DataFrame({
            '目的地ID': np.arange(1, len(self.destinations) + 1),
            '目的地X': self.destinations[:, 0],
            '目的地Y': self.destinations[:, 1],
            '指数分布パラメータλ': self.service_time_parameters,
            'サーバー数': self.num_servers  # サーバー数を追加
        })

        # 推移確率行列をDataFrameに変換
        df_transition = pd.DataFrame(
            self.transition_matrix,
            columns=[f'目的地{i+1}への推移確率' for i in range(len(self.destinations))],
            index=[f'目的地{i+1}から' for i in range(len(self.destinations))]
        )

        # 障害物情報を追加
        obstacle_data = []
        for idx, obstacle in enumerate(self.obstacles["lines"]):
            obstacle_data.append([
                "Line", idx + 1, obstacle["start"], obstacle["end"], 
                self.obstacle_A[idx], self.obstacle_B[idx]
            ])
        for idx, obstacle in enumerate(self.obstacles["polygons"]):
            obstacle_data.append([
                "Polygon", idx + 1, obstacle["vertices"], None, 
                self.obstacle_A[idx + len(self.obstacles["lines"])], 
                self.obstacle_B[idx + len(self.obstacles["lines"])]
            ])
        for idx, obstacle in enumerate(self.obstacles["circles"]):
            obstacle_data.append([
                "Circle", idx + 1, obstacle["center"], obstacle["radius"], 
                self.obstacle_A[idx + len(self.obstacles["lines"]) + len(self.obstacles["polygons"])], 
                self.obstacle_B[idx + len(self.obstacles["lines"]) + len(self.obstacles["polygons"])]
            ])

        df_obstacles = pd.DataFrame(obstacle_data, columns=[
            "Type", "ID", "Coordinates", "Extra Info", "Repulsion Strength (A)", "Repulsion Range (B)"
        ])

        # 平均系内人数の計算
        avg_queue_with_moving, avg_queue_without_moving = self.calculate_average_queue_length()
        
        # 平均系内人数のDataFrame作成（移動中を含む）
        queue_data_with_moving = []
        for node in range(self.num_nodes):
            for class_idx in range(self.num_classes):
                queue_data_with_moving.append({
                    '拠点ID': node + 1,
                    'クラス': class_idx + 1,
                    '平均系内人数（移動中含む）': avg_queue_with_moving[node, class_idx]
                })
        df_queue_with_moving = pd.DataFrame(queue_data_with_moving)

        # 平均系内人数のDataFrame作成（移動中を含まない）
        queue_data_without_moving = []
        for node in range(self.num_nodes):
            for class_idx in range(self.num_classes):
                queue_data_without_moving.append({
                    '拠点ID': node + 1,
                    'クラス': class_idx + 1,
                    '平均系内人数（移動中除く）': avg_queue_without_moving[node, class_idx]
                })
        df_queue_without_moving = pd.DataFrame(queue_data_without_moving)

        # DataFrameの結合
        df_queue_combined = pd.merge(
            df_queue_with_moving,
            df_queue_without_moving[['拠点ID', 'クラス', '平均系内人数（移動中除く）']],
            on=['拠点ID', 'クラス']
        )

        with pd.ExcelWriter(filepath) as writer:

            #df_combined.to_excel(writer, sheet_name='シミュレーション履歴と力', index=False)
            df_params.to_excel(writer, sheet_name='パラメータ', index=False)
            df_destinations.to_excel(writer, sheet_name='目的地情報', index=False)
            df_transition.to_excel(writer, sheet_name='推移確率行列')
            df_obstacles.to_excel(writer, sheet_name='障害物情報', index=False)
            # 新しい平均系内人数のシートを追加
            df_queue_combined.to_excel(writer, sheet_name='平均系内人数', index=False)

    def plot_movement(self, save_path='agent_movements.png'):
        history_np = np.array(self.history)
        plt.figure(figsize=(10, 10))
        
        # エージェントの軌跡をプロット
        for i in range(self.num_agents):
            plt.plot(history_np[:, i, 0], history_np[:, i, 1], label=f"Agent {i + 1}", alpha=0.5)
        
        # 全目的地をプロット
        for i, dest in enumerate(self.destinations):
            plt.scatter(dest[0], dest[1], color='red', marker='x', s=100, label=f"Destination {i + 1}")
        
        plt.title("Agent Movements")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # 既存の plot_velocity_changes メソッドはそのまま保持
    def plot_velocity_changes(self):
        time_steps = np.arange(0, self.total_time, self.time_step)
        plt.figure(figsize=(10, 6))
        
        for i in range(self.num_agents):
            plt.plot(time_steps, self.velocity_history[:, i], label=f"Agent {i + 1}")

        plt.title("Velocity Changes Over Time")

    # 新しい距離計算メソッドを追加
    def calculate_obstacle_distances(self, position):
        """エージェントの位置から障害物との距離を計算"""
        if not self.obstacles:  # 障害物が存在しない場合
            return []
        distances = []

        # 直線障害物との距離
        for line in self.obstacles["lines"]:
            a, b, c = line["a"], line["b"], line["c"]
            distance = abs(a * position[0] + b * position[1] + c) / np.sqrt(a**2 + b**2)
            distances.append(distance)

        # 多角形障害物との距離（最も近い辺との距離）
        for polygon in self.obstacles["polygons"]:
            vertices = polygon["vertices"]
            min_distance = float('inf')
            for j in range(len(vertices)):
                start, end = vertices[j], vertices[(j + 1) % len(vertices)]
                min_distance = min(min_distance, self.point_to_segment_distance(position, start, end))
            distances.append(min_distance)

        # 円障害物との距離
        for circle in self.obstacles["circles"]:
            center = np.array(circle["center"])
            radius = circle["radius"]
            distance = max(0, np.linalg.norm(position - center) - radius)
            distances.append(distance)

        return distances

    # 線分と点の距離を計算するユーティリティメソッド
    def point_to_segment_distance(self, point, start, end):
        """点と線分の最短距離を計算"""
        line_vec = np.array(end) - np.array(start)
        point_vec = np.array(point) - np.array(start)
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
        projection = np.array(start) + t * line_vec
        return np.linalg.norm(point - projection)
    
    def plot_movement_with_obstacles(self, save_path='agent_movements_with_obstacles.png'):
        history_np = np.array(self.history)
        plt.figure(figsize=(10, 10))

        # エージェントの軌跡をプロット
        for i in range(self.num_agents):
            plt.plot(history_np[:, i, 0], history_np[:, i, 1], label=f"Agent {i + 1}", alpha=0.5)

        # 全目的地をプロット
        for i, dest in enumerate(self.destinations):
            plt.scatter(dest[0], dest[1], color='red', marker='x', s=100, label=f"Destination {i + 1}")

        # 障害物を描画
        for line in self.obstacles["lines"]:
            plt.plot([line["start"][0], line["end"][0]], [line["start"][1], line["end"][1]], 'k-', label="Obstacle (Line)")
        for polygon in self.obstacles["polygons"]:
            vertices = polygon["vertices"] + [polygon["vertices"][0]]  # 閉じる
            vertices = np.array(vertices)
            plt.plot(vertices[:, 0], vertices[:, 1], 'b-', label="Obstacle (Polygon)")
        for circle in self.obstacles["circles"]:
            circle_artist = plt.Circle(circle["center"], circle["radius"], color='gray', alpha=0.5)
            plt.gca().add_artist(circle_artist)

        plt.title("Agent Movements with Obstacles")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def calculate_average_queue_length(self):
        """拠点ごとのクラス別平均系内人数を計算"""
        num_steps = len(self.history)
        
        # 移動中を含む系内人数の集計用
        queue_length_with_moving = np.zeros((self.num_nodes, self.num_classes))
        # サービス中・待ち行列のみの系内人数の集計用
        queue_length_without_moving = np.zeros((self.num_nodes, self.num_classes))
        
        # 各時間ステップでの集計
        for step in range(num_steps):
            # 各拠点・クラスごとの人数をカウント
            for i in range(self.num_agents):
                current_dest = self.destination_history[step][i]
                agent_class = i % self.num_classes  # エージェントのクラス
                agent_state = self.agent_states[i]
                
                # 移動中を含む場合は、目的地に向かっているエージェントもカウント
                queue_length_with_moving[current_dest, agent_class] += 1
                
                # 移動中を含まない場合は、サービス中または待機中のエージェントのみカウント
                if agent_state in ["served", "queue"]:
                    queue_length_without_moving[current_dest, agent_class] += 1
        
        # 平均を計算
        average_queue_length_with_moving = queue_length_with_moving / num_steps
        average_queue_length_without_moving = queue_length_without_moving / num_steps
        
        return average_queue_length_with_moving, average_queue_length_without_moving


# シミュレーションの実行コード
if __name__ == "__main__":
    # ファイルパスの指定
    transition_file = './BCMP/transition_probability_N33_R2_K100_Core8.csv'
    destination_file = './BCMP/destination.csv'
    agent_parameter_file = './BCMP/agent_parameter.csv'

    # シミュレーションパラメータの設定
    total_time = 10000     # シミュレーション時間
    time_step = 0.1      # 時間ステップ
    
    # SocialForceModelのインスタンスを作成
    sfm = SocialForceModel(
        transition_file_path=transition_file,
        destination_file_path=destination_file,
        agent_parameter_file_path=agent_parameter_file,
        time_step=time_step,
        total_time=total_time
    )

    # 実行開始時刻を記録
    start_time = time.time()

    # シミュレーションの実行
    print("Starting simulation...")
    sfm.update_positions()
    print("Simulation completed.")

    # 結果の保存と可視化
    print("Saving results...")
    
    # Excelファイルに結果を保存
    sfm.save_to_excel('sfm_log.csv','SocialForceModel_result_10000.xlsx')
    
    # 移動軌跡のプロット
    sfm.plot_movement('agent_movements_multiple_destinations.png')
    
    # 速度変化のプロット
    sfm.plot_velocity_changes()

    sfm.plot_movement_with_obstacles()
    
    print("Results saved and plots generated.")

    # 実行終了時刻を記録
    end_time = time.time()

    # 実行時間を表示
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
