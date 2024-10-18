import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SocialForceModel:
    def __init__(self, num_agents=3, destination=(10, 10), time_step=0.1, total_time=10):
        self.num_agents = num_agents  # エージェントの数
        self.destination = np.array(destination)  # 目的地の座標
        self.time_step = time_step  # シミュレーションの時間ステップ
        self.total_time = total_time  # シミュレーションの合計時間
        self.arrived_agents = np.zeros(num_agents, dtype=bool)  # エージェントが到達したかどうかを追跡する配列
        self.arrival_times = np.full(num_agents, np.nan)  # 到着時間を記録する配列（初期値はNaN）
        
        # エージェントの初期位置を目的地から一定距離離してランダムに設定
        self.positions = np.array([self.generate_random_position_away_from_target(self.destination) for _ in range(self.num_agents)])
        self.velocities = np.zeros((num_agents, 2))  # 初期速度は0とする
        self.masses = np.random.uniform(70, 80, num_agents)  # 各エージェントの質量（70kg～80kgの範囲でランダム）
        self.reaction_times = np.random.uniform(0.5, 0.7, num_agents)  # 反応時間（0.5秒～0.7秒）
        self.desired_speeds = np.random.uniform(1.0, 1.5, num_agents)  # 希望速度（1.0～1.5 m/s）

        # 反発力のパラメータ
        self.A = np.random.uniform(10.0, 20.0, num_agents)  # 反発力の強さ（5.0～10.0）
        self.B = np.random.uniform(1.0, 2.0, num_agents)  # 反発力の範囲（1.0～2.0）
        self.radii = np.random.uniform(0.2, 0.5, num_agents)  # 各エージェントの半径（ランダム）
        
        # 履歴を格納するリスト
        self.history = []

        # 時系列の速度変化を記録するリスト
        self.velocity_history = np.zeros((int(self.total_time / self.time_step), self.num_agents))  # 各エージェントの速度履歴
        self.desire_force_history = np.zeros((int(self.total_time / self.time_step), self.num_agents))  # 各エージェントの目的志向力の履歴
        self.repulsion_force_history = np.zeros((int(self.total_time / self.time_step), self.num_agents))  # 各エージェントの反発力の履歴

    def generate_random_position_away_from_target(self, destination, min_distance=10, max_range=20):
        # 目標地点からmin_distance以上離れた位置をランダムに生成
        while True:
            position = np.random.rand(2) * max_range  # ランダムな位置を生成
            distance_to_destination = np.linalg.norm(position - destination)
            if distance_to_destination > min_distance:  # 目標地点から一定以上離れているか確認
                return position

    # 目的志向力の計算
    def calculate_desire_force(self, agent_idx):
        # エージェントが目的地に向かうための方向ベクトルを計算
        direction = self.destination - self.positions[agent_idx]
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance  # 方向ベクトルを正規化
        desired_velocity = self.desired_speeds[agent_idx] * direction
        # 目的志向力の計算
        return (self.masses[agent_idx] / self.reaction_times[agent_idx]) * (desired_velocity - self.velocities[agent_idx])
    
    # 歩行者間の反発力の計算
    def calculate_repulsion_force(self, agent_idx):
        repulsion_force = np.zeros(2)
        for j in range(self.num_agents):
            if j != agent_idx:
                # 歩行者間の距離を計算
                d_ij = self.positions[agent_idx] - self.positions[j]
                distance = np.linalg.norm(d_ij)
                if distance > 0:
                    n_ij = d_ij / distance  # 単位ベクトルを計算
                    r_ij = self.radii[agent_idx] + self.radii[j]  # 合成半径を計算
                    # 反発力を計算
                    repulsion_force += self.A[agent_idx] * np.exp((r_ij - distance) / self.B[agent_idx]) * n_ij
        return np.linalg.norm(repulsion_force)

    # エージェントの位置を更新する関数
    def update_positions(self):
        num_steps = int(self.total_time / self.time_step)  # 総ステップ数の計算
        for step in range(num_steps):
            forces = np.zeros((self.num_agents, 2))  # 各エージェントの合力を初期化
            
            # 各エージェントに対して目的志向力および反発力を計算
            for i in range(self.num_agents):
                desire_force = self.calculate_desire_force(i)
                repulsion_force = self.calculate_repulsion_force(i)
                forces[i] = desire_force + repulsion_force  # 目的志向力と反発力を加える

                # 目的志向力と反発力を記録
                self.desire_force_history[step, i] = np.linalg.norm(desire_force)
                self.repulsion_force_history[step, i] = repulsion_force

            # 各エージェントの速度と位置を更新
            for i in range(self.num_agents):
                acceleration = forces[i] / self.masses[i]  # 加速度を計算
                self.velocities[i] += acceleration * self.time_step  # 速度の更新
                self.positions[i] += self.velocities[i] * self.time_step  # 位置の更新

                # 速度を記録
                self.velocity_history[step, i] = np.linalg.norm(self.velocities[i])

                # 目的地に到達したかどうかを確認
                #if np.linalg.norm(self.positions[i] - self.destination) < 0.1:  # 0.1以下なら到達とみなす
                #    self.arrived_agents[i] = True  # 到達フラグを立てる
                #    print(f"Agent {i+1} has arrived at the destination.")
                # 目的地に到達したかどうかを確認
                if not self.arrived_agents[i] and np.linalg.norm(self.positions[i] - self.destination) < 0.1:  # 0.1以下なら到達とみなす
                    self.arrived_agents[i] = True  # 到達フラグを立てる
                    self.arrival_times[i] = step * self.time_step  # 到達した時刻を記録
                    print(f"Agent {i+1} has arrived at the destination.")

            # 履歴に現在の位置を保存
            self.history.append(self.positions.copy())

            # 到達したエージェントの割合を計算して表示
            arrived_percentage = np.sum(self.arrived_agents) / self.num_agents * 100
            print(f"Step {step+1}: {arrived_percentage:.2f}% of agents have arrived.")

    def save_to_excel(self, filepath):
        data = []
        # シミュレーション履歴を記録
        for step, positions in enumerate(self.history):
            for agent_idx in range(self.num_agents):
                # 各エージェントの到達フラグもログに含める
                data.append([step * self.time_step, agent_idx + 1, *positions[agent_idx], *self.velocities[agent_idx], self.arrived_agents[agent_idx]])

        df_history = pd.DataFrame(data, columns=["時間", "エージェントID", "位置X", "位置Y", "速度X", "速度Y", "到達フラグ"])
        
        # パラメータ情報を記録
        df_params = pd.DataFrame({
            'エージェントID': np.arange(1, self.num_agents+1),
            '初期位置X': self.positions[:, 0],
            '初期位置Y': self.positions[:, 1],
            '質量': self.masses,
            '反応時間': self.reaction_times,
            '希望速度': self.desired_speeds,
            '目的地X': [self.destination[0]] * self.num_agents,
            '目的地Y': [self.destination[1]] * self.num_agents,
            '到達フラグ': self.arrived_agents  # ここに到達フラグを追加
        })

        with pd.ExcelWriter(filepath) as writer:
            df_history.to_excel(writer, sheet_name='シミュレーション履歴', index=False)
            df_params.to_excel(writer, sheet_name='パラメータ', index=False)


    # エージェントの動きをプロット
    def plot_movement(self, save_path='agent_movements.png'):
        history_np = np.array(self.history)
        plt.figure(figsize=(8, 8))
        for i in range(self.num_agents):
            plt.plot(history_np[:, i, 0], history_np[:, i, 1], label=f"Agent {i + 1}")
        
        # 目的地をプロット
        plt.scatter(self.destination[0], self.destination[1], color='red', marker='x', s=100, label="Destination")
        plt.title("Agent Movements")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        # グラフを保存する
        plt.savefig(save_path)
        #plt.show()

    # 速度変化をプロット (到達した時点にマーカーを付ける)
    def plot_velocity_changes(self):
        time_steps = np.arange(0, self.total_time, self.time_step)  # 横軸：時間
        plt.figure(figsize=(10, 6))
        
        for i in range(self.num_agents):
            plt.plot(time_steps, self.velocity_history[:, i], label=f"Agent {i+1}")
            if not np.isnan(self.arrival_times[i]):
                # 到着した時点にマーカーを付ける
                plt.scatter(self.arrival_times[i], self.velocity_history[int(self.arrival_times[i] / self.time_step), i], 
                            color='red', zorder=5, label=f"Agent {i+1} Arrival")

        plt.title("Velocity Changes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Speed (Velocity)")
        plt.legend()
        plt.grid(True)
        plt.savefig("velocity_changes.png")
        #plt.show()

        # 目的志向力のプロット
        #plt.subplot(3, 1, 2)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_agents):
            plt.plot(time_steps, self.desire_force_history[:, i], label=f"Agent {i+1}")
        plt.title("Desire Force Changes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Desire Force")
        plt.grid(True)
        # 目的志向力のグラフをファイルに保存
        plt.savefig("desire_force_history.png")
        plt.close()  # 現在のプロットを閉じる

        # 反発力のプロット
        #plt.subplot(3, 1, 3)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_agents):
            plt.plot(time_steps, self.repulsion_force_history[:, i], label=f"Agent {i+1}")
        plt.title("Repulsion Force Changes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Repulsion Force")

        #plt.tight_layout()
        plt.savefig("repulsion_force_history.png")
        #plt.show()



# インスタンスを作成し、シミュレーションを実行
#sfm = SocialForceModel(num_agents=30, destination=(10, 10), total_time=20)
sfm = SocialForceModel(num_agents=100, destination=(10, 10), total_time=200)  # 少数のエージェントで確認
sfm.update_positions()

# 動きの履歴をExcelに保存し、動きをプロット
sfm.save_to_excel('SocialForceModel_output_japanese_with_params.xlsx')
sfm.plot_movement()

# 速度変化をプロット
sfm.plot_velocity_changes()
