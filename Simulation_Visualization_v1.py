import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv


class Simulation_Visualization:
    def __init__(self, N, R, locations, times, mean_in_system_data, mean_in_system_class_data, 
                 mean_total_data, mean_total_class_data, theoretical, in_system_customers_data, 
                 total_customers_data, waiting_customers_data, in_service_customers_data, in_transit_customers_data,
                 log_data, process_text, class_data, cumulative_in_system_data, cumulative_total_data):
        self.N = N
        self.R = R
        self.locations = locations
        self.times = times
        self.mean_in_system_data = mean_in_system_data
        self.mean_in_system_class_data = mean_in_system_class_data
        self.mean_total_data = mean_total_data
        self.mean_total_class_data = mean_total_class_data
        self.theoretical = theoretical
        self.in_system_customers_data = in_system_customers_data
        self.total_customers_data = total_customers_data
        self.waiting_customers_data = waiting_customers_data
        self.in_service_customers_data = in_service_customers_data
        self.in_transit_customers_data = in_transit_customers_data
        self.log_data = log_data
        self.process_text = process_text
        self.class_data = class_data
        self.cumulative_in_system_data = cumulative_in_system_data
        self.cumulative_total_data = cumulative_total_data

    def calculate_and_save_rmse(self, filename="./log/rmse_over_time.csv", graph_filename="./log/rmse_graph.png"):
        """
        時刻ごとにRMSEを計算し、CSVに保存し、グラフ化する。
        """
        rmse_data = []  # RMSEデータを保存するリスト

        # データ長の確認
        if len(self.times) == 0:
            print("Error: No time data available.")
            return
        for node in range(self.N):
            for cls in range(self.R):
                if len(self.mean_total_class_data[node][cls]) != len(self.times):
                    print(f"Error: Data length mismatch for Node {node}, Class {cls}.")
                    return

        for t_idx, time in enumerate(self.times):
            # 時刻t_idxにおけるシミュレーション値の計算
            simulated_values = [
                sum(
                    self.mean_total_class_data[node][cls][t_idx]
                    for cls in range(self.R) if t_idx < len(self.mean_total_class_data[node][cls])
                ) 
                for node in range(self.N)
            ]
            # 理論値を取得
            theoretical_values = [self.theoretical.iloc[node].sum() for node in range(self.N)]

            # RMSEの計算
            squared_errors = [(sim - theo) ** 2 for sim, theo in zip(simulated_values, theoretical_values)]
            rmse = math.sqrt(sum(squared_errors) / self.N)
            rmse_data.append([time, rmse])

        # RMSEデータをCSVに保存
        df_rmse = pd.DataFrame(rmse_data, columns=["Time", "RMSE"])
        df_rmse.to_csv(filename, index=False)
        print(f"RMSE data saved to {filename}")

        # RMSEを折れ線グラフでプロット
        times, rmse_values = zip(*rmse_data)
        plt.figure(figsize=(10, 6))
        plt.plot(times, rmse_values, label="RMSE", color="blue", marker="o", markersize=3)
        plt.xlabel("Time")
        plt.ylabel("RMSE")
        plt.title("RMSE over Time")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_filename)
        plt.close()
        print(f"RMSE graph saved to {graph_filename}")

    def calculate_and_save_rmse_per_class(self, filename="./log/rmse_per_class_over_time.csv", graph_filename="./log/rmse_per_class_graph.png"):
        """
        ノード別・クラス別で差分を取り、時刻ごとのRMSEを計算し、CSVに保存し、グラフを描画する。
        RMSEは全ノードと全クラスのデータを考慮して1つの値を計算する。
        """
        rmse_data = []  # 時刻ごとのRMSEを保存するリスト

        if len(self.times) == 0:
            print("Error: No time data available.")
            return

        # データ長の確認
        for node in range(self.N):
            for cls in range(self.R):
                if len(self.mean_total_class_data[node][cls]) != len(self.times):
                    print(f"Error: Data length mismatch for Node {node}, Class {cls}.")
                    return

        # 時刻ごとにRMSEを計算
        for t_idx, time in enumerate(self.times):
            squared_errors = []  # 各時刻の平方誤差を保存
            for node in range(self.N):
                for cls in range(self.R):
                    # シミュレーション値
                    sim_value = self.mean_total_class_data[node][cls][t_idx] if t_idx < len(self.mean_total_class_data[node][cls]) else 0
                    # 理論値
                    theo_value = self.theoretical.iloc[node, cls] if cls < self.theoretical.shape[1] else 0
                    # 平方誤差を計算
                    squared_error = (sim_value - theo_value) ** 2
                    squared_errors.append(squared_error)
            
            # 平均平方誤差を計算しRMSEを算出
            mean_squared_error = sum(squared_errors) / (self.N * self.R)
            rmse = math.sqrt(mean_squared_error)
            rmse_data.append([time, rmse])

        # RMSEデータをCSVに保存
        df_rmse = pd.DataFrame(rmse_data, columns=["Time", "RMSE"])
        df_rmse.to_csv(filename, index=False)
        print(f"RMSE per time data saved to {filename}")

        # グラフのプロット
        times, rmse_values = zip(*rmse_data)
        plt.figure(figsize=(10, 6))
        plt.plot(times, rmse_values, label="RMSE", color="blue", marker="o", markersize=3)  # 点のサイズを小さく設定

        # グラフの設定
        plt.title("RMSE Over Time")
        plt.xlabel("Time")
        plt.ylabel("RMSE")
        plt.legend(loc="upper right", fontsize="small")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(graph_filename)
        plt.close()
        print(f"RMSE graph saved to {graph_filename}")

    def plot_boxplots(self, output_file_in_system = "log/in_system_boxplot.png", output_file_total = "log/total_customers_boxplot.png"):
        """
        系内人数と総系内人数のボックスプロットを作成
        """
        # 系内人数のデータを取得
        in_system_data = [self.in_system_customers_data[node] for node in range(self.N)]
        total_data = [self.total_customers_data[node] for node in range(self.N)]

        # 系内人数のボックスプロット
        plt.figure(figsize=(10, 6))
        plt.boxplot(in_system_data, tick_labels=[f"{i}" for i in range(self.N)])
        plt.title("In-System Customers (Excluding Transit)")
        plt.xlabel("Node")
        plt.ylabel("Number of Customers")
        plt.savefig(output_file_in_system)
        plt.close()
        print(f"Boxplot for In-System Customers saved as '{output_file_in_system}'.")

        # 総系内人数のボックスプロット
        plt.figure(figsize=(10, 6))
        plt.boxplot(total_data, tick_labels=[f"{i}" for i in range(self.N)])
        plt.title("Total Customers (Including Transit)")
        plt.xlabel("Node")
        plt.ylabel("Number of Customers")
        plt.savefig(output_file_total)
        plt.close()
        print(f"Boxplot for Total Customers saved as '{output_file_total}'.")

    def plot_mean_total_customers(self, region_size=1000, filename="./log/mean_total_customers_simulation.png"):
        """
        平均総系内人数 (移動中も含む) を2D等高線で表示
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        contour_cmap = "viridis"

        # グリッドとデータ初期化
        x = np.linspace(0, region_size, 300)
        y = np.linspace(0, region_size, 300)
        xx, yy = np.meshgrid(x, y)
        mean_total_grid = np.zeros_like(xx)

        # 各拠点の影響を平均総系内人数として加算
        for node, (loc_x, loc_y) in enumerate(self.locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            distances = np.maximum(distances, 1)  # 距離が0に近い場合を防ぐ
            total_customers = sum(self.mean_total_class_data[node][cls][-1] for cls in range(self.R))
            influence = total_customers / distances
            mean_total_grid += influence

        # 等高線の描画
        contour = ax.contourf(xx, yy, mean_total_grid, levels=100, cmap=contour_cmap, alpha=0.2)
        ax.contour(xx, yy, mean_total_grid, levels=10, colors="k", linewidths=0.5)
        fig.colorbar(contour, ax=ax, label="Average Total Customers (Including Transit)")

        # 拠点のプロット
        for node, (x, y) in enumerate(self.locations):
            ax.scatter(x, y, c="red", s=50, label=f"Node {node}" if node == 0 else "", zorder=5)
            ax.text(x + 10, y + 10, f"{node}", fontsize=9, color="black")

        # グラフ設定
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Mean Total Customers (Including Transit)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")

    def plot_mean_in_system_customers(self, region_size=1000, filename="./log/mean_in_system_customers_simulation.png"):
        """
        平均系内人数 (移動中を含まない) を2D等高線で表示
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        contour_cmap = "viridis"

        # グリッドとデータ初期化
        x = np.linspace(0, region_size, 300)
        y = np.linspace(0, region_size, 300)
        xx, yy = np.meshgrid(x, y)
        mean_in_system_grid = np.zeros_like(xx)

        # 各拠点の影響を平均系内人数として加算
        for node, (loc_x, loc_y) in enumerate(self.locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            distances = np.maximum(distances, 1)  # 距離が0に近い場合を防ぐ
            in_system_customers = sum(self.mean_in_system_class_data[node][cls][-1] for cls in range(self.R))
            influence = in_system_customers / distances
            mean_in_system_grid += influence

        # 等高線の描画
        contour = ax.contourf(xx, yy, mean_in_system_grid, levels=100, cmap=contour_cmap, alpha=0.2)
        ax.contour(xx, yy, mean_in_system_grid, levels=10, colors="gray", linewidths=0.5)
        fig.colorbar(contour, ax=ax, label="Average In-System Customers (Excluding Transit)")

        # 拠点のプロット
        for node, (x, y) in enumerate(self.locations):
            ax.scatter(x, y, c="blue", s=50, label=f"Node {node}" if node == 0 else "", zorder=5)
            ax.text(x + 10, y + 10, f"{node}", fontsize=9, color="black")

        # グラフ設定
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Mean In-System Customers (Excluding Transit)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved: {filename}")

    def plot_total_customers_with_movement(self):
        """
        平均総系内人数（移動中を含む）をクラス別に色分けし、
        系内人数と移動中人数で積み上げ棒グラフを作成。
        """
        nodes = np.arange(self.N)  # 拠点番号
        in_system_class_data = np.array([
            [self.mean_in_system_class_data[node][cls][-1] for cls in range(self.R)]
            for node in range(self.N)
        ]).T
        in_transit_class_data = np.array([
            [self.mean_total_class_data[node][cls][-1] - self.mean_in_system_class_data[node][cls][-1] for cls in range(self.R)]
            for node in range(self.N)
        ]).T

        # 系内人数と移動中人数の積み上げ棒グラフを描画
        plt.figure(figsize=(12, 6))
        bottoms = np.zeros(self.N)
        for cls in range(self.R):
            plt.bar(nodes, in_system_class_data[cls], bottom=bottoms, label=f"Class {cls} In System")
            bottoms += in_system_class_data[cls]
            plt.bar(nodes, in_transit_class_data[cls], bottom=bottoms, label=f"Class {cls} In Transit", alpha=0.6)
            bottoms += in_transit_class_data[cls]

        # グラフの設定
        plt.xlabel("Node")
        plt.ylabel("Average Total Customers")
        plt.title("Class-Based Average Total Customers per Node (With Movement)")
        plt.xticks(nodes, [f"{node}" for node in range(self.N)])
        plt.legend(title="Customer Class & State", loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # 画像保存
        output_file = "log/class_based_total_customers_with_movement.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Class-based stack bar graph (with movement) saved: {output_file}")

    def plot_in_system_customers(self):
        """
        平均系内人数（移動中を含まない）をクラス別に色分けした積み上げ棒グラフを作成。
        """
        nodes = np.arange(self.N)  # 拠点番号
        in_system_class_data = np.array([
            [self.mean_in_system_class_data[node][cls][-1] for cls in range(self.R)]
            for node in range(self.N)
        ]).T

        # 系内人数の積み上げ棒グラフを描画
        plt.figure(figsize=(12, 6))
        bottoms = np.zeros(self.N)
        for cls in range(self.R):
            plt.bar(nodes, in_system_class_data[cls], bottom=bottoms, label=f"Class {cls}")
            bottoms += in_system_class_data[cls]

        # グラフの設定
        plt.xlabel("Node")
        plt.ylabel("Average In-System Customers")
        plt.title("Class-Based Average In-System Customers per Node")
        plt.xticks(nodes, [f"{node}" for node in range(self.N)])
        plt.legend(title="Customer Class", loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # 画像保存
        output_file = "log/class_based_in_system_customers.png"
        plt.savefig(output_file)
        plt.close()
        print(f"Class-based in-system customers stack bar graph saved: {output_file}")

    def plot_mean_data(self, key, title, ylabel, output_file):
        """
        平均系内人数や総人数をグラフ化。
        """
        data = getattr(self, f"{key}_data")
        plt.figure(figsize=(12, 6))
        for node, values in data.items():
            # データの整合性チェック
            if len(values) != len(self.times):
                print(f"[WARNING] Length mismatch for node {node}: times={len(self.times)}, values={len(values)}")
                # 長さを揃える（短い方に合わせる）
                min_len = min(len(self.times), len(values))
                times = self.times[:min_len]
                values = values[:min_len]
            else:
                times = self.times

            plt.plot(times, values, label=f"Node {node}")

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"Graph saved: {output_file}")

    def plot_all_graphs(self):
        """
        保存されたデータからグラフを作成。
        """
        #print(f"[DEBUG] Times length: {len(self.times)}")
        #for node, values in self.waiting_customers_data.items():
        #    print(f"[DEBUG] Node {node} values length: {len(values)}")

        self.plot_realtime_customer_data(
            self.times, 
            self.waiting_customers_data, 
            "Waiting Customers", 
            "Number of Customers", 
            "log/waiting_customers_plot.png"
        )

        self.plot_realtime_customer_data(
            self.times, 
            self.in_service_customers_data, 
            "In-Service Customers", 
            "Number of Customers", 
            "log/in_service_customers_plot.png"
        )

        self.plot_realtime_customer_data(
            self.times, 
            self.in_system_customers_data, 
            "In-System Customers", 
            "Number of Customers", 
            "log/in_system_customers_plot.png"
        )

        self.plot_realtime_customer_data(
            self.times, 
            self.in_transit_customers_data, 
            "In-Transit Customers", 
            "Number of Customers", 
            "log/in_transit_customers_plot.png"
        )

        self.plot_realtime_customer_data(
            self.times, 
            self.total_customers_data, 
            "Total Customers", 
            "Number of Customers", 
            "log/total_customers_plot.png"
        )

    def plot_realtime_customer_data(self, times, stats, title, ylabel, output_file):
        """
        保存されたデータを使用してリアルタイムでグラフを描画。
        :param times: 時刻データのリスト
        :param stats: 各ノードのデータを含む辞書
        :param title: グラフのタイトル
        :param ylabel: Y軸のラベル
        :param output_file: 保存する画像ファイル名
        """
        plt.figure(figsize=(12, 6))

        # 各ノードのデータをプロット
        for node, values in stats.items():
            plt.plot(times, values, label=f"Node {node}")

        # グラフの設定
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.grid(True)

        # 画像を保存
        plt.savefig(output_file)
        plt.close()
        print(f"Graph saved: {output_file}")

    def save_log_to_csv(self):
        """
        ログデータをCSVファイルに保存する
        """

        output_file = self.process_text
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["time", "event_type", "customer_id", "customer_class", "node", "additional_info"])
            writer.writeheader()  # ヘッダーを書き込む
            writer.writerows(self.log_data)  # ログデータを書き込む

        print(f"Log saved to {output_file}")

    def save_class_data_to_csv(self):
        """
        クラス別データをCSVに保存。
        """
        for data_type, node_data in self.class_data.items():
            file_path = f"log/{data_type}_class.csv"
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)

                # ヘッダー作成
                header = ["time"] + [
                    f"node{node}(class{cls})"
                    for node in range(self.N) for cls in range(self.R)
                ]
                writer.writerow(header)

                # 時系列データを書き込む
                for t_idx, time in enumerate(self.times):
                    row = [time]
                    for node in range(self.N):
                        for cls in range(self.R):
                            row.append(node_data[node][cls][t_idx])
                    writer.writerow(row)
        print(f"Log saved to {file_path}")

    def save_extended_data_to_csv(self):
        """
        延べ人数や平均人数をCSVに保存。
        """
        # 保存ファイルの定義
        file_paths = {
            "cumulative_in_system": "log/cumulative_in_system.csv",
            "cumulative_total": "log/cumulative_total.csv",
            "mean_in_system": "log/mean_in_system.csv",
            "mean_total": "log/mean_total.csv"
        }

        # 延べ系内人数と平均人数の保存
        for key, path in file_paths.items():
            with open(path, mode="w", newline="") as file:
                writer = csv.writer(file)
                header = ["time"] + [f"node_{node}" for node in range(self.N)]
                writer.writerow(header)

                data = getattr(self, f"{key}_data")
                for t_idx, time in enumerate(self.times):
                    # デバッグ用ログ
                    if any(t_idx >= len(data[node]) for node in range(self.N)):
                        print(f"[DEBUG] Mismatch at t_idx={t_idx}, time={time}")
                        for node in range(self.N):
                            print(f"  Node {node}: data_length={len(data[node])}, t_idx={t_idx}")

                    # 時系列データの保存
                    row = [time] + [data[node][t_idx] if t_idx < len(data[node]) else None for node in range(self.N)]
                    writer.writerow(row)
        print(f"Log saved to {file_paths}")
