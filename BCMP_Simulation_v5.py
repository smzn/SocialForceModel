import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import sys
import time
import matplotlib.colors as mcolors
import datetime  # 現在時刻を取得するためのモジュール
import os
import csv

class BCMP_Simulation:
    
    def __init__(self, N, R, K_total, mu, m, type_list, p, locations, speed, theoretical, sim_time):
        self.N = N  # 網内の拠点数
        self.R = R  # クラス数
        self.K_total = K_total  # 客の総数
        self.mu = mu / 1000  # サービス率
        self.type_list = type_list  # サービスの種類
        self.p = p
        self.locations = locations
        self.speed = speed  # 移動速度 (m/s)
        self.distances, self.travel_times = self.calculate_distances_and_times()  # 距離と移動時間を計算
        self.theoretical = theoretical
        self.m = m  # 窓口配列
        self.time = sim_time  # シミュレーション時間
        self.log_data = []  # ログデータを保持するリスト
        self.initialize_logs()

        self.service = [[] for _ in range(self.N)]  # サービス中の顧客リストを初期化 # 各拠点のサービス中の顧客リスト
        self.queue = [[] for _ in range(self.N)]  # 待ち行列の初期化 # 各拠点の待ち行列

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M") # 現在時刻をフォーマットして取得 (分まで)
        self.process_text = (f'./process/process_N{self.N}_R{self.R}_K{self.K_total}_Time{self.time}_Speed{self.speed:.1f}_TimeStamp{current_time}.csv')
        self.K = [(self.K_total + i) // self.R for i in range(self.R)] # クラスごとの人数を等分に計算
        self.customers = self.initialize_customers() # 客ごとのデータを初期化
        # 最初の10件の情報を表示
        print("Customer Details:")
        for customer in self.customers:
            print(customer)
        self.save_distances_and_times() # 距離と移動時間をcsvで保存

        # 時刻ごとのデータ保存用
        self.times = []
        self.waiting_customers_data = {node: [] for node in range(self.N)}
        self.in_service_customers_data = {node: [] for node in range(self.N)}
        self.in_system_customers_data = {node: [] for node in range(self.N)}
        self.in_transit_customers_data = {node: [] for node in range(self.N)}
        self.total_customers_data = {node: [] for node in range(self.N)}

        self.class_data = {
            data_type: {node: {cls: [] for cls in range(self.R)} for node in range(self.N)}
            for data_type in ["waiting_customers", "in_service_customers", "in_system_customers", "in_transit_customers", "total_customers"]
        }
        #延べ人数、平均人数、総系内人数などのデータ構造を初期化。
        self.cumulative_in_system_data = {node: [] for node in range(self.N)}
        self.cumulative_in_system_class_data = {node: {cls: [] for cls in range(self.R)} for node in range(self.N)}
        self.cumulative_total_data = {node: [] for node in range(self.N)}
        self.cumulative_total_class_data = {node: {cls: [] for cls in range(self.R)} for node in range(self.N)}

        self.mean_in_system_data = {node: [] for node in range(self.N)}
        self.mean_in_system_class_data = {node: {cls: [] for cls in range(self.R)} for node in range(self.N)}
        self.mean_total_data = {node: [] for node in range(self.N)}
        self.mean_total_class_data = {node: {cls: [] for cls in range(self.R)} for node in range(self.N)}

        # 計算用データ構造を初期化
        self.event = [[] for _ in range(self.N)]  # 各拠点のイベント
        self.eventclass = [[] for _ in range(self.N)]  # イベント発生時の客クラス番号
        self.eventqueue = [[] for _ in range(self.N)]  # イベント発生時の待ち行列長
        self.eventtime = [[] for _ in range(self.N)]  # イベント発生時の時刻
        self.timerate = np.zeros((self.N, self.K_total + 1))  # 拠点での人数分布
        self.timerateclass = np.zeros((self.N, self.R, self.K_total + 1))  # クラス別人数分布
        
        # 平均算出用
        self.sum_L = np.zeros(self.N)  # 平均系内人数
        self.sum_Lc = np.zeros((self.N, self.R))  # クラス別平均系内人数
        self.sum_Q = np.zeros(self.N)  # 平均待ち人数
        self.sum_Qc = np.zeros((self.N, self.R))  # クラス別平均待ち人数
        
        self.start = time.time()

    def run_simulation(self):
        """
        イベントドリブン型シミュレーションのメインループ
        """

        # 初期設定
        self.initialize_simulation()

        current_time = 0
        self.cumulative_customers = 0  # 延べ人数の初期化

        while current_time < self.time:
            next_event_time, next_event_details = self.find_next_event(current_time)
            #print('next_event_time, next_event_details = {0}, {1}'.format(next_event_time, next_event_details))
            
            if next_event_details is None:
                break  # イベントがない場合は終了

            # イベント処理の前にログを記録
            self.log_node_data(current_time)

            self.times.append(current_time)
            # 時刻ごとのデータを更新
            self.update_customer_data(current_time)
            self.update_class_data(current_time)

            # イベント間隔を計算
            delta_time = next_event_time - current_time

            #if delta_time <= 0:
            #    print(f"[ERROR] Negative delta_time detected: {delta_time}, current_time: {current_time}, next_event_time: {next_event_time}")
            #    break

            # 待ち行列とサービス中の顧客数
            queue_and_service_customers = sum(
                len(self.queue[node]) + len(self.service[node]) for node in range(self.N)
            )

            # 移動中の顧客数
            in_transit_customers = sum(
                1 for customer in self.customers if customer["state"] == "in_transit"
            )

            # 延べ人数を計算
            self.update_extended_data(current_time, delta_time)
            #self.cumulative_customers += (queue_and_service_customers + in_transit_customers) * delta_time

            # システム時間を次イベントの時刻に進める
            current_time = next_event_time
            #print(f"Time {current_time}: Processing event {next_event_details['event_type']} for customer {next_event_details['customer_id']} at node {next_event_details['location']}")

            
            print(f"[Time: {current_time:.2f}] Event: {next_event_details['event_type']} | "
                f"Customer ID: {next_event_details['customer_id']} | "
                f"Node: {next_event_details['location']} | "
                f"Queue Size: {len(self.queue[next_event_details['location']])}")
            
                
            # イベント処理
            if next_event_details["event_type"] == "service_complete":
                self.process_service_complete(next_event_details, current_time, delta_time)
            elif next_event_details["event_type"] == "travel_complete":
                self.process_travel_complete(next_event_details, current_time, delta_time)

        print(f"Simulation completed. Total cumulative customers: {self.cumulative_customers}")
        # シミュレーション終了時にログを保存
        self.save_log_to_csv()
        self.save_class_data_to_csv()
        self.save_extended_data_to_csv()
        print(f"Simulation completed. Total cumulative customers: {self.cumulative_customers}")

        # シミュレーション終了後にグラフを作成
        self.plot_all_graphs()
        self.plot_mean_data("mean_in_system", "Mean In-System Customers", "Number of Customers", "log/mean_in_system.png")
        self.plot_mean_data("mean_total", "Mean Total Customers", "Number of Customers", "log/mean_total.png")

        self.plot_total_customers_with_movement()  # 移動中を含む積み上げグラフ
        self.plot_in_system_customers()           # 系内人数のみの積み上げグラフ

        self.plot_mean_total_customers()  # 平均総系内人数
        self.plot_mean_in_system_customers()  # 平均系内人数

        self.calculate_and_save_rmse()  # RMSEの計算、保存、グラフ作成
        self.calculate_and_save_rmse_per_class()

    def process_service_complete(self, event_details, current_time, delta_time):
        """
        サービス終了イベントの処理
        :param event_details: イベントの詳細情報
        """
        customer_id = event_details["customer_id"]
        location = event_details["location"]

        # ログを記録
        self.log_event(
            event_time=current_time,
            event_type="service_complete",
            customer_id=customer_id,
            customer_class=event_details["customer_class"],
            node=location,
            additional_info={"remaining_customers": len(self.queue[location])}
        )

        # サービス中の顧客をリストから削除
        customer = None
        for c in self.service[location]:
            if c["id"] == customer_id:
                customer = c
                self.service[location].remove(c)
                break

        if customer is None:
            print(f"Error: Customer {customer_id} not found in service at node {location}")
            return

        # 次の拠点を決定し、移動時間を設定
        next_location = self.get_next_location(location, customer["class"])
        customer["state"] = "in_transit"
        customer["location"] = next_location
        customer["travel_time_remaining"] = self.travel_times[location][next_location]
        customer["history"].append({
            "event": "service_complete",
            "node": location,
            "time": customer["travel_time_remaining"]
        })

        # 待ち行列から次の顧客を取り出してサービス中に
        if len(self.queue[location]) > 0:
            next_customer = self.queue[location].pop(0)
            next_customer["state"] = "in_service"
            next_customer["service_time_remaining"] = self.getExponential(self.mu[location])
            next_customer["history"].append({
                "event": "service_start",
                "node": location,
                "time": 0
            })
            self.service[location].append(next_customer)

        # 他の顧客の状態を更新
        self.update_customer_times(delta_time)

    def get_next_location(self, current_location, customer_class):
        """
        遷移確率行列に基づいて次の拠点を決定する
        :param current_location: 現在の拠点
        :param customer_class: 顧客のクラス
        :return: 次の拠点
        """
        # 顧客クラスに対応する推移確率行列の部分を抽出
        start_index = self.N * customer_class
        end_index = self.N * (customer_class + 1)

        # 推移確率行列から該当するサブマトリックスを抽出
        pr = self.p.iloc[start_index:end_index, start_index:end_index].values

        # 現在のノードに対応する確率ベクトルを取得
        probabilities = pr[current_location - start_index]

        # 確率を正規化（合計が1でない場合を考慮）
        probabilities /= probabilities.sum()

        # 次のノードを確率に基づいて選択
        return np.random.choice(range(self.N), p=probabilities)

    def update_customer_times(self, delta_time):
        """
        すべての顧客のサービス時間および移動時間を進める
        :param delta_time: 前回のイベントから現在のイベントまでの時間間隔
        """
        for customer in self.customers:
            if customer["state"] == "in_service":
                customer["service_time_remaining"] = max(0, customer["service_time_remaining"] - delta_time)
            elif customer["state"] == "in_transit":
                customer["travel_time_remaining"] = max(0, customer["travel_time_remaining"] - delta_time)

    def find_next_event(self, current_time):
        """
        次に発生するイベントを特定する。
        :return: (次イベントの時刻, 客ID, 客クラス, イベント種別, 拠点)
        """
        print(f"[DEBUG] find_next_event called at time {current_time:.2f}")

        next_event_time = float('inf')  # 最小時刻を無限大で初期化
        next_event_details = None

        # サービス中の顧客のイベントを確認
        for node in range(self.N):
            #print(f"  Node {node}: {len(self.service[node])} customers in service")
            for customer in self.service[node]:
                if customer["state"] != "in_service":
                    continue  # サービス中でない顧客をスキップ
                if customer["service_time_remaining"] + current_time < next_event_time:
                    next_event_time = current_time + customer["service_time_remaining"] # サービス時間を加算して次のイベント時刻を計算
                    next_event_details = {
                        "event_type": "service_complete",
                        "customer_id": customer["id"],
                        "customer_class": customer["class"],
                        "location": node
                    }
                    print(f"      [DEBUG] New next service event: time={next_event_time}, customer_id={customer['id']}")

        # 移動中の顧客のイベントを確認
        for customer in self.customers:
            if customer["state"] != "in_transit":
                continue  # 移動中でない顧客をスキップ
            if customer["travel_time_remaining"] + current_time < next_event_time:
                next_event_time = current_time + customer["travel_time_remaining"]
                next_event_details = {
                    "event_type": "travel_complete",
                    "customer_id": customer["id"],
                    "customer_class": customer["class"],
                    "location": customer["location"]
                }
                print(f"      [DEBUG] New next transit event: time={next_event_time}, customer_id={customer['id']}")

        print(f"[DEBUG] Next event: time={next_event_time}, details={next_event_details}")
        return next_event_time, next_event_details

    def process_travel_complete(self, event_details, current_time, delta_time):
        """
        移動完了イベントの処理
        :param event_details: イベントの詳細情報
        :param delta_time: 前回のイベントから現在のイベントまでの時間間隔
        """
        customer_id = event_details["customer_id"]
        location = event_details["location"]

        # ログを記録
        self.log_event(
            event_time=current_time,
            event_type="travel_complete",
            customer_id=customer_id,
            customer_class=event_details["customer_class"],
            node=location,
            additional_info=None
        )

        # 移動中の顧客を特定
        customer = None
        for c in self.customers:
            if c["id"] == customer_id and c["state"] == "in_transit":
                customer = c
                break

        if customer is None:
            print(f"Error: Customer {customer_id} not found in transit state.")
            return

        # 移動完了処理
        customer["state"] = "waiting"
        customer["location"] = location
        customer["travel_time_remaining"] = 0  # 移動完了なので残り時間は0
        customer["history"].append({
            "event": "travel_complete",
            "node": location,
            "time": delta_time
        })

        # 顧客を待ち行列に追加
        self.queue[location].append(customer)

        # 窓口が空いている場合はサービスを開始
        if len(self.service[location]) < self.m[location]:  # 窓口数よりサービス中の顧客数が少ない場合
            next_customer = self.queue[location].pop(0)  # 待ち行列から取り出す
            next_customer["state"] = "in_service"
            next_customer["service_time_remaining"] = self.getExponential(self.mu[location])
            next_customer["history"].append({
                "event": "service_start",
                "node": location,
                "time": delta_time
            })
            self.service[location].append(next_customer)

        # 他の顧客の状態を更新
        self.update_customer_times(delta_time)


    def initialize_customers(self):
        """
        各客の状態と属性を初期化する
        """
        # クラスごとの割り当て
        class_assignments = []
        for class_id, num_customers in enumerate(self.K):
            class_assignments.extend([class_id] * num_customers)

        # クラスをランダムにシャッフル
        random.shuffle(class_assignments)

        # 客ごとのデータを生成
        customers = [
            {
                "id": i,
                "class": class_assignments[i],  # 割り当てられたクラス
                "state": "waiting",             # 状態: waiting, in_transit, in_service
                "location": None,               # 現在の拠点 (初期値: None)
                "travel_time_remaining": 0,     # 残りの移動時間
                "service_time_remaining": 0     # 残りのサービス時間
            }
            for i in range(len(class_assignments))
        ]
        return customers
    
    def calculate_distances_and_times(self):
        """
        拠点間の距離と移動時間を計算
        :return: 距離行列 (m 単位) と移動時間行列 (秒単位)
        """
        num_locations = len(self.locations)
        distances = np.zeros((num_locations, num_locations))  # 距離行列
        travel_times = np.zeros((num_locations, num_locations))  # 移動時間行列

        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:  # 同じ拠点間の距離と移動時間は 0
                    # 距離を計算 (ユークリッド距離)
                    distances[i][j] = np.sqrt(
                        (self.locations[i][0] - self.locations[j][0])**2 +
                        (self.locations[i][1] - self.locations[j][1])**2
                    )
                    # 移動時間を計算
                    travel_times[i][j] = distances[i][j] / self.speed  # 時間 = 距離 / 速度

        return distances, travel_times

    def save_distances_and_times(self, distances_file="./csv/distances.csv", travel_times_file_prefix="./csv/travel_times"):
        """
        拠点間の距離と移動時間をCSVファイルに保存
        :param distances_file: 距離行列を保存するファイル名
        :param travel_times_file_prefix: 移動時間行列ファイルのプレフィックス
        """
        # pandasのDataFrameに変換
        distances_df = pd.DataFrame(self.distances)
        travel_times_df = pd.DataFrame(self.travel_times)

        # 移動時間ファイル名に速度を含める
        travel_times_file = f"{travel_times_file_prefix}_speed_{self.speed:.1f}mps.csv"

        # CSVファイルとして保存
        distances_df.to_csv(distances_file, index=False, header=False)
        travel_times_df.to_csv(travel_times_file, index=False, header=False)

        print(f"Distances saved to {distances_file}")
        print(f"Travel times saved to {travel_times_file}")

    def getExponential(self, param):
        return - math.log(1 - random.random()) / param

    def initialize_simulation(self):
        """
        シミュレーションの初期設定を行う。
        各顧客をランダムに拠点に割り当て、窓口数に応じてサービスを開始する顧客を設定する。
        """
        # 各拠点での初期状態を準備
        queue = [[] for _ in range(self.N)]  # 各拠点の待ち行列
        service = [[] for _ in range(self.N)]  # 各拠点のサービス中の客リスト

        # 各顧客の移動履歴を初期化
        for customer in self.customers:
            customer["history"] = []  # 移動履歴を記録するリスト

        # 各顧客をランダムに拠点へ割り当て
        for customer in self.customers:
            assigned_location = random.randint(0, self.N - 1)  # ランダムに拠点を選択
            customer["location"] = assigned_location  # 拠点を設定
            customer["state"] = "waiting"  # 初期状態を待ち行列中に設定
            customer["history"].append({"event": "assigned", "node": assigned_location, "time": 0})  # 履歴に追加
            queue[assigned_location].append(customer)  # 待ち行列に追加

        # 窓口数分の客をサービスに入れる
        for i in range(self.N):
            for _ in range(min(self.m[i], len(queue[i]))):  # 窓口数か待ち行列数の小さい方だけ処理
                customer = queue[i].pop(0)  # 待ち行列から取り出す
                customer["state"] = "in_service"  # 状態をサービス中に変更
                customer["service_time_remaining"] = self.getExponential(self.mu[i])  # サービス時間を設定
                customer["history"].append({"event": "service_start", "node": i, "time": 0})  # 履歴に追加
                service[i].append(customer)  # サービス中リストに追加

        # 結果を保存
        self.queue = queue  # 待ち行列情報を保存
        self.service = service  # サービス中情報を保存

        print("Initial service state:")
        for node in range(self.N):
            print(f"Node {node}: {len(self.service[node])} customers in service")


        # 初期イベントの記録
        for node in range(self.N):
            self.event[node].append("initial")
            self.eventtime[node].append(0)  # 初期時刻を0として記録
            self.eventqueue[node].append(len(self.queue[node]))

        print("Simulation initialized.")
        self.display_initial_state()


    def display_initial_state(self, output_file = "./plot/initial_state_visualization.png", output_file_class = "./plot/initial_state_visualization_by_class.png"):
        """
        初期状態を表示する。
        """
        print("\n--- Initial State ---")
        for i in range(self.N):
            # 系内人数、サービス中、待ち行列中の人数を取得
            num_in_service = len(self.service[i])
            num_waiting = len(self.queue[i])
            total_customers = num_in_service + num_waiting
            print(f"Node {i}: Total = {total_customers}, In Service = {num_in_service}, Waiting = {num_waiting}")

        print("\n--- Customers ---")
        for customer in self.customers:
            print(f"ID: {customer['id']}, Class: {customer['class']}, State: {customer['state']}, Location: {customer['location']}, History: {customer['history']}")

        print("\n--- Events ---")
        for i in range(self.N):
            print(f"Node {i} Events: {self.event[i]}")
            print(f"Node {i} Event Times: {self.eventtime[i]}")
            print(f"Node {i} Queue Lengths: {self.eventqueue[i]}")

        # 初期状態の可視化用データ作成
        in_service_counts = [len(self.service[i]) for i in range(self.N)]
        waiting_counts = [len(self.queue[i]) for i in range(self.N)]
        in_transit_counts = [0] * self.N  # 初期状態では移動中の顧客はいない

        # 可視化
        x = range(self.N)
        plt.figure(figsize=(10, 6))
        plt.bar(x, in_service_counts, label="In Service", bottom=[0]*self.N)
        plt.bar(x, waiting_counts, label="Waiting", bottom=in_service_counts)
        plt.bar(x, in_transit_counts, label="In Transit", bottom=[i + j for i, j in zip(in_service_counts, waiting_counts)])

        # グラフの設定
        plt.xlabel("Node")
        plt.ylabel("Number of Customers")
        plt.title("Initial State of the Simulation")
        plt.xticks(ticks=x, labels=[f"{i}" for i in x])
        plt.legend()

        # ファイルに保存
        plt.savefig(output_file)
        plt.close()

        print(f"Initial state visualization saved to {output_file}")

        # クラス別のデータ作成
        class_counts_in_service = {class_id: [0] * self.N for class_id in range(self.R)}
        class_counts_waiting = {class_id: [0] * self.N for class_id in range(self.R)}
        class_counts_in_transit = {class_id: [0] * self.N for class_id in range(self.R)}

        for customer in self.customers:
            class_id = customer['class']
            location = customer['location']
            if customer['state'] == 'in_service':
                class_counts_in_service[class_id][location] += 1
            elif customer['state'] == 'waiting':
                class_counts_waiting[class_id][location] += 1
            elif customer['state'] == 'in_transit':
                class_counts_in_transit[class_id][location] += 1

        # クラス別の積み上げグラフ
        plt.figure(figsize=(10, 6))
        bottoms = [0] * self.N
        for class_id in range(self.R):
            in_service = class_counts_in_service[class_id]
            waiting = class_counts_waiting[class_id]
            in_transit = class_counts_in_transit[class_id]

            plt.bar(x, in_service, label=f"Class {class_id} In Service", bottom=bottoms)
            bottoms = [b + s for b, s in zip(bottoms, in_service)]
            plt.bar(x, waiting, label=f"Class {class_id} Waiting", bottom=bottoms)
            bottoms = [b + w for b, w in zip(bottoms, waiting)]
            plt.bar(x, in_transit, label=f"Class {class_id} In Transit", bottom=bottoms)
            bottoms = [b + t for b, t in zip(bottoms, in_transit)]

        # グラフの設定
        plt.xlabel("Node")
        plt.ylabel("Number of Customers")
        plt.title("Initial State by Class and State")
        plt.xticks(ticks=x, labels=[f"{i}" for i in x])
        plt.legend()

        # ファイルに保存
        plt.savefig(output_file_class)
        plt.close()

        print(f"Initial state visualization by class saved to {output_file_class}")

    def log_event(self, event_time, event_type, customer_id, customer_class, node, additional_info=None):
        """
        イベントログを記録する
        :param event_time: イベント時刻
        :param event_type: イベントの種類 (e.g., service_complete, travel_complete)
        :param customer_id: 顧客ID
        :param customer_class: 顧客クラス
        :param node: イベントが発生したノード
        :param additional_info: その他の情報（任意）
        """
        if event_time == 0:
            print(f"[DEBUG] Event time is 0: event_type={event_type}, customer_id={customer_id}, node={node}")
        log_entry = {
            "time": event_time,
            "event_type": event_type,
            "customer_id": customer_id,
            "customer_class": customer_class,
            "node": node,
            "additional_info": additional_info
        }
        self.log_data.append(log_entry)

    def save_log_to_csv(self):
        """
        ログデータをCSVファイルに保存する
        """
        import csv

        output_file = self.process_text
        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["time", "event_type", "customer_id", "customer_class", "node", "additional_info"])
            writer.writeheader()  # ヘッダーを書き込む
            writer.writerows(self.log_data)  # ログデータを書き込む

        print(f"Log saved to {output_file}")


    def initialize_logs(self):
        """
        ログフォルダとファイルを初期化する。
        """
        if not os.path.exists("log"):
            os.makedirs("log")

        # データタイプ
        data_types = ["waiting_customers", "in_service_customers", "in_system_customers", "in_transit_customers", "total_customers"]

        # 各データタイプごとのファイルを準備
        for data_type in data_types:
            file_path = f"log/{data_type}.csv"
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                # ヘッダー行を作成
                header = ["time"] + [f"node_{i}" for i in range(self.N)]
                writer.writerow(header)

    def log_node_data(self, current_time):
        """
        ノードごとのデータをログに記録する。
        :param current_time: 現在の時刻
        """
        # データタイプ
        data_types = ["waiting_customers", "in_service_customers", "in_system_customers", "in_transit_customers", "total_customers"]

        # 各ノードの統計情報を取得
        node_stats = self.get_all_node_statistics()

        # 各データタイプに対応するデータを記録
        for data_type in data_types:
            file_path = f"log/{data_type}.csv"
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                # 現在時刻と各ノードのデータを記録
                row = [current_time] + [node_stats[node][data_type] for node in range(self.N)]
                writer.writerow(row)

    def get_all_node_statistics(self):
        """
        すべてのノードの統計情報を収集する。
        :return: ノード統計情報の辞書
        """
        node_stats = {}
        for node in range(self.N):
            node_stats[node] = self.get_node_statistics(node)
        return node_stats

    def get_node_statistics(self, node):
        """
        ノードの統計情報を取得する。
        :param node: ノード番号
        :return: 統計情報の辞書
        """
        # 待ち人数
        waiting_customers = len(self.queue[node])

        # サービス中の人数
        in_service_customers = len(self.service[node])

        # 系内人数 (待ち人数 + サービス中人数)
        in_system_customers = waiting_customers + in_service_customers

        # このノードへ移動中の人数
        in_transit_customers = sum(1 for c in self.customers if c["state"] == "in_transit" and c["location"] == node)

        # 総人数 (系内人数 + 移動中人数)
        total_customers = in_system_customers + in_transit_customers

        return {
            "waiting_customers": waiting_customers,
            "in_service_customers": in_service_customers,
            "in_system_customers": in_system_customers,
            "in_transit_customers": in_transit_customers,
            "total_customers": total_customers
        }

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

    def update_customer_data(self, current_time):
        """
        時刻ごとに顧客データを更新。
        :param current_time: 現在の時刻
        """
        #self.times.append(current_time)

        # ノードの統計情報を取得
        node_stats = self.get_all_node_statistics()

        # 各ノードのデータを保存
        for node in range(self.N):
            stats = node_stats[node]
            self.waiting_customers_data[node].append(stats["waiting_customers"])
            self.in_service_customers_data[node].append(stats["in_service_customers"])
            self.in_system_customers_data[node].append(stats["in_system_customers"])
            self.in_transit_customers_data[node].append(stats["in_transit_customers"])
            self.total_customers_data[node].append(stats["total_customers"])

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

    def update_class_data(self, current_time):
        """
        時刻ごとにクラス別のデータを更新。
        :param current_time: 現在の時刻
        """
        for node in range(self.N):
            for cls in range(self.R):
                # 待ち人数（クラス別）
                waiting_customers = sum(1 for c in self.queue[node] if c["class"] == cls)
                # サービス中の人数（クラス別）
                in_service_customers = sum(1 for c in self.service[node] if c["class"] == cls)
                # 系内人数（クラス別）
                in_system_customers = waiting_customers + in_service_customers
                # このノードへ移動中の人数（クラス別）
                in_transit_customers = sum(1 for c in self.customers if c["state"] == "in_transit" and c["location"] == node and c["class"] == cls)
                # 総人数（クラス別）
                total_customers = in_system_customers + in_transit_customers

                # データを保存
                self.class_data["waiting_customers"][node][cls].append(waiting_customers)
                self.class_data["in_service_customers"][node][cls].append(in_service_customers)
                self.class_data["in_system_customers"][node][cls].append(in_system_customers)
                self.class_data["in_transit_customers"][node][cls].append(in_transit_customers)
                self.class_data["total_customers"][node][cls].append(total_customers)

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

    def update_extended_data(self, current_time, delta_time):
        """
        延べ人数と平均人数を更新。
        :param current_time: 現在の時刻
        :param delta_time: 前回イベントからの時間間隔
        """
        
        for node in range(self.N):
            # 系内人数（移動中を含まない）
            in_system = len(self.queue[node]) + len(self.service[node])
            self.cumulative_in_system_data[node].append(in_system * delta_time)

            # 総人数（移動中を含む）
            in_transit = sum(1 for c in self.customers if c["state"] == "in_transit" and c["location"] == node)
            total = in_system + in_transit
            self.cumulative_total_data[node].append(total * delta_time)

            # 平均値
            if current_time > 0:
                self.mean_in_system_data[node].append(
                    sum(self.cumulative_in_system_data[node]) / current_time
                )
                self.mean_total_data[node].append(
                    sum(self.cumulative_total_data[node]) / current_time
                )
            else:
                # current_time == 0 の場合、直前の値を追加する
                self.mean_in_system_data[node].append(
                    self.mean_in_system_data[node][-1] if self.mean_in_system_data[node] else 0
                )
                self.mean_total_data[node].append(
                    self.mean_total_data[node][-1] if self.mean_total_data[node] else 0
                )


            # クラス別計算
            for cls in range(self.R):
                in_system_class = sum(1 for c in self.queue[node] if c["class"] == cls) + \
                                sum(1 for c in self.service[node] if c["class"] == cls)
                in_transit_class = sum(1 for c in self.customers if c["state"] == "in_transit" and c["location"] == node and c["class"] == cls)

                # 延べ人数
                self.cumulative_in_system_class_data[node][cls].append(in_system_class * delta_time)
                self.cumulative_total_class_data[node][cls].append((in_system_class + in_transit_class) * delta_time)

                # 平均人数
                if current_time > 0:
                    self.mean_in_system_class_data[node][cls].append(
                        sum(self.cumulative_in_system_class_data[node][cls]) / current_time
                    )
                    self.mean_total_class_data[node][cls].append(
                        sum(self.cumulative_total_class_data[node][cls]) / current_time
                    )
                else:
                    # current_time == 0 の場合、直前の値を利用
                    self.mean_in_system_class_data[node][cls].append(
                        self.mean_in_system_class_data[node][cls][-1] if self.mean_in_system_class_data[node][cls] else 0
                    )
                    self.mean_total_class_data[node][cls].append(
                        self.mean_total_class_data[node][cls][-1] if self.mean_total_class_data[node][cls] else 0
                    )

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


if __name__ == '__main__':
   
    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    sim_time = int(sys.argv[4])
    p_file = sys.argv[5]
    theoretical_file = sys.argv[6]
    location_file = sys.argv[7]
    speed = float(sys.argv[8]) # 1.4 徒歩の速度 (m/s)
    
    #BCMP_SFM用設定
    p = pd.read_csv(p_file, header=None)
    theoretical = pd.read_csv(theoretical_file, index_col=0, header=0) 
    location_data = pd.read_csv(location_file, header=0)
    mu = location_data['Service Rate'].values  # サービス率
    m = location_data['Num Counters'].values  # 窓口数
    locations = location_data[['X Coordinate', 'Y Coordinate']].values.tolist()
    

    type_list = np.full(N, 1) #サービスタイプはFCFS (N, R)
    bcmp = BCMP_Simulation(N, R, K_total, mu, m, type_list, p, locations, speed, theoretical, sim_time) 

    # 計算された距離と移動時間
    print("Distances (m):")
    print(bcmp.distances)
    print("Travel Times (s):")
    print(bcmp.travel_times)

    start = time.time()
    bcmp.run_simulation()



#python3 BCMP_Simulation_v5.py 33 2 100 100 transition_matrix.csv mean_L.csv locations_and_weights.csv 1.4 > result_33_2_100_100000.txt