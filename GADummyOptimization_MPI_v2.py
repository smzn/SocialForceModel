import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
from DummyProbabilityTranslation_v1 import DummyProbabilityTranslation
from BCMP_MVA_v4 import BCMP_MVA

class GADummyOptimization:
    def __init__(self, N, R, K_total, npop, ngen, crosspb, mutpb, lower_bound, upper_bound, transition_file_path, servicerate_file_path, sim_file_path_no_transit, sim_file_path_with_transit, size, rank):
        self.N = N  # 拠点数
        self.R = R  # クラス数
        self.num_nodes_with_dummy = self.N * self.R
        self.K_total = K_total
        self.K = [(K_total + i) // self.R for i in range(self.R)]  # 各クラスの系内人数を均等に配分
        self.type_list = np.full(self.num_nodes_with_dummy, 1)  # サービスタイプはFCFSとする
        self.npop = npop
        self.ngen = ngen
        self.scores = np.zeros(npop)  # スコアを npop のサイズで初期化
        self.crosspb = crosspb
        self.mutpb = mutpb #突然変異率
        self.best_solution = None # 最良解とスコアを初期化
        self.best_score = float('inf')  # 最小化問題の場合は無限大で初期化（最大化なら -float('inf')）
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.size = size
        self.rank = rank
        self.generation_results = []  # 各世代の最良遺伝子と全遺伝子の平均RMSEを保存するリスト
        
        # ランク0のみがシミュレーションデータを読み込む
        if rank == 0:
            self.sim_df_no_transit = pd.read_csv(sim_file_path_no_transit, usecols=['Class0', 'Class1'])
            self.sim_df_with_transit = pd.read_csv(sim_file_path_with_transit, usecols=['Class0', 'Class1'])
        else:
            self.sim_df_no_transit = None
            self.sim_df_with_transit = None

        # 各プロセスにシミュレーションデータをブロードキャスト
        self.sim_df_no_transit = comm.bcast(self.sim_df_no_transit, root=0)
        self.sim_df_with_transit = comm.bcast(self.sim_df_with_transit, root=0)

        # 推移確率行列の計算とブロードキャスト
        if rank == 0:
            df_transition = pd.read_csv(transition_file_path, header=None)
            original_transition_matrix = df_transition.values
            dummy_transition = DummyProbabilityTranslation(N, R, original_transition_matrix)
            expanded_transition_matrix = dummy_transition.expand_transition_matrix()
        else:
            expanded_transition_matrix = None

        # ブロードキャストして各プロセスが推移確率行列を受け取る
        self.expanded_transition_matrix = comm.bcast(expanded_transition_matrix, root=0)

        # ランク0のみが結果を確認する
        if rank == 0:
            print("ダミーノードを含む推移確率行列:")
            print(self.expanded_transition_matrix)
            num_rows, num_cols = self.expanded_transition_matrix.shape
            expected_size = 2 * N * R
            print(f"拡張推移確率行列のサイズ: {num_rows} x {num_cols}")
            print(f"期待されるサイズ: {expected_size} x {expected_size}")
            if num_rows == expected_size and num_cols == expected_size:
                print("変換は正しく行われました。")
            else:
                print("変換に問題があります。期待されるサイズと異なります。")

        # Num Counters と Service Rate のリスト化
        if rank == 0:
            df_servicerate = pd.read_csv(servicerate_file_path)
            counters_list = df_servicerate['Num Counters'].tolist()
            service_rate_list = df_servicerate['Service Rate'].tolist()
            dummy_num_counters = 10  # ダミーノードの窓口数
            counters_list.extend([dummy_num_counters] * N)
        else:
            counters_list = None
            service_rate_list = None

        # 各プロセスに窓口数とサービス率をブロードキャスト
        self.counters_list = comm.bcast(counters_list, root=0)
        self.service_rate_list = comm.bcast(service_rate_list, root=0)

    def initialize_population(self):
        """
        遺伝子プールを初期化する。
        - 通常ノードのサービス率は固定。
        - ダミーノードのサービス率のみ可変。
        - ダミーノードのサービス率は指定された下限・上限の範囲内でランダムに生成される。

        Args:
            lower_bound (float): ダミーノードのサービス率の下限値。
            upper_bound (float): ダミーノードのサービス率の上限値。
        """
        self.pool = []
        for _ in range(self.npop):
            # 通常ノードのサービス率（固定部分）
            fixed_part = self.service_rate_list[:self.N].copy()

            # ダミーノードのサービス率（可変部分）
            variable_part = np.random.uniform(self.lower_bound, self.upper_bound, size=self.N)

            # 結合して1つの遺伝子として格納
            individual = np.concatenate([fixed_part, variable_part])
            self.pool.append(individual)

    def evaluate_objective(self, individual):
        """
        目的関数の計算:
        - BCMPネットワークの理論値計算（ダミー）
        - 標準偏差 (RMSE) と窓口コストを合計して評価値を返す
        """
        print(individual)
        bcmp = BCMP_MVA(self.num_nodes_with_dummy, self.R, self.K, individual, self.type_list, self.expanded_transition_matrix, self.counters_list)
        L = bcmp.getMVA()
        print('L = \n{0}'.format(L))

        #移動中の客を含まない場合
        overall_rmse_no_transit = self.compare_simulation_and_bcmp(L)
        print(f"移動中客を含まないOverall RMSE: {overall_rmse_no_transit:.6f}")

        #移動中の客を拡張した推移確率で通常ノードに分配して比較する場合
        overall_rmse_with_transit = self.compare_simulation_and_bcmp_with_transit(L)
        print(f"移動中の客を含むOverall RMSE: {overall_rmse_with_transit:.6f}")

        #どちらの値を使うか選択する
        #return overall_rmse_no_transit
        return overall_rmse_with_transit

    def evaluate_population(self):
        """
        現世代の遺伝子プールを評価し、スコアを計算する。
        """
        for i in range(self.npop):
            print('遺伝子番号{0}'.format(i))
            self.scores[i] = self.evaluate_objective(self.pool[i])

    def evolve_population(self):
        """
        次世代の集団を生成する。
        通常拠点（前半 N 個）は固定し、ダミーノード（後半 N 個）のみを進化させる。
        """
        new_population = []
        
        # 既存の設定を利用
        num_nodes = self.num_nodes_with_dummy  # ダミーノードを含むノード数
        num_service_nodes = self.N             # 通常拠点の数（最初の N 個は固定）
        population_size = self.npop            # 個体数
        crossover_rate = self.crosspb          # 交叉確率
        mutation_rate = self.mutpb             # 突然変異率
        lower_bound = self.lower_bound         # サービス率の下限値
        upper_bound = self.upper_bound         # サービス率の上限値

        for _ in range(population_size):
            # 親個体を選択
            parent1, parent2 = self.select_parents()

            # 子個体の生成（交叉）
            child = self.crossover(parent1, parent2, crossover_rate)

            # 突然変異の適用
            child = self.mutate(child, mutation_rate, lower_bound, upper_bound)

            # 通常拠点（前半 N 個）のサービス率を固定
            child[:num_service_nodes] = parent1[:num_service_nodes]

            # 新しい個体を次世代集団に追加
            new_population.append(child)

        return new_population

    def select_parents(self):
        """
        親個体を選択する。トーナメント選択を使用。
        """
        tournament_size = 3

        # ランダムに候補個体のインデックスを選択
        candidate_indices1 = random.sample(range(self.npop), tournament_size)
        candidate_indices2 = random.sample(range(self.npop), tournament_size)

        # 各候補インデックスのスコアを基に親個体を選択
        selected1_idx = min(candidate_indices1, key=lambda idx: self.scores[idx])
        selected2_idx = min(candidate_indices2, key=lambda idx: self.scores[idx])

        # 選択された個体を返す
        return self.pool[selected1_idx], self.pool[selected2_idx]

    def crossover(self, parent1, parent2, crossover_rate):
        """
        交叉を実行する。一定確率で親個体を交叉させ、子個体を生成する。
        """
        if np.random.rand() < crossover_rate:
            num_nodes = len(parent1)
            # ダミーノード（後半部分）の一様交叉（ランダムに親1または親2を選択）
            child = parent1.copy()
            for i in range(self.N, num_nodes):
                if np.random.rand() < 0.5:
                    child[i] = parent2[i]
            return child
        else:
            return parent1.copy()

    def mutate(self, individual, mutation_rate, lower_bound, upper_bound):
        """
        突然変異を実行する。一定確率でダミーノード部分に突然変異を加える。
        """
        for i in range(self.N, len(individual)):
            if np.random.rand() < mutation_rate:
                # 下限値と上限値の間でランダムなサービス率に変更
                individual[i] = np.random.uniform(lower_bound, upper_bound)
        return individual
                
    def run(self):
        """
        MPIを利用した遺伝的アルゴリズムの並列実行。
        """
        comm = MPI.COMM_WORLD  # MPI通信オブジェクト

        # 初期集団の生成（ランク0が生成）
        if self.rank == 0:
            self.initialize_population()  # ここで self.pool が定義される
        else:
            self.pool = None

        # 初期集団を全プロセスにブロードキャスト
        self.pool = comm.bcast(self.pool, root=0)

        # 担当する範囲を計算 (npopを均等に分割)
        num_tasks_per_proc = self.npop // self.size
        remainder = self.npop % self.size
        if self.rank < remainder:
            start_idx = self.rank * (num_tasks_per_proc + 1)
            end_idx = start_idx + (num_tasks_per_proc + 1)
        else:
            start_idx = self.rank * num_tasks_per_proc + remainder
            end_idx = start_idx + num_tasks_per_proc

        for gen in range(self.ngen):
            print(f"Generation {gen + 1}, Rank {self.rank} processing...")

            # 担当遺伝子を評価
            local_scores = []
            for i in range(start_idx, end_idx):
                score = self.evaluate_objective(self.pool[i])
                local_scores.append((i, score))

            # 各プロセスからスコアを集約
            all_scores = comm.gather(local_scores, root=0)

            # 集約されたスコアの更新
            if self.rank == 0:
                # 全てのスコアをフラット化して統合
                flat_scores = [item for sublist in all_scores for item in sublist]
                for idx, score in flat_scores:
                    self.scores[idx] = score

                # 最良解の更新
                best_gen_idx = np.argmin(self.scores)
                if self.scores[best_gen_idx] < self.best_score:
                    self.best_solution = self.pool[best_gen_idx]
                    self.best_score = self.scores[best_gen_idx]
                    print(f"New best found: {self.best_score}")

                # **世代ごとの最良・平均RMSE値を記録**
                best_rmse = self.scores[best_gen_idx]
                average_rmse = np.mean(self.scores)
                self.generation_results.append((gen + 1, best_rmse, average_rmse))

                # 次世代集団の生成
                elite = self.pool[best_gen_idx]  # エリート保存
                next_generation = self.evolve_population()
                next_generation[0] = elite  # エリートを次世代の最初に配置
            else:
                next_generation = None

            # 次世代集団を全プロセスにブロードキャスト
            self.pool = comm.bcast(next_generation, root=0)

        # 最良結果を保存（ランク0のみ）
        if self.rank == 0:
            return self.best_solution, self.best_score
            
    def calculate_rmse(self, sim_values, theoretical_values):
        """
        RMSE (Root Mean Square Error) を計算する関数。

        Args:
            sim_values (numpy.ndarray): シミュレーションデータ
            theoretical_values (numpy.ndarray): BCMPモデルからの理論データ

        Returns:
            float: RMSE値
        """
        return np.sqrt(np.mean((sim_values - theoretical_values) ** 2))

            
    def compare_simulation_and_bcmp(self, L):
        """
        シミュレーション結果とBCMPモデル結果を比較し、RMSEを計算・保存する関数。

        Args:
            sim_file_path (str): シミュレーション結果のCSVファイルのパス
            L (numpy.ndarray): BCMPモデルの結果行列
            num_nodes (int): 通常ノード数 (ダミーノードを除く)
            output_file (str): RMSE結果を保存するCSVファイルのパス

        Returns:
            None
        """

        # BCMPの結果から最初のnum_nodesのみを抽出 (移動中の客を含まない)
        L_reduced = L[:num_nodes, :]  # ダミーノードを除いた通常拠点の結果

        # シミュレーションデータとBCMP結果のサイズを確認
        assert self.sim_df_no_transit.shape == L_reduced.shape, f"データサイズが一致しません。シミュレーション: {self.sim_df_no_transit.shape}, BCMP: {L_reduced.shape}"

        # RMSEの計算（クラスごとに）
        rmse_class0 = self.calculate_rmse(self.sim_df_no_transit['Class0'].values, L_reduced[:, 0])
        rmse_class1 = self.calculate_rmse(self.sim_df_no_transit['Class1'].values, L_reduced[:, 1])

        # 全体のRMSEを単純平均として計算
        overall_rmse = (rmse_class0 + rmse_class1) / 2

        # RMSEの結果を表示
        print(f"RMSE (Class 0): {rmse_class0:.6f}")
        print(f"RMSE (Class 1): {rmse_class1:.6f}")
        print(f"Overall RMSE: {overall_rmse:.6f}")

        return overall_rmse
    
    def distribute_dummy_nodes_to_real_nodes(self, L):
        """
        ダミーノードの平均系内人数を通常拠点に分配する関数。

        Args:
            L (numpy.ndarray): BCMPモデルの平均系内人数（通常拠点＋ダミーノード）
            num_nodes (int): 通常ノードの数
            num_classes (int): クラス数
            transition_matrix (numpy.ndarray): 拡張された推移確率行列（ダミーノードを含む）

        Returns:
            numpy.ndarray: ダミーノードの人数を分配後の通常ノードの平均系内人数
        """
        # 通常ノードの平均系内人数
        L_real_nodes = L[:self.N, :]

        # ダミーノードの平均系内人数
        L_dummy_nodes = L[self.N:, :]

        # クラスごとに分配処理
        for c in range(num_classes):
            start_index = self.N * (self.R + c)  # ダミーノードの開始インデックス
            end_index = start_index + self.N  # ダミーノードの終了インデックス

            # クラス c に属するダミーノードの行き先確率
            transition_prob = self.expanded_transition_matrix[start_index:end_index, :self.N]

            # ダミーノードの系内人数を通常ノードに加算
            for i in range(self.N):
                inflow_from_dummy = np.sum(L_dummy_nodes[:, c] * transition_prob[:, i])
                L_real_nodes[i, c] += inflow_from_dummy

        return L_real_nodes

    
    def compare_simulation_and_bcmp_with_transit(self, L):
        """
        シミュレーション結果とBCMPモデル結果を比較し、移動中の客を含めたRMSEを計算・保存する関数。

        Args:
            sim_file_path (str): シミュレーション結果のCSVファイルのパス
            L (numpy.ndarray): BCMPモデルの結果行列
            num_nodes (int): 通常ノードの数
            num_classes (int): クラス数
            transition_matrix (numpy.ndarray): 推移確率行列（拡張済み）
            output_file (str): RMSE結果を保存するCSVファイルのパス

        Returns:
            float: 全体のRMSE値
        """

        # ダミーノードの影響を考慮し、通常ノードの値を調整
        L_adjusted = self.distribute_dummy_nodes_to_real_nodes(L)

        # シミュレーションデータとBCMP結果のサイズを確認
        assert self.sim_df_with_transit.shape == L_adjusted.shape, f"データサイズが一致しません。シミュレーション: {self.sim_df_with_transit.shape}, BCMP: {L_adjusted.shape}"

        # RMSEの計算（クラスごとに）
        rmse_class0 = self.calculate_rmse(self.sim_df_with_transit['Class0'].values, L_adjusted[:, 0])
        rmse_class1 = self.calculate_rmse(self.sim_df_with_transit['Class1'].values, L_adjusted[:, 1])

        # 全体のRMSEを単純平均として計算
        overall_rmse = (rmse_class0 + rmse_class1) / 2

        # RMSEの結果を表示
        print(f"RMSE (Class 0): {rmse_class0:.6f}")
        print(f"RMSE (Class 1): {rmse_class1:.6f}")
        print(f"Overall RMSE: {overall_rmse:.6f}")

        return overall_rmse

    def save_results(self):
        """
        最良遺伝子、スコア、平均系内人数を計算・表示・保存する。
        """
        # 最良遺伝子とスコアを取得
        best_solution = self.best_solution
        best_score = self.best_score

        # サービス率の分割（通常拠点とダミーノード）
        service_rate_normal = best_solution[:self.N]  # 通常拠点のサービス率
        service_rate_dummy = best_solution[self.N:]   # ダミーノードのサービス率

        # 平均系内人数を計算
        bcmp = BCMP_MVA(self.num_nodes_with_dummy, self.R, self.K, self.best_solution, self.type_list, self.expanded_transition_matrix, self.counters_list)
        L = bcmp.getMVA()

        # 結果を表示
        print("\n--- 遺伝アルゴリズム最終結果 ---")
        print(f"最良スコア (RMSE値): {best_score}")
        print("最良遺伝子（サービス率）:")
        print("通常拠点:", service_rate_normal)
        print("ダミーノード:", service_rate_dummy)
        print("平均系内人数:", L)

        # 平均系内人数 L を Pandas DataFrame に変換する
        average_queue_lengths_df = pd.DataFrame(L, columns=[f'クラス{r}' for r in range(L.shape[1])])

        # 拠点番号を追加
        average_queue_lengths_df.insert(0, '拠点番号', list(range(L.shape[0])))

        # サービス率のデータフレームも同様に作成
        service_rate_df = pd.DataFrame({
            '拠点番号': list(range(len(best_solution))),
            'サービス率': best_solution
        })

        # サービス率と平均系内人数をマージして最終結果のデータフレームを作成
        results_df = service_rate_df.merge(average_queue_lengths_df, on='拠点番号')

        # メタ情報（RMSE、遺伝子数、世代数）を保存
        metadata_df = pd.DataFrame({
            '指標': ['RMSE', '遺伝子数', '世代数'],
            '値': [best_score, self.npop, self.ngen]
        })

        # データフレームをCSVに保存
        results_df.to_csv('genetic_algorithm_results.csv', index=False, encoding='utf-8-sig')
        metadata_df.to_csv('genetic_algorithm_metadata.csv', index=False, encoding='utf-8-sig')

        print("\n結果が 'genetic_algorithm_results.csv', 'genetic_algorithm_metadata.csv' に保存されました。")

        # 積み上げグラフの作成・保存
        self.plot_stacked_bar_chart(L)
        self.plot_dummy_nodes_stacked_bar_chart(L)

    def save_generation_results(self):
        """
        Save the best and average RMSE values of each generation to a CSV file 
        and draw a line plot showing the progression.
        """

        # Save to CSV file
        df_results = pd.DataFrame(self.generation_results, columns=['Generation', 'Best RMSE', 'Average RMSE'])
        df_results.to_csv('generation_rmse_results.csv', index=False, encoding='utf-8-sig')

        # Draw line plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['Generation'], df_results['Best RMSE'], marker='o', label='Best RMSE', linestyle='-')
        plt.plot(df_results['Generation'], df_results['Average RMSE'], marker='x', label='Average RMSE', linestyle='--')

        # Graph customization
        plt.xlabel('Generation')
        plt.ylabel('RMSE Value')
        plt.title('Best and Average RMSE per Generation')
        plt.legend()
        plt.grid(True)

        # Save the graph
        plt.savefig('generation_rmse_graph.png')
        plt.close()

        print("\n世代ごとのRMSE結果が 'generation_rmse_results.csv' と 'generation_rmse_graph.png' に保存されました。")

    def plot_stacked_bar_chart(self, L):
        """
        通常拠点と対応するダミーノードの平均系内人数を積み上げグラフで可視化。
        """
        num_classes = L.shape[1]  # クラス数
        normal_L = L[:self.N]  # 通常拠点のデータ
        dummy_L = L[self.N:]  # ダミーノードのデータ

        # 通常拠点の番号を取得
        node_indices = np.arange(self.N)

        # 積み上げデータ準備
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(self.N)  # 積み上げ開始位置

        for r in range(num_classes):
            normal_values = normal_L[:, r]
            dummy_values = dummy_L[:, r]
            ax.bar(node_indices, normal_values, label=f'Class {r} Normal', bottom=bottom)
            bottom += normal_values
            ax.bar(node_indices, dummy_values, label=f'Class {r} Dummy', bottom=bottom)
            bottom += dummy_values

        # ラベルとタイトル設定
        ax.set_xlabel("Node")
        ax.set_ylabel("Queue Length")
        ax.set_title("Stacked Bar Chart of Average Queue Lengths")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

        # グラフ保存
        plt.savefig('stacked_bar_chart.png', bbox_inches='tight')
        plt.close()
        print("積み上げグラフが 'stacked_bar_chart.png' に保存されました。")

    def plot_dummy_nodes_stacked_bar_chart(self, L):
        """
        ダミーノードのみのクラス別積み上げグラフを作成。
        """
        num_classes = L.shape[1]  # クラス数
        dummy_L = L[self.N:]  # ダミーノードのデータ

        # ダミーノードの番号を取得
        dummy_indices = np.arange(self.N)

        # 積み上げデータ準備
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(self.N)  # 積み上げ開始位置

        for r in range(num_classes):
            dummy_values = dummy_L[:, r]
            ax.bar(dummy_indices, dummy_values, label=f'Class {r} Dummy', bottom=bottom)
            bottom += dummy_values

        # ラベルとタイトル設定
        ax.set_xlabel("Dummy Node")
        ax.set_ylabel("Queue Length")
        ax.set_title("Stacked Bar Chart of Dummy Nodes")
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

        # グラフ保存
        plt.savefig('dummy_nodes_stacked_bar_chart.png', bbox_inches='tight')
        plt.close()
        print("積み上げグラフが　'dummy_nodes_stacked_bar_chart.png' に保存されました。")


if __name__ == '__main__':
    # パラメータ設定
    num_nodes = 33  # 拠点数
    num_classes = 2  # クラス数
    K_total = 100
    transition_file_path = "../1.TheoreticalValue/transition_matrix.csv"
    servicerate_file_path = '../1.TheoreticalValue/locations_and_weights.csv' # 窓口数、サービス率
    npop = 76  # 遺伝子数
    ngen = 100  # 世代数
    crosspb = 0.8  # 交叉確率(0.6 ～ 0.9)
    mutpb = 0.05  # 突然変異確率(0.01 ～ 0.2)
    lower_bound = 1.0
    upper_bound=10.0
    sim_file_path_no_transit = "../2.Simulation/log/mean_L_no_transit_final_rmse.csv"
    sim_file_path_with_transit = "../2.Simulation/log/mean_L_with_transit_final_rmse.csv" # 移動中の客を拡張した推移確率で通常ノードに分配して比較する場合

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    # 時間計測開始
    start_time = time.time()

    dummy_optimization = GADummyOptimization(num_nodes, num_classes, K_total, npop, ngen, crosspb, mutpb, lower_bound, upper_bound, transition_file_path, servicerate_file_path, sim_file_path_no_transit, sim_file_path_with_transit, size, rank)
    dummy_optimization.run() #遺伝アルゴリズムの実施
    if rank == 0:
        dummy_optimization.save_results()
        dummy_optimization.save_generation_results()

    if rank == 0:
        # 計算時間の表示
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n計算時間: {elapsed_time:.2f} 秒")

#mpiexec -n 4 python3 GADummyOptimization_MPI_v2.py
