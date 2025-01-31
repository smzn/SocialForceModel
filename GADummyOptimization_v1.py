import numpy as np
import pandas as pd
import random
from DummyProbabilityTranslation_v1 import DummyProbabilityTranslation
from BCMP_MVA_v4 import BCMP_MVA

class GADummyOptimization:
    def __init__(self, N, R, K_total, npop, ngen, crosspb, mutpb, lower_bound, upper_bound, transition_file_path, servicerate_file_path, sim_file_path_no_transit, sim_file_path_with_transit):
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
        self.sim_df_no_transit = pd.read_csv(sim_file_path_no_transit, usecols=['Class0', 'Class1'])
        self.sim_df_with_transit = pd.read_csv(sim_file_path_with_transit, usecols=['Class0', 'Class1'])
        
        #1. ダミーノードを含めた推移確率を作成
        df_transition = pd.read_csv(transition_file_path, header=None)
        original_transition_matrix = df_transition.values
        dummy_transition = DummyProbabilityTranslation(num_nodes, num_classes, original_transition_matrix)
        self.expanded_transition_matrix = dummy_transition.expand_transition_matrix() # ダミーノードを含む推移確率行列を計算
        # 結果を表示
        print("ダミーノードを含む推移確率行列:")
        print(self.expanded_transition_matrix)

        # 行数と列数の確認
        num_rows, num_cols = self.expanded_transition_matrix.shape
        expected_size = 2 * num_nodes * num_classes

        # 結果の表示
        print(f"拡張推移確率行列のサイズ: {num_rows} x {num_cols}")
        print(f"期待されるサイズ: {expected_size} x {expected_size}")

        # サイズが期待通りかの確認
        if num_rows == expected_size and num_cols == expected_size:
            print("変換は正しく行われました。")
        else:
            print("変換に問題があります。期待されるサイズと異なります。")

        # Num Counters と Service Rate のリスト化
        df_servicerate = pd.read_csv(servicerate_file_path)
        self.counters_list = df_servicerate['Num Counters'].tolist()
        dummy_num_counters = 10  # ダミーノードの窓口数
        self.counters_list.extend([dummy_num_counters] * num_nodes) # ダミーノードの追加
        self.service_rate_list = df_servicerate['Service Rate'].tolist()

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
        遺伝的アルゴリズムを実行する。
        """
        self.initialize_population()
        
        for gen in range(self.ngen):
            print(f"Generation {gen + 1} processing...")
            # 現世代の遺伝子を評価
            self.evaluate_population()
            print('各遺伝子の評価値: {0}'.format(self.scores))

            
            # 最良遺伝子の更新
            best_gen_idx = np.argmin(self.scores)
            if self.scores[best_gen_idx] < self.best_score:
                self.best_solution = self.pool[best_gen_idx]
                self.best_score = self.scores[best_gen_idx]
                print(f"New best found: {self.best_score}")

            # エリート保存: 最良遺伝子を次世代に引き継ぐ
            elite = self.pool[best_gen_idx]

            # 次世代の生成
            next_generation = self.evolve_population()
            next_generation[0] = elite  # エリート遺伝子を次世代の最初に配置
            # 次世代遺伝子を表示（確認のため）
            print("\n次世代の遺伝子集団:")
            for idx, individual in enumerate(next_generation):
                print(f"個体 {idx + 1}: {individual}")
            self.pool = next_generation
            
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


if __name__ == '__main__':
    # パラメータ設定
    num_nodes = 33  # 拠点数
    num_classes = 2  # クラス数
    K_total = 100
    transition_file_path = "transition_matrix.csv"
    servicerate_file_path = 'locations_and_weights.csv' # 窓口数、サービス率
    npop = 3  # 遺伝子数
    ngen = 1  # 世代数
    crosspb = 0.8  # 交叉確率(0.6 ～ 0.9)
    mutpb = 0.05  # 突然変異確率(0.01 ～ 0.2)
    lower_bound = 1.0
    upper_bound=10.0
    sim_file_path_no_transit = "./log/mean_L_no_transit_final_rmse.csv"
    sim_file_path_with_transit = "./log/mean_L_with_transit_final_rmse.csv" # 移動中の客を拡張した推移確率で通常ノードに分配して比較する場合

    dummy_optimization = GADummyOptimization(num_nodes, num_classes, K_total, npop, ngen, crosspb, mutpb, lower_bound, upper_bound, transition_file_path, servicerate_file_path, sim_file_path_no_transit, sim_file_path_with_transit)
    dummy_optimization.run() #遺伝アルゴリズムの実施
    dummy_optimization.save_results()

