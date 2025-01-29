import numpy as np
import pandas as pd
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
        次世代の遺伝子プールを生成する。
        """
        new_population = []
        for _ in range(self.npop // 2):
            # トーナメント選択
            p1, p2 = self.selection(), self.selection()
            # 交叉と突然変異
            c1, c2 = self.crossover(p1, p2)
            self.mutation(c1)
            self.mutation(c2)
            new_population.extend([c1, c2])
        return new_population

    def selection(self, k=3):
        """
        トーナメント選択: 最良の遺伝子を選択する。
        """
        indices = np.random.randint(0, self.npop, k)
        best_idx = indices[np.argmin([self.scores[i] for i in indices])]
        return self.pool[best_idx]

    def crossover(self, p1, p2):
        """
        遺伝子交叉を行う。
        """
        if np.random.rand() < self.crosspb:
            pt = np.random.randint(1, self.N - 1)
            return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])
        return p1.copy(), p2.copy()

    def mutation(self, individual):
        """
        遺伝子突然変異を行う。
        """
        for i in range(len(individual)):
            if np.random.rand() < self.mutpb:
                individual[i] = np.random.randint(1, self.U + 1)

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

            '''
            # 最良遺伝子の更新
            best_gen_idx = np.argmin(self.scores)
            if self.scores[best_gen_idx] < self.best_score:
                self.best_solution = self.pool[best_gen_idx]
                self.best_score = self.scores[best_gen_idx]
                print(f"New best found: {self.best_score}")

            # 次世代の生成
            self.pool = self.evolve_population()

        return self.best_solution, self.best_score
        '''
            
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

    '''
    # 初期遺伝子プール確認
    print("\nInitial Population:")
    for i, individual in enumerate(dummy_optimization.pool):
        print(f"Individual {i}: {individual}")
    '''
