import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from BCMP_MVA_v3 import BCMP_MVA  # BCMP_MVAクラスが含まれたファイルをimport
import time

class DummyOptimization:
    def __init__(self, num_nodes, num_classes, original_matrix):
        """
        クラスの初期化

        :param num_nodes: 拠点数 (N)
        :param num_classes: クラス数 (C)
        """
        self.N = num_nodes
        self.num_classes = num_classes
        self.original_matrix = original_matrix

    def expand_transition_matrix(self, output_file_path = "expanded_transition_matrix.csv"):
        """
        ダミーノードを含む拡張推移確率行列を作成（クラス数対応）

        :return: (numpy.ndarray) ダミーノードを含む拡張推移確率行列 (サイズ: 2NC x 2NC)
        """
        N, C = self.N, self.num_classes

        # クラスごとに拡張行列を作成
        expanded_matrices = []
        for c in range(C):
            # クラスごとの元の推移確率行列 (N×N)
            P_c = self.original_matrix[c * N:(c + 1) * N, c * N:(c + 1) * N]
            
            # ゼロ行列と単位行列の作成
            zero_matrix_N = np.zeros((N, N))
            identity_matrix_N = np.eye(N)

            # クラス c の拡張行列
            P_d_c = np.block([
                [zero_matrix_N, identity_matrix_N],
                [P_c, zero_matrix_N]
            ])

            expanded_matrices.append(P_d_c)

        # ブロック対角行列としてまとめる (全クラスの行列を結合)
        P_d = block_diag(*expanded_matrices)

        #np.savetxt(output_file_path, P_d, delimiter=",")
        np.savetxt(output_file_path, P_d, delimiter=",", fmt="%.6f")

        return P_d


if __name__ == "__main__":
    # 1. 元の推移確率を取り込んでダミーノードを含めた推移確率を作成
    num_nodes = 33  # 拠点数
    num_classes = 2  # クラス数

    file_path = "transition_matrix.csv" 
    df = pd.read_csv(file_path, header=None)
    original_transition_matrix = df.values

    dummy_optimizer = DummyOptimization(num_nodes, num_classes, original_transition_matrix)

    # ダミーノードを含む推移確率行列を計算
    expanded_transition_matrix = dummy_optimizer.expand_transition_matrix()

    # 結果を表示
    print("ダミーノードを含む推移確率行列:")
    print(expanded_transition_matrix)

    # 行数と列数の確認
    num_rows, num_cols = expanded_transition_matrix.shape
    expected_size = 2 * num_nodes * num_classes

    # 結果の表示
    print(f"拡張推移確率行列のサイズ: {num_rows} x {num_cols}")
    print(f"期待されるサイズ: {expected_size} x {expected_size}")

    # サイズが期待通りかの確認
    if num_rows == expected_size and num_cols == expected_size:
        print("変換は正しく行われました。")
    else:
        print("変換に問題があります。期待されるサイズと異なります。")

    #2. BCMP_MVAによる平均系内人数(ダミーノードを含めた推移確率を利用、理論値の計算)
    num_nodes_with_dummy = 33 * 2  # 拠点数
    K_total=100
    K = [(K_total + i) // num_classes for i in range(num_classes)]  # 各クラスの系内人数を均等に配分
    type_list = np.full(num_nodes_with_dummy, 1)  # サービスタイプはFCFSとする

    # CSVファイルの読み込み
    filename = 'locations_and_weights.csv'
    df = pd.read_csv(filename)

    # Num Counters と Service Rate のリスト化
    counters_list = df['Num Counters'].tolist()
    dummy_num_counters = 10  # ダミーノードの値
    counters_list.extend([dummy_num_counters] * num_nodes) # ダミーノードの追加
    service_rate_list = df['Service Rate'].tolist()
    # Service Rate のダミーノードを追加（最初の値から順番にコピー）
    for i in range(num_nodes):
        service_rate_list.append(service_rate_list[i % len(service_rate_list)])
    # クラス数分コピーして2次元リストを作成し、numpy 配列に変換
    service_rate_2d_list = np.array([service_rate_list[:] for _ in range(num_classes)])
    #service_rate_2d_list = [service_rate_list[:] for _ in range(num_classes)]

    # 結果の表示
    print("Counters List:", counters_list)
    print("Service Rate List:", service_rate_list)
    print("Service Rate 2D List with Dummy Nodes:")
    for row in service_rate_2d_list:
        print(row)

    bcmp = BCMP_MVA(N=num_nodes_with_dummy, R=num_classes, K=K, mu=service_rate_2d_list, type_list=type_list, p=expanded_transition_matrix, m=counters_list)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    print('L = \n{0}'.format(L))
    bcmp.save_mean_L_to_csv(L, "mean_L_with_dummy.csv")



    



