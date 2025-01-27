import numpy as np
from scipy.linalg import block_diag

class DummyProbabilityTranslation:
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
