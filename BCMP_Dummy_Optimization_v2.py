import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from BCMP_MVA_v3 import BCMP_MVA  # BCMP_MVAクラスが含まれたファイルをimport
from DummyProbabilityTranslation_v1 import DummyProbabilityTranslation
import time

def calculate_rmse(sim_values, theoretical_values):
    """
    RMSE (Root Mean Square Error) を計算する関数。

    Args:
        sim_values (numpy.ndarray): シミュレーションデータ
        theoretical_values (numpy.ndarray): BCMPモデルからの理論データ

    Returns:
        float: RMSE値
    """
    return np.sqrt(np.mean((sim_values - theoretical_values) ** 2))

def compare_simulation_and_bcmp(sim_file_path, L, num_nodes, output_file):
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

    # シミュレーション結果の読み込み（node_id を無視）
    sim_df = pd.read_csv(sim_file_path, usecols=['Class0', 'Class1'])

    # BCMPの結果から最初のnum_nodesのみを抽出 (移動中の客を含まない)
    L_reduced = L[:num_nodes, :]  # ダミーノードを除いた通常拠点の結果

    # シミュレーションデータとBCMP結果のサイズを確認
    assert sim_df.shape == L_reduced.shape, f"データサイズが一致しません。シミュレーション: {sim_df.shape}, BCMP: {L_reduced.shape}"

    # RMSEの計算（クラスごとに）
    rmse_class0 = calculate_rmse(sim_df['Class0'].values, L_reduced[:, 0])
    rmse_class1 = calculate_rmse(sim_df['Class1'].values, L_reduced[:, 1])

    # 全体のRMSEを単純平均として計算
    overall_rmse = (rmse_class0 + rmse_class1) / 2

    # RMSEの結果を表示
    print(f"RMSE (Class 0): {rmse_class0:.6f}")
    print(f"RMSE (Class 1): {rmse_class1:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")

    # RMSEの結果を保存
    rmse_results = pd.DataFrame({
        "Class": ["Class 0", "Class 1", "Overall"],
        "RMSE": [rmse_class0, rmse_class1, overall_rmse]
    })
    rmse_results.to_csv(output_file, index=False)
    print(f"RMSEの結果を {output_file} に保存しました。")

    return overall_rmse

def distribute_dummy_nodes_to_real_nodes(L, num_nodes, num_classes, transition_matrix):
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
    L_real_nodes = L[:num_nodes, :]

    # ダミーノードの平均系内人数
    L_dummy_nodes = L[num_nodes:, :]

    # クラスごとに分配処理
    for c in range(num_classes):
        start_index = num_nodes * (num_classes + c)  # ダミーノードの開始インデックス
        end_index = start_index + num_nodes  # ダミーノードの終了インデックス

        # クラス c に属するダミーノードの行き先確率
        transition_prob = transition_matrix[start_index:end_index, :num_nodes]

        # ダミーノードの系内人数を通常ノードに加算
        for i in range(num_nodes):
            inflow_from_dummy = np.sum(L_dummy_nodes[:, c] * transition_prob[:, i])
            L_real_nodes[i, c] += inflow_from_dummy

    return L_real_nodes

def compare_simulation_and_bcmp_with_transit(sim_file_path, L, num_nodes, num_classes, transition_matrix, output_file):
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

    # シミュレーション結果の読み込み（node_id を無視）
    sim_df = pd.read_csv(sim_file_path, usecols=['Class0', 'Class1'])

    # ダミーノードの影響を考慮し、通常ノードの値を調整
    L_adjusted = distribute_dummy_nodes_to_real_nodes(L, num_nodes, num_classes, transition_matrix)

    # シミュレーションデータとBCMP結果のサイズを確認
    assert sim_df.shape == L_adjusted.shape, f"データサイズが一致しません。シミュレーション: {sim_df.shape}, BCMP: {L_adjusted.shape}"

    # RMSEの計算（クラスごとに）
    rmse_class0 = calculate_rmse(sim_df['Class0'].values, L_adjusted[:, 0])
    rmse_class1 = calculate_rmse(sim_df['Class1'].values, L_adjusted[:, 1])

    # 全体のRMSEを単純平均として計算
    overall_rmse = (rmse_class0 + rmse_class1) / 2

    # RMSEの結果を表示
    print(f"RMSE (Class 0): {rmse_class0:.6f}")
    print(f"RMSE (Class 1): {rmse_class1:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")

    # RMSEの結果を保存
    rmse_results = pd.DataFrame({
        "Class": ["Class 0", "Class 1", "Overall"],
        "RMSE": [rmse_class0, rmse_class1, overall_rmse]
    })
    rmse_results.to_csv(output_file, index=False)
    print(f"RMSEの結果を {output_file} に保存しました。")

    return overall_rmse


if __name__ == "__main__":
    # 1. 元の推移確率を取り込んでダミーノードを含めた推移確率を作成
    num_nodes = 33  # 拠点数
    num_classes = 2  # クラス数

    file_path = "transition_matrix.csv" 
    df = pd.read_csv(file_path, header=None)
    original_transition_matrix = df.values

    dummy_optimizer = DummyProbabilityTranslation(num_nodes, num_classes, original_transition_matrix)

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
    bcmp.save_mean_L_to_csv(L, "mean_L_with_dummy.csv", "mean_L_with_dummy_stacked_bar.png")

    # 3.Simulation値とダミーノードの値を比較する
    # 移動中の客は別にして比較する場合
    sim_file_path = "./log/mean_L_no_transit_final_rmse.csv"
    output_file = "rmse_results_no_transit_vs_simulation.csv"
    overall_rmse = compare_simulation_and_bcmp(sim_file_path, L, num_nodes, output_file)
    print(f"移動中客を含まないOverall RMSE: {overall_rmse:.6f}")

    # 移動中の客を拡張した推移確率で通常ノードに分配して比較する場合
    sim_file_path = "./log/mean_L_with_transit_final_rmse.csv"
    output_file = "rmse_results_with_transit_vs_simulation.csv"
    
    # 事前に計算済みのLと推移確率行列を使用
    overall_rmse = compare_simulation_and_bcmp_with_transit(sim_file_path, L, num_nodes, num_classes, expanded_transition_matrix, output_file)
    print(f"移動中の客を含むOverall RMSE: {overall_rmse:.6f}")



