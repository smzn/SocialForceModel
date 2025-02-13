import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_weighted_distance_avg(transition_file, distance_file, ga_results_file, mean_l_file, output_file, num_locations, num_classes):
    # ファイル読み込み
    transition_matrix = pd.read_csv(transition_file, header=None)
    distances = pd.read_csv(distance_file, header=None)
    mean_l = pd.read_csv(mean_l_file)
    
    print("Transition Matrix Shape:", transition_matrix.shape)
    print("Distances Shape:", distances.shape)
    
    # クラスごとに推移確率と距離を掛け、クラス平均を取る
    avg_weighted_matrix = pd.DataFrame(0, index=range(num_locations), columns=range(num_locations), dtype=float)
    
    for c in range(num_classes):
        start_idx = c * num_locations
        end_idx = (c + 1) * num_locations
        
        # インデックスの範囲チェックを追加
        if end_idx > transition_matrix.shape[0]:
            print(f"Warning: Class {c} indices exceed matrix dimensions")
            continue
            
        transition_submatrix = transition_matrix.iloc[start_idx:end_idx, start_idx:end_idx].copy()
        distances_subset = distances.iloc[:num_locations, :num_locations].copy()
        
        # データ型を確認
        print(f"\nClass {c} data types:")
        print("Transition submatrix dtype:", transition_submatrix.dtypes.unique())
        print("Distances dtype:", distances_subset.dtypes.unique())
        
        # 無効な値をチェック
        print(f"\nClass {c} invalid values:")
        print("Transition NaN count:", transition_submatrix.isna().sum().sum())
        print("Distances NaN count:", distances_subset.isna().sum().sum())
        
        # 行列の要素ごとの積を計算
        weighted_matrix = transition_submatrix.values * distances_subset.values
        
        # 結果を確認
        print(f"\nClass {c} weighted matrix info:")
        print("Shape:", weighted_matrix.shape)
        print("Contains NaN:", np.isnan(weighted_matrix).any())
        
        avg_weighted_matrix += pd.DataFrame(weighted_matrix)
    
    avg_weighted_matrix /= num_classes
    
    # NaNをチェック
    print("\nFinal matrix NaN check:")
    print("NaN count:", avg_weighted_matrix.isna().sum().sum())
    
    # 行方向に平均を取る（NaNを除外）
    weighted_avg_distances = avg_weighted_matrix.mean(axis=1, skipna=True)

    # mean_L.csvのClass0とClass1の合計値を計算
    total_mean_l = mean_l['Class0'] + mean_l['Class1']
    
    # weighted_avg_distancesにtotal_mean_lを掛ける(通常拠点間移動時間平均に平均系内人数をかけた場合)
    weighted_avg_distances_with_l = weighted_avg_distances * total_mean_l
    
    # 結果をデータフレームに変換し、拠点番号を追加
    result = pd.DataFrame({
        "Location": range(num_locations), 
        #"Weighted_Avg_Distance": weighted_avg_distances_with_l
        "Weighted_Avg_Distance": weighted_avg_distances
    })

    result_l = pd.DataFrame({
        "Location": range(num_locations), 
        "Weighted_Avg_Distance": weighted_avg_distances_with_l
    })


    # 棒グラフの作成
    plt.figure(figsize=(15, 6))
    plt.bar(result['Location'], result['Weighted_Avg_Distance'])
    plt.xlabel('Node Number')
    plt.ylabel('Weighted_Avg_Distance')
    plt.title('Weighted_Avg_Distance')
    plt.grid(True, alpha=0.3)
    
    # x軸の目盛りを整数で表示
    plt.xticks(range(num_locations))
    
    # グラフを保存
    plt.savefig('weighted_distance_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("グラフを保存しました: weighted_distance_bar_plot.png")

        # 棒グラフの作成
    plt.figure(figsize=(15, 6))
    plt.bar(result_l['Location'], result_l['Weighted_Avg_Distance'])
    plt.xlabel('Node Number')
    plt.ylabel('Weighted_Avg_Distance × (Class0 + Class1)')
    plt.title('Weighted_Avg_Distance with L')
    plt.grid(True, alpha=0.3)
    
    # x軸の目盛りを整数で表示
    plt.xticks(range(num_locations))
    
    # グラフを保存
    plt.savefig('weighted_distance_bar_plot_with_l.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("グラフを保存しました: weighted_distance_bar_plot.png")

    # 結果をCSVとして保存
    result.to_csv(output_file, index=False)
    
    # 結果をCSVとして保存
    result.to_csv(output_file, index=False)
    
    print(f"計算完了: 結果を {output_file} に保存しました")

    # GAの結果を読み込んで相関を計算
    ga_results = pd.read_csv(ga_results_file)
    
    # N~2N-1番目のサービス率を抽出
    service_rates = ga_results['サービス率'].iloc[num_locations:2*num_locations].values
    
    # 相関係数を計算（Pearsonの相関係数）
    correlation, p_value = stats.pearsonr(
        #weighted_avg_distances_with_l,
        weighted_avg_distances,
        service_rates
    )
    
    print("\n通常拠点間移動時間平均とダミーノードサービス率との相関分析結果:")
    print(f"Pearsonの相関係数: {correlation:.4f}")
    print(f"p値: {p_value:.4f}")

        # 相関係数を計算（Pearsonの相関係数）
    correlation, p_value = stats.pearsonr(
        weighted_avg_distances_with_l,
        service_rates
    )
    
    print("\n通常拠点間移動時間平均×平均系内人数とダミーノードサービス率との相関分析結果:")
    print(f"Pearsonの相関係数: {correlation:.4f}")
    print(f"p値: {p_value:.4f}")


    # 各拠点における総人数を計算（クラス0とクラス1の和）
    total_population = ga_results['クラス0'].iloc[num_locations:2*num_locations].values + \
                      ga_results['クラス1'].iloc[num_locations:2*num_locations].values

    # 総人数との相関係数を計算
    correlation_pop, p_value_pop = stats.pearsonr(
        #weighted_avg_distances_with_l,
        weighted_avg_distances,
        total_population
    )
    
    print("\n通常拠点間移動時間平均とダミーノードの総人数との相関分析結果:")
    print(f"Pearsonの相関係数: {correlation_pop:.4f}")
    print(f"p値: {p_value_pop:.4f}")
    
        # 総人数との相関係数を計算
    correlation_pop, p_value_pop = stats.pearsonr(
        weighted_avg_distances_with_l,
        total_population
    )
    
    print("\n通常拠点間移動時間平均×平均系内人数とダミーノードの総人数との相関分析結果:")
    print(f"Pearsonの相関係数: {correlation_pop:.4f}")
    print(f"p値: {p_value_pop:.4f}")

    # クラス別の積み上げ棒グラフを作成
    plt.figure(figsize=(15, 6))
    
    # N~2N-1のデータを使用
    class0_data = ga_results['クラス0'].iloc[num_locations:2*num_locations].values
    class1_data = ga_results['クラス1'].iloc[num_locations:2*num_locations].values
    
    # 実際の拠点番号（N~2N-1）を作成
    node_numbers = range(num_locations, 2*num_locations)
    
    # 積み上げ棒グラフの作成
    plt.bar(node_numbers, class0_data, label='Class 0')
    plt.bar(node_numbers, class1_data, bottom=class0_data, label='Class 1')
    
    plt.xlabel('Node Number')
    plt.ylabel('Number of People')
    plt.title('Number of People by Class in Each Node')
    plt.grid(True, alpha=0.3)
    plt.xticks(node_numbers)  # x軸の目盛りも実際の拠点番号に
    plt.legend()
    
    plt.savefig('class_distribution_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("クラス分布のグラフを保存しました: class_distribution_bar_plot.png")

# 使用例
N = 33
C = 2
calculate_weighted_distance_avg("transition_matrix.csv", "./Simulation/csv/distances.csv", "./Optimization/genetic_algorithm_results.csv", "mean_L.csv", "weighted_avg_distance.csv", N, C)
