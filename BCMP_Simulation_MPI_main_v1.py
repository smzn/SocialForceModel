from mpi4py import MPI
import sys
import numpy as np
import pandas as pd
from BCMP_Simulation_v6 import BCMP_Simulation

if __name__ == '__main__':
    # MPIの初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 共通パラメータの取得 (各プロセスが同じ設定で動作する)
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    sim_time = int(sys.argv[4])
    p_file = sys.argv[5]
    theoretical_file = sys.argv[6]
    location_file = sys.argv[7]
    speed = float(sys.argv[8])

    p = pd.read_csv(p_file, header=None)
    theoretical = pd.read_csv(theoretical_file, index_col=0, header=0) 
    location_data = pd.read_csv(location_file, header=0)
    mu = location_data['Service Rate'].values  # サービス率
    m = location_data['Num Counters'].values  # 窓口数
    locations = location_data[['X Coordinate', 'Y Coordinate']].values.tolist()

    type_list = np.full(N, 1)  # サービスタイプはFCFS

    # BCMP Simulation オブジェクト作成
    bcmp = BCMP_Simulation(N, R, K_total, mu, m, type_list, p, locations, speed, theoretical, sim_time)

    # シミュレーション実行
    bcmp.run_simulation()

    # RMSEの計算
    final_total_rmse, final_class_rmse, min_total_rmse, min_class_rmse = bcmp.calculate_rmse_summary()

    # 各プロセスが計算したRMSEをRank 0に送信
    local_rmse = final_total_rmse
    all_rmse = comm.gather(local_rmse, root=0)

    # Rank 0プロセスが最小RMSEを持つプロセスを特定
    if rank == 0:
        min_rmse = min(all_rmse)
        best_rank = all_rmse.index(min_rmse)
        print(f"Best RMSE: {min_rmse} from Rank {best_rank}")
    else:
        best_rank = None

    # 全プロセスに最小RMSEを持つプロセスのRankをブロードキャスト
    best_rank = comm.bcast(best_rank, root=0)

    # 最小RMSEを持つプロセスだけがシミュレーション結果を保存
    if rank == best_rank:
        bcmp.process_simulation_results()
        print(f"Process {rank}: Results saved.")
        print(f"Process {rank} RMSE Summary (with transit):")
        print(f"  - Final Total RMSE (with transit): {final_total_rmse}")
        print(f"  - Final Class-wise RMSE (with transit): {final_class_rmse}")
        print(f"  - Min Total RMSE (with transit): {min_total_rmse}")
        print(f"  - Min Class-wise RMSE (with transit): {min_class_rmse}")
        
        # 移動中の顧客を含まないRMSE
        final_total_rmse_without_transit, final_class_rmse_without_transit, min_total_rmse_without_transit, min_class_rmse_without_transit = bcmp.calculate_rmse_without_transit()        # 結果を表示
        print(f"Without Transit:")
        print(f"  Final Total RMSE: {final_total_rmse_without_transit}")
        print(f"  Final Class-wise RMSE: {final_class_rmse_without_transit}")
        print(f"  Min Total RMSE: {min_total_rmse_without_transit}")
        print(f"  Min Class-wise RMSE: {min_class_rmse_without_transit}")
    else:
        print(f"Process {rank}: Skipping result saving.")

#mpiexec -n 8 python3 BCMP_Simulation_MPI_main_v1.py 33 2 100 1000 transition_matrix.csv mean_L.csv locations_and_weights.csv 1.4
