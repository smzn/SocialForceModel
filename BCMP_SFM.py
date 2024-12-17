from GravityTransition import GravityTransition
from BCMP_MVA_v3 import BCMP_MVA  # BCMP_MVAクラスが含まれたファイルをimport
import numpy as np
import pandas as pd
import time

# 1. GravityTransitionクラスによる推移確率生成
# インスタンス作成
num_locations=33
num_customer_classes=2
model = GravityTransition(num_locations, num_customer_classes)

# 推移確率行列を作成し保存
transition_matrix = model.save_transition_matrix()
print(transition_matrix)
# 行数と列数を表示
rows, cols = transition_matrix.shape
print(f"Transition Matrix: {rows} rows, {cols} columns")

# 窓口数とサービス率を取得
num_counters, service_rates = model.get_counters_and_service_rates()
print("窓口数:", num_counters)
print("サービス率:", service_rates)

# 拠点位置をプロットし保存
model.plot_locations(num_counters)

# 人気度をプロットし保存
model.plot_weights()

# 拠点の位置と人気度を保存
model.save_locations_and_weights("locations_and_weights.csv")


# 2. BCMP_MVAによる平均系内人数(理論値の計算)
# 推移確率をリスト形式に変換 (BCMP_MVA用)
p = pd.DataFrame(transition_matrix).values.tolist()
N = num_locations  # 拠点数
R = num_customer_classes  # 客クラス数
K_total=100
K = [(K_total + i) // R for i in range(R)]  # 各クラスの系内人数を均等に配分
mu = np.array([service_rates for _ in range(R)])  # サービス率 (R×N の形状)
type_list = np.full(N, 1)  # サービスタイプはFCFSとする
m = num_counters  # 窓口数

bcmp = BCMP_MVA(N=N, R=R, K=K, mu=mu, type_list=type_list, p=p, m=m)
start = time.time()
L = bcmp.getMVA()
elapsed_time = time.time() - start
print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
print('L = \n{0}'.format(L))
bcmp.save_mean_L_to_csv(L, "mean_L.csv")
