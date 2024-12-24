import numpy as np
from numpy.linalg import solve
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class BCMP_MVA:
    
    def __init__(self, N, R, K, mu, type_list, p, m):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.p = p
        self.alpha = self.getArrival(self.p)
        self.mu = mu #サービス率 (N×R)
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.m = m 
        self.combi_list = []
        self.combi_list = self.getCombiList2([], self.K, self.R, 0, self.combi_list) #K, Rのときの組み合わせ 
        self.km = (np.max(self.K)+1)**self.R #[2,1]なので3進法で考える(3^2=9状態) #[0,0]->0, [0,1]->1, [1,0]->3, [1,1]->4, [2,0]->6, [2,1]->7
        self.L = np.zeros((self.N, self.R, self.km), dtype= float) #平均形内人数 
        self.T = np.zeros((self.N, self.R, self.km), dtype= float) #平均系内時間
        self.lmd = np.zeros((self.R, self.km), dtype= float) #各クラスのスループット
        self.Pi = np.zeros((self.N, np.max(self.m), self.km), dtype= float)
        
        
    def getMVA(self):
        #decimal.getcontext().prec = 10
        for idx, val in enumerate(self.combi_list):
            if idx == 0:
                continue
            #Tの更新
            k_state = self.getState(val) #kの状態を10進数に変換
            #print('Index : {0}, k = {1}, state = {2}'.format(idx, val, k_state))
            
            for n in range(self.N): #P336 (8.43)
                for r in range(self.R):
                    if self.type_list[n] == 3:
                        self.T[n, r, k_state] = 1 / self.mu[r,n]
                    else:
                        r1 = np.zeros(self.R) #K-1rを計算
                        r1[r] = 1 #対象クラスのみ1
                        k1v = val - r1 #ベクトルの引き算
                        #print('n = {0}, r = {1}, k1v = {2}, k = {3}'.format(n,r,k1v, val))

                        if np.min(k1v) < 0: #k-r1で負になる要素がある場合
                            continue

                        sum_l = 0
                        for i in range(self.R):#k-1rを状態に変換
                            if np.min(k1v) >= 0: #全ての状態が0以上のとき(一応チェック)
                                kn = int(self.getState(k1v))
                                sum_l += self.L[n, i, int(kn)] #knを整数型に変換
                        if self.m[n] == 1: #P336 (8.43) Type-1,2,4 (m_i=1)
                            #print('n = {0}, r = {1}, k_state = {2}'.format(n,r,k_state))
                            self.T[n, r, k_state] = 1 / self.mu[r,n] * (1 + sum_l)
                        if self.m[n] > 1:
                            sum_pi = 0
                            for _j in range(self.m[n]-2+1):
                                pi8_45 = self.getPi(n, _j, val, r1)
                                self.Pi[n][_j][kn] = pi8_45
                                #if pi8_45 < 0:
                                #    with open('sample_Pi.txt', 'a') as f:
                                #        print('Pi[{0}][{1}][{2}] => {3}'.format(n, _j, k1v, pi8_45), file=f)
                                sum_pi += (self.m[n] - _j - 1) * pi8_45
                            
                            self.T[n, r, k_state] = 1 / (self.m[n] * self.mu[r,n]) * (1 + sum_l + sum_pi)
                        #if self.m[n] == 0:
                        #    self.T[n, r, k_state] = Decimal('1') / Decimal(str(self.mu[r,n]))
            #print('T = {0}'.format(T))

            #λの更新
            for r in range(self.R):
                sum = 0
                for n in range(self.N):
                    sum += self.alpha[r][n] * self.T[n,r,k_state]
                if sum == 0:
                    continue
                if sum > 0:
                    self.lmd[r,k_state] = val[r] / sum
                #print('r = {0}, k = {1},lambda = {2}'.format(r, val, lmd[r,k_state]))
            
            
            #Lの更新
            for n in range(self.N):#P336 (8.47)
                for r in range(self.R):
                    self.L[n,r,k_state] = self.lmd[r,k_state] * self.T[n,r,k_state] * self.alpha[r][n]
                    #print('n = {0}, r = {1}, k = {2}, L = {3}'.format(n,r,val,L[n,r,k_state]))
        
        #平均系内人数最終結果
        last = self.getState(self.combi_list[-1]) #combi_listの最終値が最終結果の人数
        return self.L[:,:,last]
    
    def getPi(self, n, j, k, kr):
        kkr = k - kr #指定クラスを1引いたもの
        state_number = int(self.getState(kkr))
        if min(kkr) < 0:
            return 0
        if j == 0 and sum(kkr) == 0: #Initializationより
            return 1
        if j > 0 and sum(kkr) == 0: #Initializationより
            return 0
        if j == 0 and sum(kkr) > 0: #(8.45)
            sum_emlam = 0
            for _r in range(self.R):
                sum_emlam += self.alpha[_r][n] / self.mu[_r][n] * self.lmd[_r][state_number]
            sum_pi = 0
            for _j in range(1, self.m[n]):
                pi8_44 = self.getPi8_44(n, _j, kkr, state_number) #このgetPiは人数を減らさない
                self.Pi[n][_j][state_number] = pi8_44
                sum_pi += (self.m[n] - _j) * pi8_44 #このgetPiは人数を減らさない

            pi = 1 - 1 / self.m[n] * (sum_emlam + sum_pi)
            if pi < 0:
                pi = 0
            return pi
            #return 1 - 1 / self.m[n] * (sum_emlam + sum_pi) #Pi[n][0][idx*self.R + r]
        if j > 0 and sum(kkr) > 0: #(8.44)
            #if self.Pi[n][j][state_number] == 0: #必要ないかも
            #    self.Pi[n][j][state_number] = self.getPi8_44(n, j, kkr, state_number)
            #return self.getPi8_44(n, j, kkr, state_number)
            return self.Pi[n][j][state_number]
            
    def getPi8_44(self, n, j, k, state_number): #(8.45)から(8.44)を呼び出すときだけ利用 (人数を減らさず呼び出し)
        #lmdはそのまま、Piは人数を減らす
        sum_val = 0
        for _r in range(self.R):
            kr = np.zeros(self.R)
            kr[_r] = 1
            kr_state_number = int(self.getState_kr(k, kr))
            if kr_state_number < 0:
                continue
            else:
                #ここで呼び出すgetPiのkrを変更しないといけない->修正、値があった
                sum_val += self.alpha[_r][n] / self.mu[_r][n] * self.lmd[_r][state_number] * self.Pi[n][j-1][kr_state_number]
                
        return 1 / j * (sum_val) #Pi[n][j][idx*self.R + r]
            
    def getState(self, k):#k=[k1,k2,...]を引数としたときにn進数を返す(R = len(K))
        k_state = 0
        for i in range(self.R): #Lを求めるときの、kの状態を求める(この例では3進数)
            k_state += k[i]*((np.max(self.K)+1)**(self.R-1-i))
        return k_state
    
    def getState_kr(self, k, kr):#Piの1つ前の状態
        kr_state = 0
        kkr = k - kr
        if np.min(kkr) < 0:
            return -1
        else:
            for i in range(self.R):
                #state += k[i] * (max(K) + 1)**int(len(k)-1-i)
                kr_state += kkr[i]*((np.max(self.K)+1)**(self.R-1-i))
            return kr_state

    def getArrival(self, p):#マルチクラスの推移確率からクラス毎の到着率を計算する
        p = np.array(p) #リストからnumpy配列に変換(やりやすいので)
        alpha = np.zeros((self.R, self.N))
        for r in range(self.R): #マルチクラス毎取り出して到着率を求める
            alpha[r] = self.getCloseTraffic(p[r * self.N : (r + 1) * self.N, r * self.N : (r + 1) * self.N])
        return alpha
    
    #閉鎖型ネットワークのノード到着率αを求める
    #https://mizunolab.sist.ac.jp/2019/09/bcmp-network1.html
    def getCloseTraffic(self, p):
        e = np.eye(len(p)-1) #次元を1つ小さくする
        pe = p[1:len(p), 1:len(p)].T - e #行と列を指定して次元を小さくする
        lmd = p[0, 1:len(p)] #0行1列からの値を右辺に用いる
        try:
            slv = solve(pe, lmd * (-1)) #2021/09/28 ここで逆行列がないとエラーが出る
        except np.linalg.LinAlgError as err: #2021/09/29 Singular Matrixが出た時は、対角成分に小さい値を足すことで対応 https://www.kuroshum.com/entry/2018/12/28/python%E3%81%A7singular_matrix%E3%81%8C%E8%B5%B7%E3%81%8D%E3%82%8B%E7%90%86%E7%94%B1
            print('Singular Matrix')
            pe += e * 0.00001 
            slv = solve(pe, lmd * (-1)) 
        #lmd *= -1
        #slv = np.linalg.pinv(pe) * lmd #疑似逆行列で求める
        alpha = np.insert(slv, 0, 1.0) #α1=1を追加
        return alpha    

    def getCombiList2(self, combi, K, R, idx, Klist):
        if len(combi) == R:
            #print(combi)
            Klist.append(combi.copy())
            #print(Klist)
            return Klist
        for v in range(K[idx]+1):
            combi.append(v)
            Klist = self.getCombiList2(combi, K, R, idx+1, Klist)
            combi.pop()
        return Klist    
    
    def save_mean_L_to_csv(self, L, filepath, graph_filepath="mean_L_stacked_bar.png"):
        """
        クラス別の平均系内人数をCSVファイルに保存します。

        Args:
            L (np.ndarray): 平均系内人数の結果 (N x R)。
            filepath (str): 保存先のCSVファイルパス。
        """
        # データをDataFrameに変換
        columns = [f"Class{i}" for i in range(self.R)]
        df = pd.DataFrame(L, columns=columns)
        df.index.name = "拠点ID"  # インデックスに拠点IDを追加
        
        # CSVとして保存
        df.to_csv(filepath, encoding='utf-8-sig')
        print(f"平均系内人数を {filepath} に保存しました。")

        # 積み上げ棒グラフの描画
        df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap="tab20")
        plt.title("Average Number of Customers in the System per Node (By Class)")
        plt.xlabel("Node ID")
        plt.ylabel("Average Number of Customers")
        plt.legend(title="Class")
        plt.tight_layout()  # レイアウト調整
        plt.savefig(graph_filepath)
        print(f"平均系内人数の積み上げグラフを {graph_filepath} に保存しました。")

    def plot_mean_L_with_weights(self, L, num_counters, locations, region_size=1000, filename="mean_L_with_weights.png"):
        """拠点の位置と平均系内人数の2D等高線表示"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # カラーマップの定義 (白黒用)
        #contour_cmap = "Greys"
        # カラーマップの定義 (濃淡をより強調するカラーマップを選択)
        contour_cmap = "hot"

        # 平均系内人数の等高線を計算
        x = np.linspace(0, region_size, 300)
        y = np.linspace(0, region_size, 300)
        xx, yy = np.meshgrid(x, y)
        mean_L_grid = np.zeros_like(xx)

        # 各拠点の影響を平均系内人数として加算
        for i, (loc_x, loc_y) in enumerate(locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            influence = np.sum(L[i, :]) / (1 + distances)  # 距離に反比例した影響
            mean_L_grid += influence

        # 等高線の描画
        contour = ax.contour(xx, yy, mean_L_grid, levels=150, cmap=contour_cmap, linewidths=1.0, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=10, fmt="%1.1f")  # ラベルを追加して濃淡を明確化

        # 拠点のプロット (形状は窓口数に対応、色は平均系内人数に対応)
        scatter_shapes = ['o', 's', '^', 'D', '*']  # 窓口数に対応する形状
        unique_counters = sorted(set(num_counters))
        scatter_cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(np.sum(L, axis=1)), vmax=max(np.sum(L, axis=1)))

        for i, (x, y) in enumerate(locations):
            shape_idx = unique_counters.index(num_counters[i]) % len(scatter_shapes)
            ax.scatter(x, y, c=[np.sum(L[i, :])], cmap=scatter_cmap, s=100, norm=norm, zorder=5, marker=scatter_shapes[shape_idx])
            ax.text(x + 10, y + 15, str(i), fontsize=9, ha='right', zorder=6)

        # カラーバーの追加 (修正箇所)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=scatter_cmap)
        sm.set_array([])  # mappable に空のデータを設定
        cbar = fig.colorbar(sm, ax=ax, label="Average Customers in System", orientation='horizontal', pad=0.1)

        # 凡例の追加 (窓口数の形状)
        legend_handles = [plt.Line2D([0], [0], marker=scatter_shapes[i % len(scatter_shapes)], color='w', markerfacecolor='gray', markersize=10, label=f'{counter} Counters') for i, counter in enumerate(unique_counters)]
        legend = ax.legend(handles=legend_handles, title="Number of Counters", loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=10, title_fontsize=10)

        # 正方形のアスペクト比を維持
        #ax.set_aspect('equal', adjustable='datalim')

        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Locations with Mean Number of Customers in System")

        # レイアウト調整 (カラーバーと凡例のために余白を確保)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def plot_mean_L_with_weights_3d(self, L, num_counters, locations, region_size=1000, filename="mean_L_with_weights_3d.png"):
        """拠点の位置と平均系内人数の3D等高線表示"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 平均系内人数の3Dグリッドを計算
        x = np.linspace(0, region_size, 300)
        y = np.linspace(0, region_size, 300)
        xx, yy = np.meshgrid(x, y)
        mean_L_grid = np.zeros_like(xx)

        # 各拠点の影響を平均系内人数として加算
        for i, (loc_x, loc_y) in enumerate(locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            influence = np.sum(L[i, :]) / (1 + distances)  # 距離に反比例した影響
            mean_L_grid += influence

        # 3Dサーフェスの描画
        surf = ax.plot_surface(xx, yy, mean_L_grid, cmap="viridis", edgecolor='k', alpha=0.2)

        # 拠点のプロット (形状は窓口数に対応、色は平均系内人数に対応)
        scatter_shapes = ['o', 's', '^', 'D', '*']  # 窓口数に対応する形状
        unique_counters = sorted(set(num_counters))
        scatter_cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(np.sum(L, axis=1)), vmax=max(np.sum(L, axis=1)))

        for i, (x, y) in enumerate(locations):
            shape_idx = unique_counters.index(num_counters[i]) % len(scatter_shapes)
            ax.scatter(x, y, 0, c=[np.sum(L[i, :])], cmap=scatter_cmap, s=100, norm=norm, zorder=5, marker=scatter_shapes[shape_idx])
            ax.text(x, y, 0, str(i), fontsize=9, ha='center', zorder=6)

        # カラーバーの追加
        sm = plt.cm.ScalarMappable(norm=norm, cmap=scatter_cmap)
        sm.set_array([])  # mappable に空のデータを設定
        #cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="Average Customers in System")
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.5, pad=0.1, label="Average Customers in System")

        # 窓口数の凡例を追加 (右上領域外)
        legend_handles = [plt.Line2D([0], [0], marker=scatter_shapes[i % len(scatter_shapes)], color='w', markerfacecolor='gray', markersize=10, label=f'{counter} Counters') for i, counter in enumerate(unique_counters)]
        ax.legend(handles=legend_handles, title="Number of Counters", loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=8, title_fontsize=10)


        # ラベルとタイトルの設定
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Mean Number of Customers")
        ax.set_title("Locations with Mean Number of Customers in System (3D)")

        # レイアウト調整と保存
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    
    #推移確率行列に合わせる
    N = 33 #33
    R = 2
    K_total = 100 
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full((R, N), 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    m = np.full(N, 2) #窓口数は一律
    #p = pd.read_csv('transition_probability_N33_R2_K100_Core8.csv', header=None).values.tolist()
    p = pd.read_csv('./transition_probability_N33_R2_K100_Core8.csv', index_col=0, header=0).values.tolist()

    bcmp = BCMP_MVA(N, R, K, mu, type_list, p, m)
    start = time.time()
    L = bcmp.getMVA()
    elapsed_time = time.time() - start
    print ("calclation_time:{0}".format(elapsed_time) + "[sec]")
    print('L = \n{0}'.format(L))

    # 結果を保存
    output_filepath = "mean_L.csv"  # 保存するCSVファイルのパス
    bcmp.save_mean_L_to_csv(L, output_filepath)

