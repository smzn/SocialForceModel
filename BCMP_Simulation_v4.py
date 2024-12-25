import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import math
import random
import sys
import time
from mpi4py import MPI
import psutil
import matplotlib.colors as mcolors

class BCMP_Simulation:
    
    def __init__(self, N, R, K, mu, m, type_list, p, theoretical, sim_time, rank, size, comm):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.mu = mu #サービス率 FCFSはクラス別で変えられない
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.p = p
        print('p : {0}'.format(self.p))
        print(self.p.shape)
        self.theoretical = theoretical
        print('Theoretical Values : {0}'.format(self.theoretical.values))
        self.m = m #窓口配列
        print('m : {0}'.format(self.m))
        self.event = [[] for i in range(self.N)] #各拠点で発生したイベント(arrival, departure)を格納
        self.eventclass = [[] for i in range(self.N)] #各拠点でイベント発生時の客クラス番号
        self.eventqueue = [[] for i in range(N)] #各拠点でイベント発生時のqueueの長さ
        self.eventtime = [[] for i in range(N)] #各拠点でイベントが発生した時の時刻
        self.timerate = np.zeros((self.N, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る
        self.timerateclass = np.zeros((self.N, self.R, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る、クラス別
        self.time = sim_time #シミュレーション時間
        if rank == 0:
            #平均算出用
            self.sum_L = np.zeros(self.N) #平均系内人数(結果の和)
            self.sum_Lc = np.zeros((self.N, self.R)) # #平均系内人数(結果の和)(クラス別)
            self.sum_Q = np.zeros(self.N) #平均待ち人数(結果の和)
            self.sum_Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果の和)(クラス別)
        #print(self.p.iloc[0,1])
        self.rank = rank
        self.size = size
        self.comm = comm
        self.process_text = './process/process_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(sum(self.K))+'_Time'+str(self.time)+'.txt'
        self.cpu_list = []
        self.mem_list = []
        self.start = time.time()
        
        
    def getSimulation(self):
        queue = np.zeros(self.N) #各拠点のサービス中を含むqueueの長さ(クラスをまとめたもの)
        queueclass = np.zeros((self.N, self.R)) #各拠点のサービス中を含むqueueの長さ(クラス別)
        classorder = [[] for i in range(self.N)] #拠点に並んでいる順のクラス番号
        window_number = [np.zeros(self.m[i], dtype=int) for i in range(self.N)]  # ノードごとの窓口数で初期化
        window = [np.full(self.m[i], self.R) for i in range(self.N)]  # ノードごとの窓口数で初期化
        service = [np.zeros(self.m[i]) for i in range(self.N)]  # ノードごとの窓口数で初期化
        total_length = np.zeros(self.N) #各拠点の延べ系内人数(クラスをまとめたもの)
        total_lengthclass = np.zeros((self.N, self.R)) #各拠点の延べ人数(クラス別)
        total_waiting = np.zeros(self.N) #延べ待ち人数(クラスをまとめたもの)
        total_waitingclass = np.zeros((self.N, self.R))#延べ待ち人数(クラス別)
        L = np.zeros(self.N) #平均系内人数(結果)
        Lc = np.zeros((self.N, self.R)) #平均系内人数(結果)(クラス別)
        Q = np.zeros(self.N) #平均待ち人数(結果)
        Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果)(クラス別)
        rmse = [] #100単位時間でのrmseの値を格納
        rmse_time = [] #rmseを登録した時間
        regist_time = 50 #rmseの登録時刻
        regist_span = 50 #50単位で登録
        length = [[] for i in range(self.N)] #系内人数の変化率
        L_list = [[] for i in range(self.N)] #各ノードの平均系内人数
        label_list = [] #boxplotのラベルリスト
        for n in range(self.N):
            label_list.append(n)        
        
        elapse = 0
        #Step1 開始時の客の分配 (開始時のノードは拠点番号0)
        initial_node = 0
        for i in range(self.R):
            for j in range(self.K[i]):
                initial_node = random.randrange(self.N)#20220320 最初はランダムにいる拠点を決定
                self.event[initial_node].append("arrival") #イベントを登録
                self.eventclass[initial_node].append(i) #到着客のクラス番号
                self.eventqueue[initial_node].append(queue[initial_node])#イベントが発生した時のqueueの長さ(到着客は含まない)
                self.eventtime[initial_node].append(elapse) #(移動時間0)
                queue[initial_node] += 1 #最初はノード0にn人いるとする
                queueclass[initial_node][i] += 1 #拠点0にクラス別人数を追加
                classorder[initial_node].append(i) #拠点0にクラス番号を追加
                #空いている窓口に客クラスとサービス時間を登録
                if queue[initial_node] <= self.m[initial_node]:
                    window[initial_node][int(queue[initial_node] - 1)] = i #窓口にクラス番号を登録
                    service[initial_node][int(queue[initial_node] - 1)] = self.getExponential(self.mu[initial_node]) #窓口客のサービス時間設定

        #window_numberを設定
        for i in range(self.N):
            num = 1
            for j in range(self.m[i]):
                window_number[i][j] = num
                num += 1
        
        #print('Simulation Start')
        #Step2 シミュレーション開始
        while elapse < self.time:
            #print('経過時間 : {0} / {1}'.format(elapse, self.time))
            mini_service = 100000 #最小のサービス時間
            mini_index = -1 #最小のサービス時間をもつノード
            window_index = -1 #最小のサービス時間の窓口
            classorder_index = 0 #最小のサービス時間が先頭から何番目の客か
           
            #print('Step2.1 次に退去が起こる拠点を検索')
            #Step2.1 次に退去が起こる拠点を検索
            for i in range(self.N): #待ち人数がいる中で最小のサービス時間を持つノードの窓口を算出
                if queue[i] > 0: #待ち人数がいるとき
                    for j in range(self.m[i]):
                        if window[i][j] < self.R: #窓口に客がいるとき(self.Rは空状態)
                            if mini_service > service[i][j]: #最小のサービス時間の更新
                                mini_service = service[i][j]
                                mini_index = i  #最小のサービス時間をもつノード
                                window_index = j #最小のサービス時間の窓口
            #(window_index+1)番目の窓口のindexを取得(classorderに対応)
            for i in range(self.m[mini_index]):
                if window_number[mini_index][i] == window_index+1: #
                    classorder_index = i
            departure_class = classorder[mini_index].pop(classorder_index) #退去する客のクラスを取り出す
            #window_numberをclassorderに合わせて調整
            for i in range(classorder_index, self.m[mini_index]):
                if i+1 == self.m[mini_index]:
                    window_number[mini_index][i] = 0
                else:
                    window_number[mini_index][i] = window_number[mini_index][i+1]
            #departure_Node.append(mini_index)

            #Step2.2 イベント拠点確定後、全ノードの情報更新(サービス時間、延べ人数)
            for i in range(self.N):#ノードiから退去(全拠点で更新)
                total_length[i] += queue[i] * mini_service #ノードでの延べ系内人数
                for r in range(self.R): #クラス別延べ人数更新
                    total_lengthclass[i,r] += queueclass[i,r] * mini_service
                if queue[i] > 0: #系内人数がいる場合(サービス中の客がいるとき)
                    for j in range(self.m[i]):
                        if service[i][j] > 0:
                            service[i][j] -= mini_service #サービス時間を減らす
                    if queue[i] > self.m[i]:
                        total_waiting[i] += ( queue[i] - self.m[i] ) * mini_service #ノードでの延べ待ち人数(queueの長さが窓口数以上か以下かで変わる)
                    #else:
                    #    total_waiting[i] += 0 * mini_service
                    for r in range(R):
                        if queueclass[i,r] > 0: #クラス別延べ待ち人数の更新(windowの各クラスの数をカウント -> カウント分queueclassから引く)
                            c = np.count_nonzero(window[i]==r)
                            total_waitingclass[i,r] += ( queueclass[i,r] - c ) * mini_service
                            #print('クラス別延べ待ち人数window[{0}] : {1} c={2}'.format(i, window[i], c)) 
                elif queue[i] == 0: #いらないかも
                    total_waiting[i] += queue[i] * mini_service
                self.timerate[i, int(queue[i])] += mini_service #人数分布の時間帯を更新
                for r in range(R):
                    self.timerateclass[i, r, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
                    
            #Step2.3 退去を反映
            self.event[mini_index].append("departure") #退去を登録
            self.eventclass[mini_index].append(departure_class) #退去するクラスを登録
            self.eventqueue[mini_index].append(queue[mini_index]) #イベント時の系内人数を登録
            queue[mini_index] -= 1 #ノードの系内人数を減らす
            queueclass[mini_index, departure_class] -= 1 #ノードの系内人数を減らす(クラス別)
            elapse += mini_service #最小のサービス時間分進める
            self.eventtime[mini_index].append(elapse) #経過時間の登録はイベント後
            window[mini_index][window_index] = self.R #窓口を空にする(self.Rは空状態)
            #窓口がすべて埋まっている -> １つ空いた
            if queue[mini_index] > 0: #退去が発生したノードに系内人数が1以上のとき
                for i in range(self.m[mini_index]):
                    #サービス時間が0 かつ 窓口が空 かつ 系内人数が窓口数以上の場合(サービス時間が0の条件は余分かも)
                    if service[mini_index][i] == 0 and window[mini_index][i] == self.R and queue[mini_index] >= self.m[mini_index]:
                        #print('classorder[{0}] : {1}'.format(mini_index, classorder[mini_index]))
                        #print('queue[{0}] = {1} window[{0}][{2}]に追加'.format(mini_index, queue[mini_index], i)) #i=3ならOK
                        window_number[mini_index][int(self.m[mini_index] - 1)] = i+1 #窓口番号を登録
                        window[mini_index][i] = classorder[mini_index][int(self.m[mini_index] - 1)] #窓口にクラスを登録
                        service[mini_index][i] = self.getExponential(self.mu[mini_index]) #退去後まだ待ち人数がある場合、サービス時間設定
                        break

            #Step2.4 退去客の行き先決定
            #推移確率行列が N*R × N*Rになっている。departure_class = 0の時は最初のN×N (0~N-1の要素)を見ればいい
            #departure_class = 1の時は (N~2N-1の要素)、departure_class = 2の時は (2N~3N-1の要素)
            #departure_class = rの時は (N*r~N*(r+1)-1)を見ればいい
            rand = random.random()
            sum_rand = 0
            destination_index = -1 #行き先のノード番号
            pr = np.zeros((self.N, self.N))#今回退去する客クラスの推移確率行列を抜き出す
            for i in range(self.N * departure_class, self.N * (departure_class + 1)):
                for j in range(self.N * departure_class, self.N * (departure_class + 1)):
                    pr[i - self.N * departure_class, j - self.N * departure_class] = self.p.iloc[i,j]
            #print('今回退去する客クラス{0}の推移確率行列'.format(departure_class))
            #print(pr)
            #print(pr.shape)
            for i in range(len(pr)):
                sum_rand += pr[mini_index][i]
                if rand < sum_rand:
                    destination_index = i
                    break
            if destination_index == -1: #これは確率が1になってしまったとき用
                destination_index = len(pr) -1 #一番最後のノードに移動することにする

            self.event[destination_index].append("arrival") #イベント登録
            self.eventclass[destination_index].append(departure_class) #移動する客クラス番号登録
            self.eventqueue[destination_index].append(queue[destination_index]) #イベント時の系内人数を登録(到着客は含まない)
            self.eventtime[destination_index].append(elapse) #(移動時間0)
            queue[destination_index] += 1 #推移先の待ち行列に並ぶ
            queueclass[destination_index][departure_class] += 1 #推移先の待ち行列(クラス別)に登録 
            classorder[destination_index].append(departure_class) #推移先にクラス番号登録
            #推移先で待っている客がいなければサービス時間設定(即時サービス)
            if queue[destination_index] <= self.m[destination_index]: #推移先の系内人数(到着客も含める)が窓口数以下の場合
                for i in range(self.m[destination_index]):
                    if service[destination_index][i] == 0 and window[destination_index][i] == self.R: #サービス時間が0 かつ 窓口が空の場合
                        window_number[destination_index][int(queue[destination_index] - 1)] = i+1 #窓口番号を登録
                        window[destination_index][i] = departure_class #窓口に到着客を登録
                        service[destination_index][i] = self.getExponential(self.mu[destination_index]) #サービス時間設定
                        break
            #arrival_Node.append(destination_index)
            
           
            #Step2.5 RMSEの計算
            if elapse > regist_time:
                rmse_sum = 0
                theoretical_value = self.theoretical.values
                theoretical_row = np.sum(self.theoretical.values, axis=1)
                l = total_length / elapse #今までの時刻での平均系内人数
                lc = total_lengthclass / elapse #今までの時刻での平均系内人数(クラス別)
                for n in range(self.N):
                    for r in range(self.R):
                        rmse_sum += (theoretical_value[n,r] - lc[n,r])**2
                rmse_sum /= self.N * self.R
                rmse_value = math.sqrt(rmse_sum)
                rmse.append(rmse_value)
                rmse_time.append(regist_time)
                regist_time += regist_span
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
                print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value))
                print('Elapse = {0}, Lc = {1}'.format(elapse, lc))
                if self.rank == 0:
                    mem = psutil.virtual_memory()
                    cpu = psutil.cpu_percent(interval=1)
                    with open(self.process_text, 'a') as f:
                        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                        print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value), file=f)
                        print('Elapse = {0}, Memory = {1}GB, CPU = {2}'.format(elapse, mem.used/10**9, cpu), file=f)
                        print('Elapse = {0}, Lc = {1}'.format(elapse, lc), file=f)
                    self.cpu_list.append(cpu)
                    self.mem_list.append(mem.used/10**9)
                #時間経過による系内人数の変動
                for n in range(self.N):
                    length[n].append(queue[n])
                    L_list[n].append(l[n])
                #5000時間までの平均系内人数のboxplot
                if 5000 <= regist_time and regist_time < (5000 + regist_span):
                    plt.figure(figsize=(12,5))
                    plt.boxplot(L_list, tick_labels=label_list)
                    # U を削除し、ファイル名のコンマをアンダースコアに置き換え
                    plt.savefig('./plot/L_box_Time5000_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.png', format='png', dpi=300, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                                  
            
        L = total_length / self.time #平均系内人数
        Lc = total_lengthclass / self.time #平均系内人数(クラス別)
        self.Lc = Lc #描画用にself.Lcを用意
        Q = total_waiting / self.time #平均待ち人数
        Qc = total_waitingclass / self.time #平均待ち人数(クラス別)
        
        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
        print('平均系内人数L : {0}'.format(L))
        print('平均系内人数(クラス別)Lc : {0}'.format(Lc))
        print('平均待ち人数Q : {0}'.format(Q))
        print('平均待ち人数(クラス別)Qc : {0}'.format(Qc))
        if self.rank == 0:
            with open(self.process_text, 'a') as f:
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                print('平均系内人数L : {0}'.format(L), file=f)
                print('平均系内人数(クラス別)Lc : {0}'.format(Lc), file=f)
                print('平均待ち人数Q : {0}'.format(Q), file=f)
                print('平均待ち人数(クラス別)Qc : {0}'.format(Qc), file=f)

        pd.DataFrame(L).to_csv('./csv/L_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.csv')
        pd.DataFrame(Lc).to_csv('./csv/Lc_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.csv')
        pd.DataFrame(Q).to_csv('./csv/Q_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.csv')
        pd.DataFrame(Qc).to_csv('./csv/Qc_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.csv')

        rmse_index = {'time': rmse_time, 'RMSE': rmse}
        df_rmse = pd.DataFrame(rmse_index)
        df_rmse.to_csv('./csv/RMSE_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.csv')

        plt.figure(figsize=(12,5))
        tmp = list(matplotlib.colors.CSS4_COLORS.values())
        colorlist = tmp[:self.N] #先頭からN個
        #系内人数の変動
        for n in range(self.N):
            plt.plot(rmse_time, length[n], '-', lw=0.5, color=colorlist[n], label='node'+str(n))
        plt.legend(fontsize='xx-small', ncol=3, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)
        plt.grid(which='major', axis='y', color='black', alpha=0.5, linestyle='-', linewidth=0.5)
        plt.savefig('./plot/length_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.png', format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        #平均系内人数のboxplot
        plt.boxplot(L_list, tick_labels=label_list) #boxplot(平均系内人数)
        plt.savefig('./plot/box_L_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.png', format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        #各rankのRMSEの最終値を取得
        if self.rank == 0:
            last_rmse = []
            last_rmse.append(rmse[-1])

            for i in range(1, self.size):
                rmse_rank = self.comm.recv(source=i, tag=0)
                last_rmse.append(rmse_rank[-1])
            #print('rank({1}) last_rmse : {0}'.format(last_rmse, self.rank))
        else:
            self.comm.send(rmse, dest=0, tag=0)
            #print('rank = {0} 送信完了'.format(self.rank))
        self.comm.barrier() #プロセス同期
        #RMSEの最終値が最小と最大のものを除いて、平均系内人数の平均を算出
        #各シミュレーション時間におけるRMSEの平均の算出
        if self.rank == 0:
            #平均算出用
            sum_L = np.zeros(self.N) #平均系内人数(結果の和)
            sum_Lc = np.zeros((self.N, self.R)) # #平均系内人数(結果の和)(クラス別)
            sum_Q = np.zeros(self.N) #平均待ち人数(結果の和)
            sum_Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果の和)(クラス別)
            avg_rmse = np.zeros(len(rmse_time)) #RMSEの平均
            #RMSEの総和が最大と最小のrank取得
            max_index = last_rmse.index(max(last_rmse))
            min_index = last_rmse.index(min(last_rmse))
            #print('max : index={0} values={1}'.format(max_index, max(last_rmse)))
            #print('min : index={0} values={1}'.format(min_index, min(last_rmse)))

            plt.figure(figsize=(12,5))
            if 0 == max_index or 0 == min_index: #rank0が最大最小の場合
                plt.plot(rmse_time, rmse, linestyle = 'dotted', color = 'black', alpha = 0.5) #平均に含まれない
            else:
                sum_L += L
                sum_Lc += Lc
                sum_Q += Q
                sum_Qc += Qc
                avg_rmse = np.add(avg_rmse, rmse)
                plt.plot(rmse_time, rmse)
            #rank0以外の処理
            for i in range(1, self.size):                                   
                L_rank = self.comm.recv(source=i, tag=1)
                Lc_rank = self.comm.recv(source=i, tag=2)
                Q_rank = self.comm.recv(source=i, tag=3)
                Qc_rank = self.comm.recv(source=i, tag=4)
                rmse_rank = self.comm.recv(source=i, tag=10)
                time_rank = self.comm.recv(source=i, tag=11)

                if i == max_index or i == min_index:
                    plt.plot(time_rank, rmse_rank, linestyle = 'dotted', color = 'black', alpha = 0.5) #平均に含まれない
                else:
                    sum_L += L_rank
                    sum_Lc += Lc_rank
                    sum_Q += Q_rank
                    sum_Qc += Qc_rank
                    avg_rmse = np.add(avg_rmse, rmse_rank)
                    plt.plot(time_rank, rmse_rank)
            #RMSEの折れ線グラフ
            plt.savefig('./plot/RMSE_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Rank_'+str(self.rank)+'.png', format='png', dpi=300)
            plt.clf()
                    
            #平均の算出 (最大値と最小値は除く)
            if self.size > 2:
                avg_L = sum_L / (self.size - 2)
                avg_Lc = sum_Lc / (self.size - 2)
                avg_Q = sum_Q / (self.size - 2)
                avg_Qc = sum_Qc / (self.size - 2)
                avg_rmse = [n / (self.size - 2) for n in avg_rmse]
            else:
                avg_L = sum_L
                avg_Lc = sum_Lc
                avg_Q = sum_Q
                avg_Qc = sum_Qc
                avg_rmse = avg_rmse  # 必要に応じて初期化や代替処理を実装

            print('平均系内人数avg_L : {0}'.format(avg_L))
            print('平均系内人数(クラス別)avg_Lc : {0}'.format(avg_Lc))
            print('平均待ち人数avg_Q : {0}'.format(avg_Q))
            print('平均待ち人数(クラス別)avg_Qc : {0}'.format(avg_Qc))

            pd.DataFrame(avg_L).to_csv('./csv/avg_L_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Size_'+str(self.size)+'.csv')
            pd.DataFrame(avg_Lc).to_csv('./csv/avg_Lc_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Size_'+str(self.size)+'.csv')
            pd.DataFrame(avg_Q).to_csv('./csv/avg_Q_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Size_'+str(self.size)+'.csv')
            pd.DataFrame(avg_Qc).to_csv('./csv/avg_Qc_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Size_'+str(self.size)+'.csv')

            avg_rmse_index = {'time': rmse_time, 'RMSE': avg_rmse}
            df_avg_rmse = pd.DataFrame(avg_rmse_index)
            df_avg_rmse.to_csv('./csv/avg_RMSE_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Size_'+str(self.size)+'.csv')

            plt.plot(rmse_time, avg_rmse)
            plt.savefig('./plot/avg_RMSE_N_'+str(self.N)+'_R_'+str(self.R)+'_K_'+str(self.K)+'_Time_'+str(self.time)+'_Size_'+str(self.size)+'.png', format='png', dpi=300)
            plt.clf()
            plt.close()
        
        else:
            self.comm.send(L, dest=0, tag=1)
            self.comm.send(Lc, dest=0, tag=2)
            self.comm.send(Q, dest=0, tag=3)
            self.comm.send(Qc, dest=0, tag=4)
            self.comm.send(rmse, dest=0, tag=10)
            self.comm.send(rmse_time, dest=0, tag=11)
            #print('rank = {0} 送信完了'.format(self.rank))
        self.comm.barrier() #プロセス同期
        

    def plot_mean_L_with_weights(self, num_counters, locations, region_size=1000, filename="./plot/mean_L_with_weights_simulation.png"):
        """拠点の位置と平均系内人数の2D等高線表示"""
        fig, ax = plt.subplots(figsize=(10, 8))
        contour_cmap = "hot"

        # 平均系内人数の等高線を計算
        x = np.linspace(0, region_size, 300)
        y = np.linspace(0, region_size, 300)
        xx, yy = np.meshgrid(x, y)
        mean_L_grid = np.zeros_like(xx)

        # 各拠点の影響を平均系内人数として加算
        for i, (loc_x, loc_y) in enumerate(locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            influence = np.sum(self.Lc[i, :]) / (1 + distances)  # 距離に反比例した影響
            mean_L_grid += influence

        # 等高線の描画
        contour = ax.contour(xx, yy, mean_L_grid, levels=150, cmap=contour_cmap, linewidths=1.0, alpha=0.5)
        ax.clabel(contour, inline=True, fontsize=10, fmt="%1.1f")

        # 拠点のプロット
        scatter_shapes = ['o', 's', '^', 'D', '*']
        unique_counters = sorted(set(num_counters))
        scatter_cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(np.sum(self.Lc, axis=1)), vmax=max(np.sum(self.Lc, axis=1)))

        for i, (x, y) in enumerate(locations):
            shape_idx = unique_counters.index(num_counters[i]) % len(scatter_shapes)
            ax.scatter(x, y, c=[np.sum(self.Lc[i, :])], cmap=scatter_cmap, s=100, norm=norm, zorder=5, marker=scatter_shapes[shape_idx])
            ax.text(x + 10, y + 15, str(i), fontsize=9, ha='right', zorder=6)

        # カラーバーの追加
        sm = plt.cm.ScalarMappable(norm=norm, cmap=scatter_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label="Average Customers in System", orientation='horizontal', pad=0.1)

        # 凡例の追加
        legend_handles = [plt.Line2D([0], [0], marker=scatter_shapes[i % len(scatter_shapes)], color='w', markerfacecolor='gray', markersize=10, label=f'{counter} Counters') for i, counter in enumerate(unique_counters)]
        ax.legend(handles=legend_handles, title="Number of Counters", loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=10, title_fontsize=10)

        # レイアウト調整
        ax.set_xlim(0, region_size)
        ax.set_ylim(0, region_size)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Locations with Mean Number of Customers in System")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def plot_mean_L_with_weights_3d(self, num_counters, locations, region_size=1000, filename="./plot/mean_L_with_weights_3d_simulation.png"):
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
            influence = np.sum(self.Lc[i, :]) / (1 + distances)
            mean_L_grid += influence

        # 3Dサーフェスの描画
        surf = ax.plot_surface(xx, yy, mean_L_grid, cmap="viridis", edgecolor='k', alpha=0.2)

        # 拠点のプロット
        scatter_shapes = ['o', 's', '^', 'D', '*']
        unique_counters = sorted(set(num_counters))
        scatter_cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(np.sum(self.Lc, axis=1)), vmax=max(np.sum(self.Lc, axis=1)))

        for i, (x, y) in enumerate(locations):
            shape_idx = unique_counters.index(num_counters[i]) % len(scatter_shapes)
            ax.scatter(x, y, 0, c=[np.sum(self.Lc[i, :])], cmap=scatter_cmap, s=100, norm=norm, zorder=5, marker=scatter_shapes[shape_idx])
            ax.text(x, y, 0, str(i), fontsize=9, ha='center', zorder=6)

        # カラーバーの追加
        sm = plt.cm.ScalarMappable(norm=norm, cmap=scatter_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.5, pad=0.1, label="Average Customers in System")

        # 凡例の追加
        legend_handles = [plt.Line2D([0], [0], marker=scatter_shapes[i % len(scatter_shapes)], color='w', markerfacecolor='gray', markersize=10, label=f'{counter} Counters') for i, counter in enumerate(unique_counters)]
        ax.legend(handles=legend_handles, title="Number of Counters", loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0., fontsize=8, title_fontsize=10)

        # ラベルとタイトルの設定
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Mean Number of Customers")
        ax.set_title("Locations with Mean Number of Customers in System (3D)")

        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def getExponential(self, param):
        return - math.log(1 - random.random()) / param
    
if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    sim_time = int(sys.argv[4])
    p_file = sys.argv[5]
    theoretical_file = sys.argv[6]
    location_file = sys.argv[7]
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    
    #BCMP_SFM用設定
    p = pd.read_csv(p_file, header=None)
    theoretical = pd.read_csv(theoretical_file, index_col=0, header=0) 
    location_data = pd.read_csv(location_file, header=0)
    mu = location_data['Service Rate'].values  # サービス率
    m = location_data['Num Counters'].values  # 窓口数
    locations = location_data[['X Coordinate', 'Y Coordinate']].values.tolist()

    type_list = np.full(N, 1) #サービスタイプはFCFS (N, R)
    bcmp = BCMP_Simulation(N, R, K, mu, m, type_list, p, theoretical, sim_time, rank, size, comm) 
    start = time.time()
    bcmp.getSimulation()
    bcmp.plot_mean_L_with_weights(m, locations)
    bcmp.plot_mean_L_with_weights_3d(m, locations)
    elapsed_time = time.time() - start
    print ("rank : {1}, calclation_time:{0}".format(elapsed_time, rank) + "[sec]")
    
    #BCMP_SFM
    #mpiexec -n 2 python3 BCMP_Simulation_v4.py 33 2 100 100000 transition_matrix.csv mean_L.csv locations_and_weights.csv > result_33_2_100_100000.txt

    #並列計算用
    #mpiexec -n 8 python BCMP_Simulation_v3.py 33 2 500 2 100000 tp/transition_probability_N33_R2_K500_Core8.csv tp/L_N33_R2_K500_Core8.csv tp/node_N33_R2_K500_Core8.csv > result_33_2_500_100000.txt
    #mpiexec -n 10 python BCMP_Simulation_v3.py 33 2 500 2 100000 tp/transition_probability_N33_R2_K500_Core1.csv tp/L_N33_R2_K500_Core1.csv tp/node_N33_R2_K500_Core1.csv > result_33_2_500_100000.txt
    #mpiexec -n 8 python BCMP_Simulation_v3.py 33 2 500 2 5500 tp/transition_probability_N33_R2_K500_Core1.csv tp/L_N33_R2_K500_Core1.csv tp/node_N33_R2_K500_Core1.csv > result_33_2_500_5500.txt
    