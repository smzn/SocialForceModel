import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

class GravityTransition:
    def __init__(self, num_locations, num_customer_classes, beta=2, region_size=1000, counter_range=(1, 3), service_time_range=(5/60, 15/60)):
        """
        初期化メソッド

        Parameters:
        - num_locations: 拠点数
        - num_customer_classes: 客クラス数
        - beta: 距離減衰係数
        - region_size: 領域サイズ（例: 1000）
        """
        self.num_locations = num_locations
        self.num_customer_classes = num_customer_classes
        self.beta = beta
        self.region_size = region_size

        # 拠点位置をランダムに設定（分散を考慮）
        self.locations = self._generate_locations()

        # 客クラスごとの人気度をランダムに設定
        self.weights = self._generate_weights()

        # 距離行列を計算
        self.distance_matrix = self._calculate_distance_matrix()

        # 窓口数とサービス率の初期化
        self.num_counters = self._generate_num_counters(counter_range)
        self.service_rates = self._calculate_service_rates(service_time_range)

    def _generate_locations(self):
        """拠点位置をランダムに生成（距離の分散を確保）"""
        locations = []
        while len(locations) < self.num_locations:
            x, y = np.random.uniform(0, self.region_size, size=2)
            # 既存の点と十分離れている場合のみ追加
            if all(np.linalg.norm(np.array([x, y]) - np.array(loc)) > self.region_size / 10 for loc in locations):
                locations.append((x, y))
        return np.array(locations)

    def _generate_weights(self):
        """客クラスごとの人気度を生成"""
        return np.random.randint(1, 100, size=(self.num_customer_classes, self.num_locations))

    def _calculate_distance_matrix(self):
        """距離行列を計算"""
        num_locations = self.num_locations
        distance_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(self.locations[i] - self.locations[j])
        return distance_matrix

    def _calculate_transition_matrix(self, weights):
        """重力モデルを用いて推移確率行列を計算"""
        weights_matrix = np.outer(weights, weights)
        distance_effect = np.power(self.distance_matrix, self.beta)
        gravity_matrix = weights_matrix / distance_effect
        np.fill_diagonal(gravity_matrix, 0)  # 自己移動はゼロにする
        row_sums = gravity_matrix.sum(axis=1, keepdims=True)
        return gravity_matrix / row_sums

    def create_transition_matrix(self):
        """推移確率行列を作成"""
        num_classes = self.num_customer_classes
        num_locations = self.num_locations

        # 全体推移確率行列を初期化
        full_transition_matrix = np.zeros((num_classes * num_locations, num_classes * num_locations))

        for c in range(num_classes):
            # クラス c の重みを基に推移確率を計算
            transition_matrix = self._calculate_transition_matrix(self.weights[c])
            # 対角ブロックに設定
            start = c * num_locations
            end = (c + 1) * num_locations
            full_transition_matrix[start:end, start:end] = transition_matrix

        return full_transition_matrix

    def save_transition_matrix(self, filename="transition_matrix.csv"):
        """推移確率行列を保存"""
        transition_matrix = self.create_transition_matrix()
        pd.DataFrame(transition_matrix).to_csv(filename, index=False, header=False)
        return transition_matrix

    def plot_locations(self, num_counters ,filename="locations.png"):
        """拠点の位置をプロット"""
        plt.figure(figsize=(8, 8))
        # カラーマップの定義 (窓口数の範囲に対応)
        unique_counters = sorted(set(num_counters))  # 窓口数のユニーク値を取得
        cmap = cm.get_cmap('viridis', len(unique_counters))  # 色をユニーク数に合わせて分割
        norm = mcolors.Normalize(vmin=min(unique_counters), vmax=max(unique_counters))  # 正規化
        colors = [cmap(norm(c)) for c in unique_counters]  # 各窓口数に対応する色

        # 拠点のプロット (色は窓口数に対応)
        scatter = plt.scatter(self.locations[:, 0], self.locations[:, 1], 
                            c=num_counters, cmap=cmap, s=100, norm=norm)
        
        for i, (x, y) in enumerate(self.locations):
            plt.text(x + 10, y + 15, str(i), fontsize=9, ha='right')

        # 凡例の作成
        legend_patches = [mpatches.Patch(color=colors[i], label=f"Counter: {unique_counters[i]}") 
                        for i in range(len(unique_counters))]
        plt.legend(handles=legend_patches, title="Number of Counters", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

        plt.xlim(0, self.region_size)
        plt.ylim(0, self.region_size)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Locations of Facilities")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def plot_weights(self, filename="weights.png"):
        """人気度を棒グラフで表示"""
        indices = np.arange(self.num_locations)
        bottom = np.zeros(self.num_locations)  # 積み上げの開始位置

        plt.figure(figsize=(10, 6))
        for c in range(self.num_customer_classes):
            plt.bar(indices, self.weights[c], bottom=bottom, label=f"Class {c+1}")
            bottom += self.weights[c]  # 次のクラスの棒グラフの開始位置を更新

        plt.xlabel("Location Index")
        plt.ylabel("Popularity (Weight)")
        plt.title("Popularity by Location and Customer Class (Stacked)")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def plot_locations_with_weights(self, num_counters, filename="locations_with_weights.png"):
        """拠点の位置をプロットし、人気度の等高線を追加"""
        plt.figure(figsize=(8, 8))  # 正方形の比率に変更

        # カラーマップの定義 (白黒用)
        contour_cmap = "Greys"  # 白黒のカラーマップ

        # 人気度の等高線を計算
        x = np.linspace(0, self.region_size, 300)  # 格子点をさらに増やす
        y = np.linspace(0, self.region_size, 300)
        xx, yy = np.meshgrid(x, y)
        popularity_grid = np.zeros_like(xx)

        # 各拠点の影響を人気度として加算
        for i, (loc_x, loc_y) in enumerate(self.locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            influence = self.weights.sum(axis=0)[i] / (1 + distances)  # 距離に反比例した影響
            popularity_grid += influence

        # 等高線の描画 (線をさらに濃く)
        contour = plt.contour(xx, yy, popularity_grid, levels=150, cmap=contour_cmap, linewidths=2.5, alpha=1.0)

        # 拠点のプロット (色は窓口数に対応)
        unique_counters = sorted(set(num_counters))  # 窓口数のユニーク値を取得
        scatter_cmap = plt.cm.viridis  # カラーマップを定義
        norm = mcolors.Normalize(vmin=min(unique_counters), vmax=max(unique_counters))

        for i, (x, y) in enumerate(self.locations):
            plt.scatter(x, y, c=[num_counters[i]], cmap=scatter_cmap, s=100, norm=norm, zorder=5)  # マーカーを小さくして色分け
            plt.text(x + 10, y + 15, str(i), fontsize=9, ha='right', zorder=6)

        # 窓口数の凡例を対応で表示
        legend_patches = [mpatches.Patch(color=scatter_cmap(norm(c)), label=f"Counter: {c}") 
                           for c in unique_counters]
        plt.legend(handles=legend_patches, title="Number of Counters", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

        plt.xlim(0, self.region_size)
        plt.ylim(0, self.region_size)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Locations of Facilities with Popularity Contours")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    def plot_3d_locations_with_weights(self, num_counters, filename="3d_locations_with_weights.png"):
        """拠点の位置を3次元で山の高さとして表示"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 3Dグリッドの計算
        x = np.linspace(0, self.region_size, 300)  # 格子点をさらに増やす
        y = np.linspace(0, self.region_size, 300)
        xx, yy = np.meshgrid(x, y)
        popularity_grid = np.zeros_like(xx)

        # 各拠点の影響を人気度として加算
        for i, (loc_x, loc_y) in enumerate(self.locations):
            distances = np.sqrt((xx - loc_x)**2 + (yy - loc_y)**2)
            influence = self.weights.sum(axis=0)[i] / (1 + distances)  # 距離に反比例した影響
            popularity_grid += influence

        # 3Dプロットの描画
        ax.plot_surface(xx, yy, popularity_grid, cmap="viridis", edgecolor='k', alpha=0.3)

        # 拠点のプロット (色は窓口数に対応)
        unique_counters = sorted(set(num_counters))  # 窓口数のユニーク値を取得
        scatter_cmap = plt.cm.viridis  # カラーマップを定義
        norm = mcolors.Normalize(vmin=min(unique_counters), vmax=max(unique_counters))

        for i, (x, y) in enumerate(self.locations):
            ax.scatter(x, y, 0, c=[num_counters[i]], cmap=scatter_cmap, s=50, norm=norm, edgecolor="black", zorder=5)
            ax.text(x, y, 0, str(i), fontsize=8, ha='center', zorder=6)

        # ラベルとタイトル
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Popularity Influence")
        ax.set_title("3D Locations of Facilities with Popularity Heights")

        # 凡例の追加
        legend_patches = [mpatches.Patch(color=scatter_cmap(norm(c)), label=f"Counter: {c}")
                           for c in unique_counters]
        fig.legend(handles=legend_patches, title="Number of Counters", bbox_to_anchor=(1, 0.8), loc="upper left", borderaxespad=0.)

        # ファイル保存
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


    def save_locations_and_weights(self, filename="locations_and_weights.csv"):
        """
        拠点の場所と客クラスごとの人気度をCSVに保存

        Parameters:
        - filename: 保存するファイル名
        """
        # 拠点番号、位置(x, y)、各クラスの人気度をまとめる
        data = {
            "Location ID": np.arange(self.num_locations),
            "X Coordinate": self.locations[:, 0],
            "Y Coordinate": self.locations[:, 1],
            "Num Counters": self.num_counters,
            "Service Rate": self.service_rates
        }

        for c in range(self.num_customer_classes):
            data[f"Class {c+1} Popularity"] = self.weights[c]

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def _generate_num_counters(self, counter_range):
        """窓口数 (1~3) をランダムに生成"""
        return np.random.randint(counter_range[0], counter_range[1]+1, size=self.num_locations)

    def _calculate_service_rates(self, service_time_range):
        """平均サービス時間からサービス率を計算"""
        avg_service_times = np.random.uniform(service_time_range[0], service_time_range[1], size=self.num_locations)  # 5分～15分
        return 1 / avg_service_times  # サービス率 = 1 / 平均サービス時間
    
    def get_counters_and_service_rates(self):
        """
        窓口数とサービス率をリストとして返す

        Returns:
        - num_counters: 窓口数リスト
        - service_rates: サービス率リスト
        """
        return self.num_counters, self.service_rates


# 動作確認
if __name__ == "__main__":
    # インスタンス作成
    model = GravityTransition(num_locations=33, num_customer_classes=2)

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

    #拠点の位置をプロットし、人気度の等高線を追加
    model.plot_locations_with_weights(num_counters)

    #拠点の位置を3次元で山の高さとして表示
    model.plot_3d_locations_with_weights(num_counters)


