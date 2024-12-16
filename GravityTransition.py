import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GravityTransition:
    def __init__(self, num_locations, num_customer_classes, beta=2, region_size=1000):
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

    def plot_locations(self, filename="locations.png"):
        """拠点の位置をプロット"""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.locations[:, 0], self.locations[:, 1], c='blue', s=100, label="Locations")
        for i, (x, y) in enumerate(self.locations):
            plt.text(x, y, str(i), fontsize=9, ha='right')
        plt.xlim(0, self.region_size)
        plt.ylim(0, self.region_size)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Locations of Facilities")
        plt.legend()
        plt.savefig(filename)
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
            "Y Coordinate": self.locations[:, 1]
        }

        for c in range(self.num_customer_classes):
            data[f"Class {c+1} Popularity"] = self.weights[c]

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)


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

    # 拠点位置をプロットし保存
    model.plot_locations()

    # 人気度をプロットし保存
    model.plot_weights()

    # 拠点の位置と人気度を保存
    model.save_locations_and_weights("locations_and_weights.csv")
