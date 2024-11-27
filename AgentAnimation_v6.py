import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle
import ast  # 安全に文字列を評価するために使用

class AgentAnimation:
    def __init__(self, file_path):
        # CSVファイルを取り込んでデータフレームに読み込む
        self.df = pd.read_csv(csv_file_path)
        # エクセルファイルを取り込んでデータフレームに読み込む
        #self.df = pd.read_excel(file_path, sheet_name='シミュレーション履歴と力')
        # 目的地情報を読み込む
        self.df_destinations = pd.read_excel(file_path, sheet_name='目的地情報')
        # 障害物情報を読み込む
        self.df_obstacles = pd.read_excel(file_path, sheet_name='障害物情報')
        # 障害物情報を確認
        print(self.df_obstacles.head())
        # 障害物データの構造化
        self.obstacles = self.process_obstacle_data()
        # 文字列として読み込まれた座標データを適切な形式に変換
        #self.df_obstacles['Coordinates'] = self.df_obstacles['Coordinates'].apply(self.parse_coordinates)

        # エージェントIDをユニークな値として取得
        self.agent_ids = self.df['エージェントID'].unique().tolist()

        # 時間、目的地、状態ごとの人数を集計
        self.df_grouped = self.df.groupby(['時間', '現在の目的地ID', '状態']).size().reset_index(name='人数')

        #BCMP理論値を取り込む
        #self.df_bcmp = pd.read_csv('./BCMP/mean_L.csv')

    def process_obstacle_data(self):
        """Excelから読み込んだ障害物データを構造化する"""
        obstacles = {
            "lines": [],
            "polygons": [],
            "circles": []
        }
        
        for _, row in self.df_obstacles.iterrows():
            try:
                if row['Type'] == 'Line':
                    # 文字列から座標を抽出
                    start = self.parse_coordinates(row['Coordinates'])
                    end = self.parse_coordinates(row['Extra Info'])
                    
                    # 直線の方程式のパラメータを計算 (ax + by + c = 0)
                    if start and end:
                        x1, y1 = start
                        x2, y2 = end
                        if x2 - x1 != 0:  # 垂直線でない場合
                            a = (y2 - y1) / (x2 - x1)
                            b = -1
                            c = y1 - a * x1
                        else:  # 垂直線の場合
                            a = 1
                            b = 0
                            c = -x1
                            
                        obstacles["lines"].append({
                            "id": row['ID'],
                            "start": start,
                            "end": end,
                            "a": a,
                            "b": b,
                            "c": c,
                            "repulsion_strength": row['Repulsion Strength (A)'],
                            "repulsion_range": row['Repulsion Range (B)']
                        })
                
                elif row['Type'] == 'Circle':
                    center = self.parse_coordinates(row['Coordinates'])
                    radius = float(row['Extra Info'])
                    if center and radius:
                        obstacles["circles"].append({
                            "id": row['ID'],
                            "center": center,
                            "radius": radius,
                            "repulsion_strength": row['Repulsion Strength (A)'],
                            "repulsion_range": row['Repulsion Range (B)']
                        })
                
                elif row['Type'] == 'Polygon':
                    vertices = ast.literal_eval(row['Coordinates'])
                    if vertices:
                        obstacles["polygons"].append({
                            "id": row['ID'],
                            "vertices": vertices,
                            "repulsion_strength": row['Repulsion Strength (A)'],
                            "repulsion_range": row['Repulsion Range (B)']
                        })
                
            except Exception as e:
                print(f"Error processing obstacle {row['Type']} {row['ID']}: {e}")
        
        return obstacles

    def parse_coordinates(self, coord_str):
        """座標文字列をタプルに変換する"""
        try:
            if isinstance(coord_str, str):
                # 括弧と空白を削除し、カンマで分割
                coords = coord_str.strip('()[] ').split(',')
                return tuple(float(x.strip()) for x in coords)
            return coord_str
        except Exception as e:
            print(f"Error parsing coordinates {coord_str}: {e}")
            return None
        
    def animate_agents(self, step_interval=1):
        # プロットの初期設定 (2行1列のレイアウト)
        fig, (ax_movements, ax_stacked) = plt.subplots(2, 1, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.4, right=0.85)  # 右側の余白を調整して凡例を外に配置可能に

        '''
        # アニメーション領域1: エージェント移動
        ax_movements.set_xlim(min(self.df['位置X'].min(), self.df_destinations['目的地X'].min()) - 1, 
                              max(self.df['位置X'].max(), self.df_destinations['目的地X'].max()) + 1)
        ax_movements.set_ylim(min(self.df['位置Y'].min(), self.df_destinations['目的地Y'].min()) - 1, 
                              max(self.df['位置Y'].max(), self.df_destinations['目的地Y'].max()) + 1)
        '''
        ax_movements.set_xlim(0, 20)  # X軸の範囲を固定
        ax_movements.set_ylim(0, 20)  # Y軸の範囲を固定
        ax_movements.set_aspect('equal', adjustable='box')  # アスペクト比を固定
        ax_movements.set_title("Agent Movements Over Time")
        ax_movements.set_xlabel("X Position")
        ax_movements.set_ylabel("Y Position")
        ax_movements.grid(True)

        # 各エージェントのプロットを初期化
        agent_plots = {}
        for agent_id in self.agent_ids:
            agent_plots[agent_id], = ax_movements.plot([], [], 'o', label=f"Agent {agent_id}")

        # 目的地のプロット
        for _, dest in self.df_destinations.iterrows():
            ax_movements.scatter(dest['目的地X'], dest['目的地Y'], color='red', marker='x', s=100, 
                                 label=f"Destination {int(dest['目的地ID'])}")
            ax_movements.text(dest['目的地X'], dest['目的地Y'], str(int(dest['目的地ID'])), 
                              color='black', fontsize=10, ha='center', va='center')
            
        # 障害物の描画
        # 直線の描画
        for line in self.obstacles["lines"]:
            ax_movements.plot([line["start"][0], line["end"][0]], 
                            [line["start"][1], line["end"][1]], 
                            color='black', linestyle='--', linewidth=2,
                            zorder=1, label=f'Line {line["id"]}')

        # 円の描画
        for circle in self.obstacles["circles"]:
            circle_patch = Circle(circle["center"], circle["radius"],
                                color='brown', alpha=0.5,
                                zorder=1, label=f'Circle {circle["id"]}')
            ax_movements.add_patch(circle_patch)

        # 多角形の描画
        for polygon in self.obstacles["polygons"]:
            poly_patch = Polygon(polygon["vertices"],
                               closed=True, color='gray', alpha=0.5,
                               zorder=1, label=f'Polygon {polygon["id"]}')
            ax_movements.add_patch(poly_patch)

        # 凡例を領域外に設定
        #ax_movements.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Agents")

        # アニメーション領域2: 積み上げグラフ
        ax_stacked.set_title("State Distribution by Destination")
        ax_stacked.set_xlabel("Destination ID")
        ax_stacked.set_ylabel("Number of Agents")
        ax_stacked.grid(True)

        # アニメーションの最大フレーム数を決定
        max_frames = len(self.df['時間'].unique())

        # 領域内外のエージェント数を表示するテキスト
        count_text = ax_movements.text(0.05, 0.90, '', transform=ax_movements.transAxes, fontsize=12, 
                                       verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # アニメーションの更新関数
        def update(frame):
            # 現在の時間
            current_time = self.df['時間'].unique()[frame]

            # 表示領域の取得
            x_min, x_max = ax_movements.get_xlim()
            y_min, y_max = ax_movements.get_ylim()

            # 領域内にいるエージェントをカウント
            in_count = 0
            out_count = 0

            for agent_id in self.agent_ids:
                agent_df = self.df[self.df['エージェントID'] == agent_id]
                if frame < len(agent_df):
                    x_pos = agent_df['位置X'].iloc[frame]
                    y_pos = agent_df['位置Y'].iloc[frame]
                    if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
                        in_count += 1
                    else:
                        out_count += 1

            # エージェント移動の更新
            for agent_id in self.agent_ids:
                agent_df = self.df[self.df['エージェントID'] == agent_id]
                if frame < len(agent_df):
                    x_pos = [agent_df['位置X'].iloc[frame]]
                    y_pos = [agent_df['位置Y'].iloc[frame]]
                    agent_plots[agent_id].set_data(x_pos, y_pos)

            # 積み上げグラフの更新
            df_current = self.df[self.df['時間'] == current_time]

            # タイトルを更新して時刻を表示
            ax_movements.set_title(f"Agent Movements Over Time (Time: {current_time:.2f})")

            # エージェント数を表示
            count_text.set_text(f"In: {in_count}, Out: {out_count}")

            # 各目的地ごとの状態人数を計算
            destination_ids = self.df_destinations['目的地ID'].unique()
            move_counts = []
            serve_counts = []
            queue_counts = []

            for destination in destination_ids:
                move_count = len(df_current[(df_current['状態'] == 'move') & (df_current['現在の目的地ID'] == destination)])
                serve_count = len(df_current[(df_current['状態'] == 'served') & (df_current['現在の目的地ID'] == destination)])
                queue_count = len(df_current[(df_current['状態'] == 'queue') & (df_current['現在の目的地ID'] == destination)])

                move_counts.append(move_count)
                serve_counts.append(serve_count)
                queue_counts.append(queue_count)

            ax_stacked.clear()
            ax_stacked.bar(destination_ids, move_counts, label='Move', color='blue', bottom=[q + s for q, s in zip(queue_counts, serve_counts)])
            ax_stacked.bar(destination_ids, serve_counts, label='Served', color='green', bottom=queue_counts)
            ax_stacked.bar(destination_ids, queue_counts, label='Queue', color='red')
            ax_stacked.set_title("State Distribution by Destination")
            ax_stacked.set_xlabel("Destination ID")
            ax_stacked.set_ylabel("Number of Agents")
            ax_stacked.grid(True)
            ax_stacked.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="States")

            return list(agent_plots.values())

        # アニメーションを作成
        ani = FuncAnimation(fig, update, frames=range(0, max_frames, step_interval), blit=False)
        plt.show()

    def calculate_average_in_system_per_destination(self):
        """
        拠点ごとの平均系内人数（移動を含む・除く）を計算して表示する
        """
        # 各時刻、目的地ごとの状態別人数を計算
        def count_by_state(group):
            without_move = len(group[group['状態'].isin(['served', 'queue'])])
            with_move = len(group)
            return pd.Series({
                '系内人数(移動を除く)': without_move,
                '系内人数(移動を含む)': with_move
            })

        # 新しい方法でグループ化と集計を行う
        results = (self.df.groupby(['現在の目的地ID', '時間'])
                .apply(count_by_state, include_groups=False)
                .reset_index())

        # 拠点ごとの平均を計算
        averages = (results.groupby('現在の目的地ID')
                    [['系内人数(移動を除く)', '系内人数(移動を含む)']]
                    .mean()
                    .reset_index())
        
        print("\n拠点ごとの平均系内人数:")
        print(averages)
        
        return averages
    
    def plot_average_in_system_per_destination(self, averages, save_path='average_in_system_per_destination.png'):
        """
        Plot average number of agents per destination
        """
        plt.figure(figsize=(10, 6))
        
        # Get data
        x = averages['現在の目的地ID']  # or 'Destination ID' depending on your column name
        y_without_move = averages['系内人数(移動を除く)']  # or 'In System (Excluding Moving)'
        y_with_move = averages['系内人数(移動を含む)']  # or 'In System (Including Moving)'

        # Create bar chart
        width = 0.35
        plt.bar(x - width/2, y_without_move, width, label='Excluding Moving', color='blue', alpha=0.7)
        plt.bar(x + width/2, y_with_move, width, label='Including Moving', color='green', alpha=0.7)

        # Graph decoration
        plt.title("Average Number of Agents per Destination")
        plt.xlabel("Destination ID")
        plt.ylabel("Average Number of Agents")
        plt.xticks(x)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save graph
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph saved as {save_path}")

    def plot_boxplot_per_destination(self, exclude_move=True, save_path='boxplot_per_destination.png'):
        """
        Create boxplots for each destination showing the distribution of agents in the system over time and save the plot.
        
        Args:
            exclude_move (bool): If True, exclude agents in the "move" state.
            save_path (str): Path to save the plot as an image.
        """
        # Filter data to exclude "move" state if specified
        if exclude_move:
            df_filtered = self.df[self.df['状態'] != 'move']
        else:
            df_filtered = self.df

        # Aggregate data by time and destination
        df_grouped = df_filtered.groupby(['時間', '現在の目的地ID'])['エージェントID'].count().reset_index()
        df_grouped.columns = ['Time', 'Destination ID', 'Number of Agents']

        plt.figure(figsize=(12, 6))

        # Create boxplot for each destination
        destination_ids = df_grouped['Destination ID'].unique()
        data_to_plot = [df_grouped[df_grouped['Destination ID'] == dest]['Number of Agents'].values for dest in destination_ids]

        plt.boxplot(
            data_to_plot, 
            positions=destination_ids, 
            widths=0.6, 
            patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='blue'),
            medianprops=dict(color='red', linewidth=2)
        )

        # Set labels and title
        title = "Boxplot of Number of Agents per Destination Over Time"
        if exclude_move:
            title += " (Excluding Move)"
        plt.title(title)
        plt.xlabel("Destination ID")
        plt.ylabel("Number of Agents")
        plt.xticks(destination_ids)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Boxplot saved as {save_path}")


# 使用方法
csv_file_path = 'sfm_log.csv'
file_path = 'SocialForceModel_multiple_destinations.xlsx'
animation = AgentAnimation(file_path)
animation.animate_agents(step_interval=5)

# 拠点ごとの平均系内人数を算出
averages = animation.calculate_average_in_system_per_destination()

# 平均値を可視化して画像として保存
animation.plot_average_in_system_per_destination(averages, save_path='average_in_system_per_destination.png')

# Boxplotを描画
#animation.plot_boxplot_per_destination(exclude_move=True, save_path='boxplot_without_move.png') #移動中は除く
animation.plot_boxplot_per_destination(exclude_move=False, save_path='boxplot_with_move.png') #移動中も含める

