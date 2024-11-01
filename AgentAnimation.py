import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AgentAnimation:
    def __init__(self, file_path):
        # エクセルファイルを取り込んでデータフレームに読み込む
        self.df = pd.read_excel(file_path, sheet_name='シミュレーション履歴と力')
        # パラメータシートを読み込んで目的地を取得
        df_params = pd.read_excel(file_path, sheet_name='パラメータ')
        self.destination = (df_params['目的地X'].iloc[0], df_params['目的地Y'].iloc[0])
        
    def animate_agents(self, agent_ids, step_interval=1):
        # プロットの初期設定
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(self.df['位置X'].min() - 1, self.df['位置X'].max() + 1)
        ax.set_ylim(self.df['位置Y'].min() - 1, self.df['位置Y'].max() + 1)
        ax.set_title("Agent Movements Over Time")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)

        # 各エージェントのプロットを初期化
        agent_plots = {}
        for agent_id in agent_ids:
            agent_plots[agent_id], = ax.plot([], [], 'o', label=f"Agent {agent_id}")

        # 目的地をプロット
        ax.scatter(self.destination[0], self.destination[1], color='red', marker='x', s=100, label="Destination")
        
        # アニメーションの最大フレーム数を決定
        max_frames = max([len(self.df[self.df['エージェントID'] == agent_id]) for agent_id in agent_ids])

        # アニメーションの更新関数
        def update(frame):
            for agent_id in agent_ids:
                agent_df = self.df[self.df['エージェントID'] == agent_id]
                if frame < len(agent_df):
                    x_pos = [agent_df['位置X'].iloc[frame]]  # シーケンスとして渡す
                    y_pos = [agent_df['位置Y'].iloc[frame]]  # シーケンスとして渡す
                    agent_plots[agent_id].set_data(x_pos, y_pos)
            return agent_plots.values()

        # アニメーションを作成
        ani = FuncAnimation(fig, update, frames=range(0, max_frames, step_interval), blit=True)

        # 凡例を追加し、アニメーションを表示
        plt.legend(loc='upper right')
        plt.show()

# 使用方法
# アニメーションクラスのインスタンスを作成し、アニメーションを実行
file_path = 'SocialForceModel_output_japanese_with_params.xlsx'  # ファイルパスを指定
animation = AgentAnimation(file_path)
animation.animate_agents(agent_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], step_interval=5)  # エージェントIDリストとフレーム間隔を指定
