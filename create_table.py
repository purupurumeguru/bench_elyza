import numpy as np
import json
import os
import pandas as pd

def create_performance_table(results_dir):
    models = []
    all_scores = {}

    # 各モデルのJSONファイルを読み込む
    #os.listdir(results_dir)
    model_dirs = [
            "bartowski_Meta-Llama-3.1-8B-Instruct-GGUF_Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            "mradermacher_Llama-3.1-8B-EZO-1.1-it-GGUF_Llama-3.1s-8B-EZO-1.1-it.Q8_0.gguf",
            "mmnga_Llama-3-ELYZA-JP-8B-gguf_Llama-3-ELYZA-JP-8B-Q8_0.gguf",
            "bartowski_gemma-2-9b-it-GGUF_gemma-2-9b-it-Q8_0.gguf",
            "mradermacher_EZO-Common-9B-gemma-2-it-GGUF_EZO-Common-9B-gemma-2-it.Q8_0.gguf",
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet-20240620",
    ]
    for model_dir in model_dirs:
        model_path = os.path.join(results_dir, model_dir, 'results.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                data = json.load(f)

            models.append(model_dir)
            for i, item in enumerate(data, 1):
                if f'Task {i}' not in all_scores:
                    all_scores[f'Task {i}'] = {}
                all_scores[f'Task {i}'][model_dir] = item['score']

    # DataFrameを作成
    df = pd.DataFrame(all_scores).T

    # 平均を計算
    df['Average'] = df.mean(axis=1)

    # 分散を計算
    df['Variance'] = df[models].var(axis=1)

    # モデルごとの平均と分散を計算
    model_stats = pd.DataFrame({
        'Average': df[models].mean(),
        'Variance': df[models].var()
    })

    # 全体の平均と分散を計算
    overall_avg = df[models].values.mean()
    overall_var = df[models].values.var()

    # DataFrameに追加
    df.loc['Average'] = model_stats['Average'].tolist() + [overall_avg, np.nan]
    df.loc['Variance'] = model_stats['Variance'].tolist() + [np.nan, overall_var]

    # 小数点以下2桁に丸める
    df = df.round(2)

    # Markdown形式で表を作成
    markdown_table = "| Task | " + " | ".join(df.columns) + " |\n"
    markdown_table += "|" + "---|" * (len(df.columns) + 1) + "\n"

    for index, row in df.iterrows():
        markdown_table += f"| {index} | " + " | ".join(map(lambda x: str(x) if pd.notnull(x) else '-', row.values)) + " |\n"

    # 平均スコアが低いタスク10個を抽出
    low_score_tasks = df.nsmallest(10, 'Average', keep='last')


    # 分散が大きいタスク10個を抽出
    high_variance_tasks = df.nlargest(10, 'Variance')

    # タスク詳細を取得する関数
    def get_task_details(task_index):
        for model in models:
            with open(os.path.join(results_dir, model, 'results.json'), 'r') as f:
                data = json.load(f)
                if 0 <= task_index < len(data):
                    return data[task_index]
        return None

    # Markdown形式で表を作成
    markdown_content = "# Performance Table\n\n"
    markdown_content += "| Task | " + " | ".join(df.columns) + " |\n"
    markdown_content += "|" + "---|" * (len(df.columns) + 1) + "\n"

    for index, row in df.iterrows():
        markdown_content += f"| {index} | " + " | ".join(map(lambda x: str(x) if pd.notnull(x) else '-', row.values)) + " |\n"

    # 平均スコアが低いタスクの詳細情報を追加
    markdown_content += "\n\n# Tasks with Lowest Average Scores\n\n"
    for task in low_score_tasks.index:
        markdown_content += f"## {task}\n\n"
        task_index = int(task.split()[1]) - 1  # "Task X" から数値を抽出し、0-indexedに変換
        task_data = get_task_details(task_index)
        if task_data:
            markdown_content += f"**Input:** {task_data['input']}\n\n"
            for model in models:
                with open(os.path.join(results_dir, model, 'results.json'), 'r') as f:
                    model_data = json.load(f)[task_index]
                markdown_content += f"**{model}** (Score: {model_data['score']})\n\n{model_data['model_output']}\n\n---\n\n"
        else:
            markdown_content += "Task details not found.\n\n"

    # 分散が大きいタスクの詳細情報を追加
    markdown_content += "\n\n# Tasks with Highest Variance\n\n"
    for task in high_variance_tasks.index:
        markdown_content += f"## {task}\n\n"
        task_index = int(task.split()[1]) - 1  # "Task X" から数値を抽出し、0-indexedに変換
        task_data = get_task_details(task_index)
        if task_data:
            markdown_content += f"**Input:** {task_data['input']}\n\n"
            for model in models:
                with open(os.path.join(results_dir, model, 'results.json'), 'r') as f:
                    model_data = json.load(f)[task_index]
                markdown_content += f"**{model}** (Score: {model_data['score']})\n\n{model_data['model_output']}\n\n---\n\n"
        else:
            markdown_content += "Task details not found.\n\n"

    # table.mdファイルに保存
    with open('table.md', 'w') as f:
        f.write(markdown_content)

    print("Table and detailed information have been saved to table.md")

    return df


if __name__ == "__main__":
    results_dir = './results'
    performance_table = create_performance_table(results_dir)
