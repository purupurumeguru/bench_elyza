# bench_elyza

[ELYZA-tasks-100](https://huggingface.co/datasets/elyza/ELYZA-tasks-100)の評価用スクリプトと評価結果をまとめたレポジトリです。

各モデルの平均スコアだけでなく、各タスクに対する返答を保存しているので、モデル選びの参考にしていただければと思います。

現状の評価用のモデルはgpt-4o-miniです。

# 結果まとめ

| Model | Average Score |
|-------|---------------|
| [sample](results/sample/README.md) | 5.00 |
| gpt-4o-mini | 4.14 |
| gpt-4o | 4.39 |
| claude-3-5-sonnet-20240620 | 4.35 |
| bartowski_gemma-2-9b-it-GGUF_gemma-2-9b-it-Q8_0.gguf | 3.86 |
| mradermacher_EZO-Common-9B-gemma-2-it-GGUF_EZO-Common-9B-gemma-2-it.Q8_0.gguf | 3.92 |
| bartowski_Meta-Llama-3.1-8B-Instruct-GGUF_Meta-Llama-3.1-8B-Instruct-Q8_0.gguf | 3.09 |
| mradermacher_Llama-3.1-8B-EZO-1.1-it-GGUF_Llama-3.1-8B-EZO-1.1-it.Q8_0.gguf | 3.34 |
| mmnga_Llama-3-ELYZA-JP-8B-gguf_Llama-3-ELYZA-JP-8B-Q8_0.gguf | 3.53 |

# インストール

1. このリポジトリをクローンします：
   ```
   git clone https://github.com/yourusername/bench_elyza.git
   cd bench_elyza
   ```

2. 必要な依存関係をインストールします：
   ```
   pip install -r requirements.txt
   ```

# 使用方法

```python
python bench_elyza.py
```

カスタム設定でベンチマークを実行する場合：

```
python bench_elyza.py --repo_id "mradermacher/EZO-Common-9B-gemma-2-it-GGUF" --filename "EZO-Common-9B-gemma-2-it.Q8_0.gguf" --judge_model "gpt-4o-mini" --seed 314159265
```

- `repo_id`: 評価するモデルのHugging Face リポジトリID
- `filename`: モデルのファイル名
- `judge_model`: 採点に使用するモデル（"gpt-4o-mini"またはローカルモデル）
- `seed`: 乱数シードの設定

ベンチマーク実行後、以下の場所に結果が保存されます：

- `results/{model_name}/results.json`: 詳細な評価結果（JSON形式）
- `results/{model_name}/README.md`: 人間が読みやすい形式の評価結果サマリー

また、プロジェクトのルートディレクトリにある`README.md`ファイルに、全モデルの平均スコアの概要が追加されます。
