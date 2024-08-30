from bench_elyza import bench_elyza

def run_models():
    judge_model = "gpt-4o-mini"
    seed = 314159265
    model_list = [
        ("bartowski/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q8_0.gguf"),
        ("mradermacher/EZO-Common-9B-gemma-2-it-GGUF", "EZO-Common-9B-gemma-2-it.Q8_0.gguf"),
        ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
        ("mradermacher/Llama-3.1-8B-EZO-1.1-it-GGUF", "Llama-3.1-8B-EZO-1.1-it.Q8_0.gguf"),
        ("mmnga/Llama-3-ELYZA-JP-8B-gguf", "Llama-3-ELYZA-JP-8B-Q8_0.gguf")
    ]

    for repo_id, filename in model_list:
        bench_elyza(repo_id=repo_id, filename=filename, judge_model=judge_model, seed=seed)
    return

if __name__ == "__main__":
    run_models()