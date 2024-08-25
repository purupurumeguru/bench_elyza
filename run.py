from bench_elyza import bench_elyza

def run_models():
    judge_model = "gpt-4o-mini"
    seed = 314159265
    model_list = [
        ("mradermacher/EZO-Common-9B-gemma-2-it-GGUF", "EZO-Common-9B-gemma-2-it.f16.gguf"),
        ("mradermacher/Llama-3.1-8B-EZO-1.1-it-GGUF", "Llama-3.1-8B-EZO-1.1-it.f16.gguf"),
        ("mradermacher/gemma-2-9B-it-advanced-v2.1-GGUF", "gemma-2-9B-it-advanced-v2.1.f16.gguf"),
    ]

    for repo_id, filename in model_list:
        bench_elyza(repo_id=repo_id, filename=filename, judge_model=judge_model, seed=seed)
    return

if __name__ == "__main__":
    run_models()