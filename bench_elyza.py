import argparse
import json
import os
import re
from datasets import load_dataset
from llama_cpp import Llama
from openai import OpenAI

judge_prompt = """問題, 正解例, 採点基準, 言語モデルが生成した回答が与えられます。

# 指示
「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{input_text}

# 正解例
{output_text}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 日本語以外で述べている: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{eval_aspect}

# 言語モデルの回答
{pred}

# ここまでが'言語モデルの回答'です。回答が空白だった場合、1点にしてください。

# 指示
「採点基準」と「正解例」を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。
"""

def load_model(repo_id, filename):
    model = Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        verbose=False
    )
    return model

def add_role(prompt, repo_id):
    system_prompt = "あなたは誠実で優秀な日本人のアシスタントです。"

    if "llama" in repo_id.lower():
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
    elif "gemma" in repo_id.lower():
        return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif "mistral" in repo_id.lower():
        return f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
    elif "phi" in repo_id.lower():
        return f"Instruct: {system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
    elif "gpt-neox" in repo_id.lower() or "pythia" in repo_id.lower():
        return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
    elif "falcon" in repo_id.lower():
        return f"System: {system_prompt}\nHuman: {prompt}\nAssistant:"
    elif "mpt" in repo_id.lower():
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Default case, use Llama format
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

def eval_elyza(repo_id, filename, inference_settings=None):
    if inference_settings is None:
        inference_settings = {
            "max_tokens": 10,
            "temperature": 1.0,
            "seed": 314159265,
            "stop": ["Q:", "User"],
            "echo": False,
        }

    model = load_model(repo_id=repo_id, filename=filename)
    elyza_dataset = load_dataset("elyza/ELYZA-tasks-100")

    eval_results = []

    for i, test_data in enumerate(elyza_dataset["test"]):
        input_data = test_data["input"]
        input_with_role = add_role(input_data, repo_id)

        output = model(
            prompt=input_with_role,
            **inference_settings,
        )
        output_text = output["choices"][-1]["text"]
        print(f"Data {i+1}: ")
        print(input_data)
        print("output")
        print(output_text)
        print("-"*100, flush=True)
        eval_results.append({
            "input": input_data,
            "sample_output": test_data["output"],
            "model_output": output_text,
            "eval_aspect": test_data["eval_aspect"],
        })

    return eval_results

def judge_elyza_local(judge_prompt, eval_results, judge_inference_settings=None):
    if judge_inference_settings == None:
        judge_inference_settings = {
            "max_tokens": 1,
            "temperature": 1.0,
            "seed": 314159265,
            "stop": [" ", "　", "\n", "<", ":", "："],
            "echo": False,
        }

    repo_id = "mradermacher/EZO-Common-9B-gemma-2-it-GGUF"
    filename = "EZO-Common-9B-gemma-2-it.Q8_0.gguf"
    judge_model = load_model(repo_id=repo_id, filename=filename)

    judge_results = []
    for result in eval_results:
        prompt = judge_prompt.format(
            input_text=result["input"],
            output_text=result["sample_output"],
            eval_aspect=result["eval_aspect"],
            pred=result["model_output"]
        )
        output = judge_model(
            prompt=prompt,
            **judge_inference_settings,
        )
        score = output["choices"][0]["text"]
        if isinstance(score, int):
            score = int(score)
        else:
            score = 0

        judge_results.append({
            "input": result["input"],
            "sample_output": result["sample_output"],
            "model_output": result["model_output"],
            "eval_aspect": result["eval_aspect"],
            "score": score
        })

    return judge_results

def judge_elyza_gpt(judge_prompt, eval_results, judge_model="gpt-4o-mini", judge_inference_settings=None):
    if judge_inference_settings is None:
        judge_inference_settings = {
            "max_tokens": 1,
            "temperature": 1.0,
            "seed": 314159265,
            "stop": [" ", "　", "\n", "<", ":", "："]
        }

    client = OpenAI()  # Assumes you've set OPENAI_API_KEY in your environment
    judge_results = []

    for result in eval_results:
        prompt = judge_prompt.format(
            input_text=result["input"],
            output_text=result["sample_output"],
            eval_aspect=result["eval_aspect"],
            pred=result["model_output"]
        )

        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that evaluates responses based on given criteria."},
                {"role": "user", "content": prompt}
            ],
            **judge_inference_settings
        )

        score = response.choices[0].message.content.strip()

        try:
            score = int(score)
        except ValueError:
            score = 0

        judge_results.append({
            "input": result["input"],
            "sample_output": result["sample_output"],
            "model_output": result["model_output"],
            "eval_aspect": result["eval_aspect"],
            "score": score
        })

    return judge_results

def save_results(results, model_name):
    output_dir = f"results/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    with open(f"{output_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save Markdown
    with open(f"{output_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Results for {model_name}\n\n")

        # Calculate and write average score
        average_score = sum(result["score"] for result in results) / len(results)
        f.write(f"## Summary\n\n")
        f.write(f"**Average Score:** {average_score:.2f}\n")

        for i, result in enumerate(results, 1):
            f.write(f"## Task {i}\n\n")
            f.write(f"**Score:** {result['score']}\n\n")
            f.write("### Input\n")
            f.write("```\n")
            f.write(result['input'])
            f.write("\n```\n\n")

            f.write("### Model Output\n")
            f.write("```\n")
            f.write(result['model_output'])
            f.write("\n```\n\n")
            f.write("---\n\n")

def update_readme(model_name, average_score):
    readme_path = "README.md"

    # Read existing content
    with open(readme_path, "r") as f:
        content = f.read()

    # Update or add the results table
    results_table_pattern = r"# 結果まとめ\n\n(.*?)\n\n"
    new_result = f"| {model_name} | {average_score:.2f} |"

    if "# 結果まとめ" in content:
        results_table_match = re.search(results_table_pattern, content, re.DOTALL)
        if results_table_match:
            current_table = results_table_match.group(1)
            if "| Model | Average Score |" in current_table:
                # Add new result to existing table
                updated_table = current_table + "\n" + new_result
            else:
                # Create new table with header and new result
                updated_table = "| Model | Average Score |\n|-------|---------------|\n" + new_result
            updated_content = content.replace(results_table_match.group(0), f"# 結果まとめ\n\n{updated_table}\n\n")
        else:
            # Create new table after "# 結果まとめ"
            new_table = f"# 結果まとめ\n\n| Model | Average Score |\n|-------|---------------|\n{new_result}\n\n"
            updated_content = content.replace("# 結果まとめ", new_table)
    else:
        # Add new section with table if "# 結果まとめ" doesn't exist
        new_section = f"# 結果まとめ\n\n| Model | Average Score |\n|-------|---------------|\n{new_result}\n\n"
        updated_content = content.replace("# インストール", f"{new_section}\n# インストール")

    # Write updated content back to README
    with open(readme_path, "w") as f:
        f.write(updated_content)

    print(f"README.md updated with results for {model_name}")

def bench_elyza(repo_id, filename, judge_model="gpt-4o-mini", seed=314159265):
    model_name = f"{repo_id}/{filename}".replace("/", "_")

    inference_settings = {
        "max_tokens": 1000,
        "temperature": 1.0,
        "seed": seed,
        "stop": ["Q:", "User"],
    }

    judge_inference_setteing = {
        "max_tokens": 1,
        "temperature": 1.0,
        "seed": seed,
    }

    eval_results = eval_elyza(repo_id=repo_id, filename=filename, inference_settings=inference_settings)

    if "gpt" in judge_model:
        judge_results = judge_elyza_gpt(judge_prompt, eval_results, judge_model=judge_model, judge_inference_settings=judge_inference_setteing)
    else:
        judge_results = judge_elyza_local(judge_prompt, eval_results, judge_inference_settings=judge_inference_setteing)


    save_results(judge_results, model_name=model_name)

    # Calculate and print average score
    average_score = sum(result["score"] for result in judge_results) / len(judge_results)
    print(f"Average score for {model_name}: {average_score:.2f}")

    update_readme(model_name, average_score)

def main():
    parser = argparse.ArgumentParser(description="Benchmark language models using the ELYZA dataset.")
    parser.add_argument("--repo_id", type=str, default="mradermacher/EZO-Common-9B-gemma-2-it-GGUF",
                        help="Hugging Face repository ID for the model")
    parser.add_argument("--filename", type=str, default="EZO-Common-9B-gemma-2-it.Q8_0.gguf",
                        help="Filename of the model")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                        help="Model to use for judging (e.g., 'gpt-4o-mini' or a local model)")
    parser.add_argument("--seed", type=int, default=314159265,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    bench_elyza(
        repo_id=args.repo_id,
        filename=args.filename,
        judge_model=args.judge_model,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
