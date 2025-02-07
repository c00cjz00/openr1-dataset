# - INSTALL package
# sudo apt install git-lfs
# curl -LsSf https://astral.sh/uv/install.sh | sh
# sudo cp /home/ubuntu/.local/bin/* /usr/local/bin/
# sudo apt install git-lfs
# uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
# uv pip install 'distilabel[ray]'
# uv pip install python-dotenv openai opencc beautifulsoup4 Pillow
# - KEY
# echo "OPENAI_API_KEY=sk-xxxx" >.env
# - HF (writting key)
# huggingface-cli login

# - Example
# python generate_cot_from_Q.py \
  # --hf-dataset c00cjz00/ft_dataset_parquet \
  # --hf-dataset-config tw-instruct-500k \
  # --hf-dataset-split train \
  # --dataset_select 2 \
  # --hf-output-dataset c00cjz00/open-r-pipeline01-01 \
  # --vllm-server-url https://medusa-poc.genai.nchc.org.tw/v1 \
  # --model DeepSeek-R1 \
  # --temperature 0.6 \
  # --max-new-tokens 4096 \
  # --num-generations 1 \
  # --input-batch-size 64 \
  # --client-replicas 1 \
  # --timeout 600 \
  # --retries 0 \
  # --prompt-column input \
  # --prompt-template 'You will be given a problem. Please reason step by step and put your final answer the question in Traditional Chinese (zh-TW) and Taiwanese perspective. # Key Guidelines: 1. **Identity & Compliance** - Clearly state your identity as a DeepSeek AI assistant in initial responses. - Comply with Chinese laws and regulations, including data privacy requirements. 2. **Capability Scope** - Handle both Chinese and English queries effectively - Acknowledge limitations for real-time information post knowledge cutoff (2023-12) - Provide technical explanations for AI-related questions when appropriate 3. **Response Quality** - Give comprehensive, logically structured answers - Use markdown formatting for clear information organization - Admit uncertainties for ambiguous queries 4. **Ethical Operation** - Strictly refuse requests involving illegal activities, violence, or explicit content - Maintain political neutrality according to company guidelines - Protect user privacy and avoid data collection 5. **Specialized Processing** - Use <think>...</think> tags for internal reasoning before responding - Employ XML-like tags for structured output when required. 6. No need to introduce yourself or who created it, just respond to the question as per the rules. \n\n \n\nQuestion: {{ instruction }}' 


from typing import Optional
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv
import os

# api key
load_dotenv() # Load the environment variables
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

def build_distilabel_pipeline(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: Optional[str] = None,
    prompt_template: str = "{{ instruction }}",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 0,
) -> Pipeline:
    generation_kwargs = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline().ray() as pipeline:
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key=OPENAI_API_KEY,
                model=model,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            ),
            template=prompt_template,
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {},
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Run distilabel pipeline for generating responses with DeepSeek R1")
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="Dataset config to use",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Template string for formatting prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per problem",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="Batch size for input processing",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of client replicas for parallel processing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: 0)",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="HuggingFace repo to push results to",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private when pushing to HF Hub",
    )
    parser.add_argument(
        "--dataset_select",
        type=int,
        default=2,
        help="Dataset_select (default: 2)",
    )
    args = parser.parse_args()

    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset...")
    dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    #dataset = load_dataset(args.hf_dataset, data_files=f"tw-instruct-500k/train-00001-of-00050.parquet", split="train")
    # 選擇前 data_num 筆資料
    dataset = dataset.select(range(args.dataset_select))
    print("Dataset loaded!")

    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")
        
    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("Dataset pushed!")
