## 知識蒸餾 (Think) 生成   
### INSTALL package
```bash=
sudo apt install git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo cp /home/ubuntu/.local/bin/* /usr/local/bin/
sudo apt install git-lfs
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
uv pip install 'distilabel[ray]'
uv pip install python-dotenv openai opencc beautifulsoup4 Pillow
- KEY
echo "OPENAI_API_KEY=sk-xxxx" >.env
- HF (writting key)
huggingface-cli login
```

### 執行範例一 (只需要有Q)
- Example
```python=
python generate_cot_from_Q.py \
  --hf-dataset c00cjz00/ft_dataset_parquet \
  --hf-dataset-config tw-instruct-500k \
  --hf-dataset-split train \
  --dataset_select 2 \
  --hf-output-dataset c00cjz00/open-r-pipeline01 \
  --vllm-server-url https://medusa-poc.genai.nchc.org.tw/v1 \
  --model DeepSeek-R1 \
  --temperature 0.6 \
  --max-new-tokens 4096 \
  --num-generations 1 \
  --input-batch-size 64 \
  --client-replicas 1 \
  --timeout 600 \
  --retries 0 \
  --prompt-column input \
  --prompt-template 'You will be given a problem. Please reason step by step and put your final answer the question in Traditional Chinese (zh-TW) and Taiwanese perspective. # Key Guidelines: 1. **Identity & Compliance** - Clearly state your identity as a DeepSeek AI assistant in initial responses. - Comply with Chinese laws and regulations, including data privacy requirements. 2. **Capability Scope** - Handle both Chinese and English queries effectively - Acknowledge limitations for real-time information post knowledge cutoff (2023-12) - Provide technical explanations for AI-related questions when appropriate 3. **Response Quality** - Give comprehensive, logically structured answers - Use markdown formatting for clear information organization - Admit uncertainties for ambiguous queries 4. **Ethical Operation** - Strictly refuse requests involving illegal activities, violence, or explicit content - Maintain political neutrality according to company guidelines - Protect user privacy and avoid data collection 5. **Specialized Processing** - Use <think>...</think> tags for internal reasoning before responding - Employ XML-like tags for structured output when required. 6. No need to introduce yourself or who created it, just respond to the question as per the rules. \n\nQuestion: {{ instruction }}' 
```

### 執行範例二 (需要有Q+A)
- 讀取資料0:9999
```
--page 1 --page-size 10000
```
- 執行程式
```python=
python generate_cot_from_qa.py \
  --hf-dataset c00cjz00/ft_dataset_parquet \
  --hf-dataset-config tw-instruct-500k-00000 \
  --hf-dataset-split train \
  --hf-output-dataset c00cjz00/tw-instruct-500k-00000-save \
  --vllm-server-url https://medusa-poc.genai.nchc.org.tw/v1 \
  --model DeepSeek-R1 \
  --temperature 0.6 \
  --max-new-tokens 4096 \
  --num-generations 1 \
  --input-batch-size 16 \
  --page 1 \
  --page-size 10000 \
  --client-replicas 1 \
  --timeout 600 \
  --retries 0 \
  --prompt-column prompt \
  --question-column-name input \
  --answer-column-name output \
  --prompt-template 'You will be given a problem with a reference answer. Please reason step by step and put your final answer the question in Traditional Chinese (zh-TW) and Taiwanese perspective. # Key Guidelines: 1. **Identity & Compliance** - Clearly state your identity as a DeepSeek AI assistant in initial responses. - Comply with Chinese laws and regulations, including data privacy requirements. 2. **Capability Scope** - Handle both Chinese and English queries effectively - Acknowledge limitations for real-time information post knowledge cutoff (2023-12) - Provide technical explanations for AI-related questions when appropriate 3. **Response Quality** - Give comprehensive, logically structured answers - Use markdown formatting for clear information organization - Admit uncertainties for ambiguous queries 4. **Ethical Operation** - Strictly refuse requests involving illegal activities, violence, or explicit content - Maintain political neutrality according to company guidelines - Protect user privacy and avoid data collection 5. **Specialized Processing** - Use <think>...</think> tags for internal reasoning before responding - Employ XML-like tags for structured output when required. 6. No need to introduce yourself or who created it, just respond to the question as per the rules. \n\n {{ instruction }}' 
```
