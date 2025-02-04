## openr1-dataset
### Install
```bash=
sudo apt install git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo cp /home/ubuntu/.local/bin/* /usr/local/bin/
sudo apt install git-lfs
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
uv pip install "distilabel>=1.5.2"
uv pip install openai opencc beautifulsoup4 Pillow
```
### HF login
```bash=
huggingface-cli login
```

### Run dataset_api_s2twp.py
#### Edit dataset_api_s2twp.py
- 修改以下設定欄位資料 
```python=
# 資料相關設定
pipeline_id = "pipeline01"  # 管道 ID
source_hf_repo_id = "ticoAg/Chinese-medical-dialogue" # 資料來源 Hugging Face 的 repository ID
question_column_name = "input"  # 在 Hugging Face 數據集中的問題欄位名稱
data_num = 10    # 加載數據集並選擇前 10 條數據
distill_num_generations = 2  # 進行的 QA 生成 distillation 次數
save2hf_repo_id = "c00cjz00/pipeline01-deepseek-r1"  # 儲存至 Hugging Face 的 repository ID

# 設定 模型 API 請求的基礎 URL 和金鑰
base_url = r"https://medusa-poc.genai.nchc.org.tw/v1"  # 服務的 API 基礎 URL
api_key = "sk-"  # 用於 API 認證的金鑰
model_id = "DeepSeek-R1"  # 使用的模型 ID
temperature = 0.6  # 生成文本時的溫度參數，影響生成的隨機性
max_new_tokens = 8192  # 生成文本的最大 token 數量

# 模板設置，根據給定的問題一步步推理並生成答案
prompt_template = """
請根據給定的問題，以台灣人的思維方式，用繁體中文一步一步推理，請確保 `<think>` 標籤內的內容皆以繁體中文呈現，並保持原意。最後將答案放入 \boxed{} 中：  
{{ instruction }}"""
```

- 修改以下翻譯欄位名稱
"input", "output", "instruction"
```txt=
# 執行文本轉換（簡體轉繁體），將數據中的相關欄位進行轉換
datasets = convert_dataset_text(datasets, ["input", "output", "instruction"])
``` 
