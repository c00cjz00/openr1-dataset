# 這段程式碼是用來進行文本生成任務，並處理簡體中文轉換為繁體中文的過程，
# 最終生成文本並將結果儲存至 Hugging Face 的 repository
from datasets import load_dataset  # 用於加載數據集
from opencc import OpenCC  # 引入 opencc 進行簡體轉繁體（台灣用語）
from distilabel.pipeline import Pipeline  # 引入 Distilabel 的 Pipeline 結構
from distilabel.steps.tasks import TextGeneration  # 引入文本生成任務步驟
from distilabel.models.llms import OpenAILLM  # 引入 OpenAI LLM 模型
import json  # 用於將字典轉換為 JSON 字串

# 資料相關設定
pipeline_id = "pipeline04-s2twp"  # 管道 ID
source_hf_repo_id = "lianghsun/tw-instruct-500k" # 資料來源 Hugging Face 的 repository ID
question_column_name = "input"  # 在 Hugging Face 數據集中的問題欄位名稱
answer_column_name = "output"  # 在 Hugging Face 數據集中的答案欄位名稱
data_num = 2    # 加載數據集並選擇前 10 條數據
distill_num_generations = 1  # 進行的 QA 生成 distillation 次數
save2hf_repo_id = "c00cjz00/pipeline04-deepseek-r1-s2twp"  # 儲存至 Hugging Face 的 repository ID

# 設定 模型 API 請求的基礎 URL 和金鑰
base_url = r"https://medusa-poc.genai.nchc.org.tw/v1"  # 服務的 API 基礎 URL
api_key = "sk-"  # 用於 API 認證的金鑰
model_id = "DeepSeek-R1"  # 使用的模型 ID
temperature = 0.6  # 生成文本時的溫度參數，影響生成的隨機性
max_new_tokens = 8192  # 生成文本的最大 token 數量

# 模板設置，根據給定的問題一步步推理並生成答案
prompt_template = """
請根據給定的"問題"及"參考答案"，以台灣人的思維方式，用繁體中文一步一步推理，請確保 `<think>` 標籤內的內容皆以繁體中文呈現，並保持原意。最後將答案放入 \boxed{} 中：
{{ instruction }}
"""
#prompt_template = """\
#You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
#{{ instruction }}"""

# 函數：將數據中的簡體中文轉換為台灣繁體
def convert_dataset_text(dataset, columns):
    for column in columns:
        dataset = dataset.map(lambda x: {column: cc.convert(x[column])})  # 使用 OpenCC 進行轉換
    return dataset

# 函數：將 distilabel 的元數據中的簡體中文轉換為台灣繁體
def convert_distilabel_metadata(dataset, columns):
    for column in columns:
        if column in dataset.features:
            # 使用 OpenCC 進行簡體轉繁體處理字典中的文本
            dataset = dataset.map(lambda x: {
                **x,
                column: {key: cc.convert(value) if isinstance(value, str) else value
                         for key, value in x[column].items()}  # 僅轉換字典中的文本
            })
    return dataset

# 函數：問題QA集合
def replace_input_with_combined_data(dataset):
    dataset = dataset.map(lambda x: {
        **x,
        "combination": f"""
### 問題  
{x[question_column_name]}
### 參考答案  
{x[answer_column_name]}"""
    })
    return dataset

# 初始化 OpenCC 進行簡體轉繁體（台灣用語）
cc = OpenCC('s2twp')

# 加載數據集並選擇前 10 條數據
datasets = load_dataset(source_hf_repo_id, split="train").select(range(data_num))

# 執行文本轉換（簡體轉繁體），將數據中的相關欄位進行轉換, 並新增欄位
datasets = convert_dataset_text(datasets, ["input", "output", "instruction"])

# 組合新提示欄位
datasets_combination = replace_input_with_combined_data(datasets)

# 設定 Distilabel pipeline 進行文本生成
with Pipeline(
    name=pipeline_id,  # 指定管道名稱
    description="A pipeline to generate data from a distilled r1 model",  # 描述
) as pipeline:
    
    # 初始化 OpenAILLM（OpenAI 模型）
    llm = OpenAILLM(
        model=model_id,
        base_url=base_url,
        api_key=api_key,
        generation_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens}  # 設定生成參數
    )

    prompt_column = "combination"  # 設定問題欄位名稱
    text_generation = TextGeneration(
        llm=llm,  # 使用的模型
        template=prompt_template,  # 使用的提示模板
        num_generations=distill_num_generations,  # 設定生成次數
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}  # 設定輸入欄位映射
    )

# 執行管道並生成結果
if __name__ == "__main__":
    distiset = pipeline.run(dataset=datasets_combination)  # 執行管道，並將數據集傳入管道進行生成

    # 確保 distiset 內部的 'train' 部分應用 s2twp 轉換
    train_dataset = distiset["default"]["train"]
    train_dataset = convert_dataset_text(train_dataset, ["generation"])  # 轉換生成的文本
    train_dataset = convert_distilabel_metadata(train_dataset, ["distilabel_metadata"])  # 轉換元數據
    distiset["default"]["train"] = train_dataset  # 更新處理過的數據集

    # 推送生成結果到 Hugging Face Hub
    distiset.push_to_hub(repo_id=save2hf_repo_id)  # 將結果推送到指定的 Hugging Face repository
