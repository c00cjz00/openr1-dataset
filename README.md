## openr1-dataset
介紹: 輸入問題, 執行思考資料生成, 翻譯成繁體中文, 並上傳到HF
### 1. Install
```bash=
sudo apt install git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo cp /home/ubuntu/.local/bin/* /usr/local/bin/
sudo apt install git-lfs
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
uv pip install "distilabel>=1.5.2"
uv pip install openai opencc beautifulsoup4 Pillow
```
### 2. HF login
請產生具備write權限的key
```bash=
huggingface-cli login
```

### 3. Run dataset_api.py (從Q 生成<think>思考過程</think>)
輸入問題, 執行思考資料生成, 並上傳到HF
#### Edit dataset_api.py
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
- 執行程式
```
source openr1/bin/activate
python  dataset_api.py
```


### 4. Run dataset_api_qa.py (從Q+A 生成<think>思考過程</think>)
輸入問題, 執行思考資料生成, 並上傳到HF
#### Edit dataset_api_qa.py
- 修改以下設定欄位資料 
```python=
# 資料相關設定
pipeline_id = "pipeline03"  # 管道 ID
source_hf_repo_id = "lianghsun/tw-instruct-500k" # 資料來源 Hugging Face 的 repository ID
question_column_name = "input"  # 在 Hugging Face 數據集中的問題欄位名稱
answer_column_name = "output"  # 在 Hugging Face 數據集中的答案欄位名稱
data_num = 2    # 加載數據集並選擇前 10 條數據
distill_num_generations = 1  # 進行的 QA 生成 distillation 次數
save2hf_repo_id = "c00cjz00/pipeline03-deepseek-r1"  # 儲存至 Hugging Face 的 repository ID

# 設定 模型 API 請求的基礎 URL 和金鑰
base_url = r"https://medusa-poc.genai.nchc.org.tw/v1"  # 服務的 API 基礎 URL
api_key = "sk-v96Jgfqo65NwGKjxDcvzaw"  # 用於 API 認證的金鑰
model_id = "DeepSeek-R1"  # 使用的模型 ID
temperature = 0.6  # 生成文本時的溫度參數，影響生成的隨機性
max_new_tokens = 8192  # 生成文本的最大 token 數量

# 模板設置，根據給定的問題一步步推理並生成答案
prompt_template = """
請根據給定的"問題"及"參考答案"，以台灣人的思維方式，用繁體中文一步一步推理，請確保 `<think>` 標籤內的內容皆以繁體中文呈現，並保持原意。最後將答案放入 \boxed{} 中：
{{ instruction }}
"""
```

- 執行程式
```
source openr1/bin/activate
python  dataset_api_qa.py
```




### 5. Run dataset_api_qa_s2twp.py (從Q+A 生成<think>思考過程</think>+翻譯成繁體中文)
輸入問題, 執行思考資料生成, 翻譯成繁體中文, 並上傳到HF
#### Edit dataset_api_s2twp.py
- 修改以下設定欄位資料 
```python=
# 資料相關設定
pipeline_id = "pipeline02"  # 管道 ID
source_hf_repo_id = "ticoAg/Chinese-medical-dialogue" # 資料來源 Hugging Face 的 repository ID
question_column_name = "input"  # 在 Hugging Face 數據集中的問題欄位名稱
data_num = 10    # 加載數據集並選擇前 10 條數據
distill_num_generations = 2  # 進行的 QA 生成 distillation 次數
save2hf_repo_id = "c00cjz00/pipeline02-deepseek-r1-s2twp"  # 儲存至 Hugging Face 的 repository ID

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

- 6. 執行程式
```
source openr1/bin/activate
python  dataset_api_qa_s2twp.py
```

### Input/Output
輸入問題及產出資料
#### Input (string)
```txt=
女寶寶，剛7歲，這一年，察覺到，我家孩子身上肉很多，而且，食量非常的大，平時都不喜歡吃去玩，請問：小兒肥胖超重該如何治療
```
#### Output: generation (string)
```txt!
<think> 嗯，使用者的問題是關於7歲女寶寶肥胖超重如何治療。首先，我需要確認小兒肥胖的診斷標準。通常，兒童肥胖會用到BMI指數，根據年齡和性別來判定。 7歲女孩的BMI超過同齡95%以上才算肥胖，可能需要生長曲線圖來確認。接下來要考慮可能的原因。家長提到食量大，不喜歡活動，這可能是主要因素。但也要排除內分泌問題，例如甲狀腺功能低下或庫欣綜合徵，不過這些情況相對少見，大多數還是單純性肥胖，也就是攝入熱量超過消耗，缺乏運動。治療方面，應從飲食、運動、行為乾預、家庭支援等方面著手。飲食控制不是節食，而是調整飲食結構，減少高糖高脂食物，增加蔬菜、全穀物和優質蛋白質。避免含糖飲料和零食，定時定量進餐。可能需要營養師制定個性化的飲食計劃。運動方面，需逐漸增加活動量，每天至少60分鐘中高強度運動，如游泳、騎腳踏車、跳繩等。減少久坐時間，限制電子螢幕時間，鼓勵戶外活動。行為乾預很重要，例如記錄飲食和運動情況，設定合理目標，給予獎勵機制，但不要用食物作為獎勵。同時，家長要以身作則，全家一起參與健康的生活方式，而不是單獨針對孩子。心理支援也不能忽視，肥胖的孩子可能會有自卑或社交問題，需要家長和老師的鼓勵，避免批評，增強自信心。如果生活方式乾預效果不佳，可能需要醫學評估是否有並發症，例如糖尿病、高血壓、脂肪肝等。對於兒童來說，一般不建議藥物或手術治療，除非極端情況且有專業醫生指導。另外，家長需要耐心，體重控制是長期過程，重點是培養健康習慣，而不是快速減重，以免影響生長發育。定期監測身高體重，評估生長發育情況，避免營養不良。也要注意是否有其他症狀，如頭痛、視力問題、皮膚紫紋等，如果有需要排除繼發性肥胖。此外，睡眠情況也很重要，睡眠不足可能影響代謝，導致肥胖。總之，綜合性的生活方式調整是關鍵，家長積極參與，必要時尋求專業團隊協助，包括兒科醫生、營養師、心理醫生等。教育孩子建立正確的飲食和運動習慣，才能長期維持健康體重。 </think> 對於7歲兒童肥胖超重的治療，需採取綜合性幹預措施，重點在於長期生活方式的調整而非短期減重。具體建議如下： **1. 醫學評估** \- 先計算BMI並對照兒童生長曲線，確認是否達到肥胖標準（≥95th百分位） \- 排除內分泌疾病（如甲狀腺功能減退）等繼發性肥胖\- 檢查是否伴隨代謝症候群（血糖、血脂、肝功能、血壓等） **2. 飲食管理** • **結構調整**： ✓ 增加蔬菜、全穀物、優質蛋白（魚、豆類、瘦肉） ✓ 減少精製糖（含糖飲料、甜點）、油炸食品、加工零食✓ 用低GI水果替代高糖水果，控制每日水果攝入量• **習慣改進**： ✓ 固定三餐時間，避免隨意加餐✓ 使用兒童專用餐具控制份量✓避免以食物作為獎勵，減少外食頻率**3. 運動介入** ◇ 每日保證60分鐘中高強度運動（如跳繩、游泳、球類運動） ◇ 減少久坐行為（每日螢幕時間≤1小時） ◇ 增加日常活動（步行上學、做家務等） ◇ 推薦親子運動（家庭騎行、戶外遊戲等提高參與度） **4. 行為矯正** • 建立飲食運動日記，與孩子共同製定可達成的小目標• 採用非食物獎勵機制（如集星星兌換繪本、郊遊等） • 改善進食習慣：細嚼慢嚥、專注於進食（避免邊吃邊看螢幕） •保證充足睡眠（7歲兒童建議每日9-12小時） **5. 家庭與心理支援** ✓ 全家共同參與飲食運動計劃，避免單獨針對孩子✓ 定期進行生長發育監測（每3個月測量身高體重） ✓ 關注孩子情緒，預防因肥胖導致的自卑或社交迴避✓ 學校溝通協作，避免體育課限製或飲食歧視**6. 醫療幹預（必要時）** ❗ 僅適用於嚴重肥胖且合併並發症者（需兒童內分泌專科評估） ❗ 12歲以下原則上不採用藥物/手術，特殊情況需嚴格評估風險**注意事項**： •兒童減重應以「體重增速放緩」為目標，而非絕對減重• 需保證每日至少1600kcal熱量攝入，避免影響生長發育• 若6個月生活方式乾預無效，建議轉診至兒童肥胖多學科門診透過以上綜合幹預，多數兒童可實現體重逐步回歸正常軌跡。關鍵在於將健康習慣融入家庭日常生活，而非短期強制控制，同時注意保護孩子的心理健康。
```
#### Output: distilabel_metadata (dict)
```json!
{ "raw_input_text_generation_0": [ { "content": "You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:\n女寶寶，7歲，這一年，剛察覺到，我家孩子身上肉很多，而且，食量非常的大，平時都不喜歡吃去玩，請問：小兒肥胖超重該如何治療。 "<think>\n嗯，使用者的問題是關於7歲女寶寶肥胖超重如何治療。首先，我需要確認小兒肥胖的診斷標準。通常，兒童肥胖會用到BMI指數，根據年齡和性別來判定。 。些情況相對少見，大多數還是單純性肥胖，也就是攝入熱量超過消耗，缺乏運動。 \n\n治療方面，應該從飲食、運動、行為乾預、家庭支援等方面著手。而是調整飲食結構，減少高糖高脂食物，增加蔬菜、全穀物和優質蛋白。 ，需要逐漸增加活動量，每天至少60分鐘中高強度運動，如游泳、騎自行車、跳繩等。同時，家長要以身作則，全家一起參與健康的生活方式，而不是單獨針對孩子。 ，增強自信。可能需要醫學評估是否有並發症，例如糖尿病、高血壓、脂肪肝等。 ，體重控制是一個長期過程，重點是培養健康習慣，而不是快速減重，以免影響生長發育。例如頭痛、視力問題、皮膚紫紋等，如果有需要排除繼發性肥胖。此外，睡眠情況也很重要，睡眠不足可能影響代謝，導致肥胖。 \n\n總之，綜合性的生活方式調整是關鍵，家長積極參與，必要時尋求專業團隊幫助，包括兒科醫生、營養師、心理醫生等。教育孩子建立正確的飲食和運動習慣，才能長期維持健康體重。 \n</think>\n\n對於7歲兒童肥胖超重的治療，需採取綜合性幹預措施，重點在於長期生活方式的調整而非短期減肥。具體建議如下：\n\n**1. 醫學評估** \n\\- 先計算BMI並對照兒童生長曲線，確認是否達到肥胖標準（≥95th百分位） \n\\- 排除內分泌疾病（如甲狀腺功能減退）等繼發性肥胖\n\\- 檢查是否伴隨代謝症候群（血糖、血脂、肝功能、血壓等）\n\n**2. 飲食管理** \n• * *結構調整**： \n ✓ 增加蔬菜、全穀物、優質蛋白（魚、豆類、瘦肉） \n ✓ 減少精製糖（含糖飲料、甜點）、油炸食品、加工零食\n ✓ 用低GI水果取代高糖水果，控制每日水果攝取量\n• **習慣改進**： \n ✓固定三餐時間，避免隨意加餐\n ✓ 使用兒童專用餐具控制份量\n ✓ 避免用食物作為獎勵，減少外食頻率\n\n**3. 運動幹預** \n◇ 每日保證60分鐘中高強度運動（如跳繩、游泳、球類運動） \n◇ 減少久坐行為（每日屏幕時間≤1小時） \n◇ 增加日常活動（步行上學、做家務等） \n◇ 推薦親子運動（家庭騎乘、戶外遊戲等提高參與度）\n\n**4. 行為矯正** \n• 建立飲食運動日記，與孩子共同製定可達成的小目標\n• 採用非食物獎勵機制（如集星星兌換繪本、郊遊等） \n•改善進食習慣：細嚼慢嚥、專注進食（避免邊吃邊看屏幕） \n• 保證充足睡眠（7歲兒童建議每日9-12小時）\n\n**5. 家庭與心理支援* * \n✓ 全家共同參與飲食運動計畫，避免單獨針對孩子\n✓ 定期進行生長發育監測（每3個月測量身高體重） \n✓ 關注孩子情緒，預防因肥胖導致的自卑或社交迴避\n ✓ 學校溝通協作，避免體育課限製或飲食歧視\n\n**6. 醫療幹預（必要時）** \n❗ 僅適用於嚴重肥胖且合併並發症者（需兒童內分泌專科評估） \n ❗ 12歲以下原則上不採用藥物/手術，特殊情況需嚴格評估風險\n\n**注意事項**： \n• 兒童減重應以「體重增速放緩」為目標，而非絕對減重\n• 需保證每日至少1600kcal熱量攝入，避免影響生長發育\n• 若6個月生活方式乾預無效，建議轉診至兒童肥胖多學科門診\n\n通過以上綜合幹預，多數兒童可實現體重逐步回歸正常軌跡。關鍵在於將健康習慣融入家庭日常生活，而非短期強制控制，同時注意保護孩子的心理健康。 ", "statistics_text_generation_0": { "input_tokens": 73, "output_tokens": 1988 } }
```
