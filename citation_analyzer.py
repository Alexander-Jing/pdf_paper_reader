import os
import json
import csv
import re
import concurrent.futures
import requests
import time
from extract_pdf import *

# ====== 配置区域 ======
API_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"  # 替换为实际API地址
API_KEY = ""  # 替换为API密钥
PDF_FOLDER = "./papers"  # PDF文件夹路径
OUTPUT_CSV = "results.csv"  # 输出CSV文件名
MAX_CONCURRENT = 4  # 最大并发数（根据API限制调整）
YOUR_PAPER_TITLE = ""  # 被引用的论文标题
# =====================

def call_llm_api(pdf_text: str) -> dict:
    """
    调用硅基流动API分析PDF内容
    优化点：
    1. 使用官方推荐模型
    2. 更健壮的错误处理机制
    3. 请求参数优化
    4. 响应解析简化
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    # 增强提示词设计（聚焦目标论文引用）
    prompt = f"""请严格分析当前PDF文件内容，当前输入的pdf文件内容是一篇学术论文：
    ## 核心任务
    1. 在论文正文中定位对参考文献《{YOUR_PAPER_TITLE}》的引用评价（✅正面/⚠️中性/❌负面）
    2. 提取施引文献的标题和期刊名称
    3. 提取所有作者姓名(按标准格式: Zhang et al. 或 Li, Y.)

    ## 输出要求
    返回纯净JSON对象：
    {{
        "title": "施引文献标题",
        "journal": "期刊名称",
        "authors": ["作者1", "作者2"]，如果是中文名，请直接使用中文名,
        "citations": [{{
            "quote": "寻找论文正文中对参考文献《{YOUR_PAPER_TITLE}》的引用评价原文",
            "sentiment": "评价性质",
            "page": "页码"
        }}]
    }}

    ## 处理规则
    - 忽略非引用内容的文本
    - 无相关引用时返回空列表
    - 保持JSON结构完整

    --- PDF内容(前30000字符) ---
    {pdf_text}"""
    
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # 官方推荐满血版模型
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # 低随机性保证稳定性
        "max_tokens": 2000,   # 增加token限额确保完整输出
        "response_format": {"type": "json_object"}  # 强制JSON输出
    }
    
    max_retries = 3
    wait_time = 5  # 初始等待时间(秒)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_ENDPOINT, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            # 成功响应处理
            if response.status_code == 200:
                response_data = response.json()
                try:
                    # 直接提取模型返回的JSON内容
                    content = response_data["choices"][0]["message"]["content"]
                    return json.loads(content)
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"📊 JSON解析错误: {str(e)}")
                    print(f"原始响应: {content}")
                    return {}
            
            # 限速处理（429）
            elif response.status_code == 429:
                print(f"⏳ API限速(尝试 {attempt+1}/{max_retries}), 等待{wait_time}秒...")
                time.sleep(wait_time)
                wait_time *= 2  # 指数退避策略
            
            # 服务不可用（503）
            elif response.status_code == 503:
                print(f"🛠️ 服务暂时不可用(尝试 {attempt+1}/{max_retries}), 等待{wait_time}秒...")
                time.sleep(wait_time)
                wait_time *= 2
            
            # 其他API错误
            else:
                print(f"❌ API错误({response.status_code}): {response.text[:200]}")
                return {}
                
        # 网络异常处理
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"🌐 网络异常({attempt+1}/{max_retries}): {str(e)}")
            time.sleep(wait_time)
            wait_time *= 2
        
        # 其他异常处理
        except Exception as e:
            print(f"⚠️ 未预期错误: {str(e)}")
            return {}
    
    print(f"🚫 超过最大重试次数({max_retries})，放弃请求")
    return {}

def process_single_pdf(pdf_path: str) -> dict:
    """独立处理单份PDF的全流程"""
    filename = os.path.basename(pdf_path)
    print(f"🔍 开始处理: {filename}")
    
    # 步骤1：独立提取文本（隔离其他文件）
    pdf_text = extract_pdf(pdf_path)
    if not pdf_text:
        return {"filename": filename, "error": "TEXT_EXTRACTION_FAILED"}
    
    # 步骤2：独立调用API（确保无上下文干扰）
    result = call_llm_api(pdf_text)
    if not result:
        return {"filename": filename, "error": "API_CALL_FAILED"}
    
    # 返回结构化结果
    return {
        "filename": filename,
        "title": result.get("title", ""),
        "journal": result.get("journal", ""),
        "authors": result.get("authors", []),
        "citations": result.get("citations", [])
    }

def main():
    # 获取所有PDF文件
    pdf_files = [
        os.path.join(PDF_FOLDER, f) 
        for f in os.listdir(PDF_FOLDER) 
        if f.lower().endswith(".pdf")
    ]
    print(f"📂 发现 {len(pdf_files)} 个PDF文件 | 并发数: {MAX_CONCURRENT}")
    
    # 并发处理（每个PDF独立调用）[8](@ref)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        future_to_file = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        for future in concurrent.futures.as_completed(future_to_file):
            pdf_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                cite_count = len(result.get("citations", []))
                status = f"✅ 发现{cite_count}处引用" if cite_count > 0 else "⚠️ 未找到引用"
                print(f"{status} | {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"❌ 处理失败: {os.path.basename(pdf_path)} - {str(e)}")
    
    # 保存CSV结果[2](@ref)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "施引标题", "期刊", "作者", "引用原文", "情感", "页码"])
        for res in results:
            for cite in res.get("citations", []):
                writer.writerow([
                    res["filename"],
                    res.get("title", ""),
                    res.get("journal", ""),
                    "; ".join(res.get("authors", [])),
                    cite.get("quote", "")[:200] + "..." if len(cite.get("quote", "")) > 200 else cite.get("quote", ""),
                    cite.get("sentiment", ""),
                    cite.get("page", "")
                ])
    print(f"\n🎉 处理完成！结果已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
