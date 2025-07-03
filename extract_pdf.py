import os
import json
import csv
import re
import concurrent.futures
import requests
import time
from PyPDF2 import PdfReader
import langid
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure, LAParams
import pdfplumber
# import fitz  # PyMuPDF
import io

def extract_text(pdf_path: str) -> str:
    """独立提取单份PDF文本(避免跨文件干扰)[2](@ref)"""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += clean_text(page_text) + "\n"
        return text
    except Exception as e:
        print(f"⚠️ 文件解析失败: {pdf_path} - {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """清理文本（保留可打印字符）[2](@ref)"""
    return re.sub(r'[^\x20-\x7E]', '', text).strip()

def split_chinese_sentences(text):
    # 按中文标点分句
    sentences = re.split(r'([。！？；])', text)  
    # 重组完整句子
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)]
    # 过滤短句
    return [s for s in sentences if len(s) > 1]  

def extract_chinese_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 提取当前页文本
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1)  # 宽松容差防断句
            if page_text:
                # 按段落分割
                paragraphs = page_text.split('\n')
                for para in paragraphs:
                    # 中文分句处理
                    sentences = split_chinese_sentences(para)
                    full_text += "\n".join(sentences)
    return full_text

def split_mixed_sentences(text):
    # 同时匹配中英文标点：句号、感叹号、问号、分号
    pattern = r'([。！？；]|\.[\s]|!|\?|;)'
    sentences = re.split(pattern, text)
    # 重组句子：标点与前文合并
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            combined = sentences[i] + sentences[i+1]
            if len(combined.strip()) > 1:  # 过滤空句和短句
                result.append(combined.strip())
    return result

def extract_mixed_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)  # 宽松容差避免断词
            if not page_text:
                continue
                
            paragraphs = page_text.split('\n')
            for para in paragraphs:
                # 语言检测：若段落为英文则按换行符分句
                lang, _ = langid.classify(para)
                if lang == 'en':
                    sentences = [s.strip() for s in para.split('.') if s]
                    full_text += '.\n'.join(sentences) + '\n'
                else:
                    sentences = split_mixed_sentences(para)
                    full_text += '\n'.join(sentences) + '\n'
    return full_text

def extract_pdf(pdf_path: str, engine: str = "pdfminer") -> str:
    """
    提取PDF中的文字内容（支持中英文混合的学术论文）
    
    参数：
        pdf_path: PDF文件路径
        engine: 解析引擎，可选["pdfminer"(默认) | "pymupdf" | "pdfplumber"]
        ocr_fallback: 当遇到扫描件时是否尝试OCR（需额外安装pytesseract和pdf2image）
    
    返回：
        拼接后的文本字符串（保留段落结构）
    """
    def clean_text(text: str) -> str:
        """清理多余空格、保留段落换行"""
        # 合并连续空格/换行，保留单换行作为段落分隔 [3,8](@ref)
        text = re.sub(r'[ \t\x0c]{2,}', ' ', text)  # 合并连续空格/制表符
        text = re.sub(r'\n{2,}', '\n\n', text)       # 保留双换行为段落分隔
        return text.strip()

    # 引擎选择（根据PDF复杂度自适应）
    """if engine == "pymupdf":  # 速度最快，适合文字型PDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n"  # sort=True保持阅读顺序
        doc.close()
        return clean_text(text)"""
    
    if  engine == "pdfplumber":  # 表格处理能力强
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text(keep_blank_chars=False, use_text_flow=True) + "\n"
        return clean_text(text)
    
    else:  # 默认使用pdfminer.six（布局解析最精确）
        laparams = LAParams(
            char_margin=1.5,   # 字符间距阈值（避免错误分词）
            line_overlap=0.7,  # 行重叠判定
            boxes_flow=0.4,    # 文本流方向敏感度（适合分栏）
        )
        output_str = io.StringIO()
        for page_layout in extract_pages(pdf_path, laparams=laparams):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    output_str.write(element.get_text() + "\n")
        return clean_text(output_str.getvalue())
    
