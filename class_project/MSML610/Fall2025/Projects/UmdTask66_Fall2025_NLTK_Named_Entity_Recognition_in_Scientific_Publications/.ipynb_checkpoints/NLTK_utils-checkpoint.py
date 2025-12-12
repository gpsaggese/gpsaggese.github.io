import json
import re
import nltk
import spacy
import tarfile
import os
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------
# 1. 配置与环境准备
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "data"
CORD19_FILENAME = 'cord-19_2022-06-02.tar.gz'
CORD19_FILE_PATH = str(DATA_DIR / CORD19_FILENAME)
CORD19_URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz"
OUTPUT_DIR = PROJECT_ROOT / "output"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_dataset(url, dest_path, chunk_size=1024*1024):
    """
    下载大文件，支持断点续传和进度条显示。
    """
    dest_path = Path(dest_path)
    
    # 1. 获取远程文件大小
    try:
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to URL: {e}")
        return

    # 2. 检查本地文件是否存在及大小
    downloaded_size = 0
    if dest_path.exists():
        downloaded_size = dest_path.stat().st_size
        if downloaded_size == total_size:
            print(f"Dataset already exists and is complete: {dest_path}")
            return
        elif downloaded_size > total_size:
            print("Local file is larger than remote file. Redownloading...")
            downloaded_size = 0
        else:
            print(f"Resuming download from {downloaded_size / (1024**3):.2f} GB...")

    # 3. 设置请求头进行断点续传
    headers = {}
    if downloaded_size > 0:
        headers['Range'] = f'bytes={downloaded_size}-'

    # 4. 开始下载
    print(f"Downloading dataset to {dest_path}...")
    try:
        with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            # 进度条设置
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            with tqdm(total=total_size, initial=downloaded_size, unit='B', unit_scale=True, desc=CORD19_FILENAME, bar_format=bar_format) as pbar:
                with open(dest_path, 'ab') as f: # 使用 'ab' (append binary) 模式
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print("\nDownload complete!")
    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed: {e}")
        print("Please run the script again to resume.")
        exit(1) # 下载失败直接退出

def setup_nltk():
    """下载必要的NLTK数据包"""
    # 更新列表，加入 'maxent_ne_chunker_tab'
    required_packages = [
        'punkt', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng',
        'maxent_ne_chunker', 
        'maxent_ne_chunker_tab', # 必须添加这个包
        'words', 
        'punkt_tab'
    ]
    
    print("Checking NLTK packages...")
    for package in required_packages:
        # 直接调用 download，它会自动跳过已安装的包，比手动检查路径更稳健
        nltk.download(package, quiet=True)

setup_nltk()

# ---------------------------------------------------------
# 2. 数据流式加载 (Streaming Data Acquisition)
# ---------------------------------------------------------
def stream_cord19_data(tar_path, limit=5):
    """
    处理嵌套的 tar.gz 结构：
    Outer tar -> 2022-06-02/document_parses.tar.gz -> pdf_json/*.json
    
    Args:
        tar_path (str): .tar.gz 文件的路径
        limit (int): 为了演示，默认只处理前几篇论文。设为 None 则处理所有。
    
    Yields:
        dict: 处理后的论文数据
    """
    if not os.path.exists(tar_path):
        print(f"Error: File not found at {tar_path}")
        return

    print(f"Opening dataset: {tar_path}...")
    
    count = 0
    try:
        # 1. 打开外层 tar
        with tarfile.open(tar_path, mode="r:gz") as outer_tar:
            # 寻找内部的 document_parses.tar.gz
            inner_tar_member = None
            for member in outer_tar:
                if 'document_parses.tar.gz' in member.name:
                    inner_tar_member = member
                    break
            
            if not inner_tar_member:
                return

            # 2. 提取内部 tar 为文件对象 (BytesIO)
            f_obj = outer_tar.extractfile(inner_tar_member)
            if f_obj:
                # 3. 打开内部 tar
                with tarfile.open(fileobj=f_obj, mode="r:gz") as inner_tar:
                    for member in inner_tar:
                        # 4. 寻找 JSON 文件
                        if member.isfile() and member.name.endswith('.json'):
                            json_file = inner_tar.extractfile(member)
                            if json_file:
                                try:
                                    content = json.load(json_file)
                                    
                                    # 提取数据
                                    paper_id = content.get('paper_id')
                                    title = content.get('metadata', {}).get('title', 'Unknown Title')
                                    body_text_list = content.get('body_text', [])
                                    full_text = " ".join([p.get('text', '') for p in body_text_list])
                                    
                                    if full_text:
                                        yield {
                                            'id': paper_id,
                                            'title': title,
                                            'text': full_text
                                        }
                                        count += 1
                                    
                                    if limit and count >= limit:
                                        return
                                        
                                except Exception:
                                    continue
    except Exception as e:
        print(f"Error: {e}")

# ---------------------------------------------------------
# 3. 文本清洗 (Text Cleaning)
# ---------------------------------------------------------
def clean_text(text):
    """
    清洗文本：移除引用标记 [1], 移除多余空白等
    """
    # 移除类似 [1], [12] 的引用
    text = re.sub(r'\[\d+\]', '', text)
    # 移除 URL
    text = re.sub(r'http\S+', '', text)
    # 替换特殊字符但保留标点（为了分句）
    text = text.replace('\n', ' ')
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# 4. 实体识别模型 (Entity Recognition)
# ---------------------------------------------------------
def extract_entities_nltk(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(pos_tags)
    
    entities = set() # 使用集合去重
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity_name = ' '.join(c[0] for c in chunk)
            # 统一转小写以便比较
            entities.add(entity_name.lower())
    return list(entities)

def extract_entities_spacy(text, nlp_model):
    # 截断过长的文本以防内存溢出 (spaCy默认限制1,000,000字符)
    if len(text) > 900000:
        text = text[:900000]
    doc = nlp_model(text)
    # 只返回实体文本，转小写
    return list(set([ent.text.lower() for ent in doc.ents]))

def extract_entities_transformers(text, pipe):
    # Transformers 处理长文本较慢且有长度限制，这里只截取前512个字符做演示
    # 实际生产中需要切分文本(sliding window)处理
    sample_text = text[:512] 
    results = pipe(sample_text)
    entities = set()
    current_entity = ""
    
    for res in results:
        word = res['word']
        if word.startswith("##"):
            current_entity += word[2:]
        else:
            if current_entity:
                entities.add(current_entity.lower())
            current_entity = word
    if current_entity:
        entities.add(current_entity.lower())
    return list(entities)

# ---------------------------------------------------------
# 5. 评估逻辑 (Evaluation)
# ---------------------------------------------------------
def calculate_metrics(reference_entities, candidate_entities):
    """
    计算 Precision, Recall, F1
    reference_entities: 被视为"正确答案"的实体列表 (例如 spaCy 的结果)
    candidate_entities: 待评估的实体列表 (例如 NLTK 的结果)
    """
    ref_set = set(reference_entities)
    cand_set = set(candidate_entities)
    
    # True Positives: 两个模型都找到的实体
    tp = len(ref_set.intersection(cand_set))
    # False Positives: 候选模型找到但参考模型没找到
    fp = len(cand_set - ref_set)
    # False Negatives: 参考模型找到但候选模型没找到
    fn = len(ref_set - cand_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "overlap_count": tp,
        "nltk_only_count": fp,
        "spacy_only_count": fn
    }

# ---------------------------------------------------------
# 6. 主程序 (Main Execution)
# ---------------------------------------------------------
def main():
    # 0. 自动下载数据集
    download_dataset(CORD19_URL, CORD19_FILE_PATH)

    # 初始化模型
    print("Initializing models...")
    
    # 加载 spaCy
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        return

    # 存储所有结果的列表
    all_entities_data = []
    performance_metrics = []

    # 处理前 10 篇论文
    LIMIT = 10
    print(f"\n--- Processing {LIMIT} papers from CORD-19 ---")
    
    paper_generator = stream_cord19_data(CORD19_FILE_PATH, limit=LIMIT)
    
    for i, paper in enumerate(paper_generator):
        paper_id = paper['id']
        print(f"Processing [{i+1}/{LIMIT}]: {paper_id}")
        
        cleaned_text = clean_text(paper['text'])
        
        # 1. 运行模型
        # 注意：为了公平比较，我们对 NLTK 也使用全文（虽然慢一点），或者都截断
        # 这里为了演示 F1，我们使用前 2000 个字符
        eval_text = cleaned_text[:2000]
        
        ents_nltk = extract_entities_nltk(eval_text)
        ents_spacy = extract_entities_spacy(eval_text, nlp_spacy)
        
        # 2. 存储实体数据 (用于后续分析)
        for ent in ents_nltk:
            all_entities_data.append({'paper_id': paper_id, 'model': 'NLTK', 'entity': ent})
        for ent in ents_spacy:
            all_entities_data.append({'paper_id': paper_id, 'model': 'spaCy', 'entity': ent})
            
        # 3. 计算性能 (以 spaCy 为 Silver Standard)
        metrics = calculate_metrics(reference_entities=ents_spacy, candidate_entities=ents_nltk)
        metrics['paper_id'] = paper_id
        performance_metrics.append(metrics)

    # ---------------------------------------------------------
    # 7. 结果保存与展示
    # ---------------------------------------------------------
    
    # 保存实体列表
    df_entities = pd.DataFrame(all_entities_data)
    entities_csv_path = OUTPUT_DIR / "extracted_entities.csv"
    df_entities.to_csv(entities_csv_path, index=False)
    print(f"\n[Saved] All extracted entities saved to: {entities_csv_path}")
    
    # 保存性能指标
    df_perf = pd.DataFrame(performance_metrics)
    perf_csv_path = OUTPUT_DIR / "performance_metrics.csv"
    df_perf.to_csv(perf_csv_path, index=False)
    print(f"[Saved] Performance metrics saved to: {perf_csv_path}")
    
    # 打印平均性能
    if not df_perf.empty:
        print("\n" + "="*40)
        print("Average Performance (NLTK vs spaCy as Baseline)")
        print("="*40)
        print(df_perf[['precision', 'recall', 'f1_score']].mean())
        print("="*40)
        print("Note: Since CORD-19 is unlabeled, we treat spaCy results as the")
        print("'Silver Standard' (Ground Truth) to evaluate NLTK's relative performance.")

if __name__ == "__main__":
    main()