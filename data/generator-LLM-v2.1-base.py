import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os
import json
import re
import gc
import random
import traceback
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- 1. 모델 설정 ---

# 8B 모델들
llama3= "meta-llama/Llama-3.1-8B-Instruct"
# --- 시험 모델 ---
nvidia_nemotron = "nvidia/Nemotron-Orchestrator-8B"
qwen3_8b = "Qwen/Qwen3-8B"
foundation = "fdtn-ai/Foundation-Sec-8B-Instruct" #  Foundation AI at Cisco

# 12B 모델들
mistral_n= "mistralai/Mistral-Nemo-Instruct-2407" 
nvidia_nemo_nano = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
# gemma3 = "google/gemma-3-12b-it" # system prompt occurs

# 14B 모델들
qwen3 = "Qwen/Qwen3-14B"
deepseek_d = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
phi4 = "microsoft/phi-4" 
# phi4_r = "microsoft/Phi-4-reasoning" # output problem occurs
# mistral3 = "mistralai/Ministral-3-14B-Reasoning-2512" # Unable to load model

MODEL_NAME = foundation
SAVE_DIR = "./baseline_scenarios_xml"  # 결과 저장 경로 (구분됨)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

FILE_PATHS = {
    "cti_data": "preprocessed_cti_data.json",
    "kev_data": "known_exploited_vulnerabilities.csv",
    "campaign_data": "campaign-merge.json",
    "attack_levels": "attack_levels.json",
    "complexity": "complexity.json",
    "surfaces": "attack_surface_en.json",
    "vectors": "attack_vector_en.json",
    "prompt_txt": "prompt_file.txt",
    "faiss_index": "capec_index.faiss",
    "capec_pkl": "capec_data.pkl",
    "embed_model": "all-MiniLM-L6-v2"
}


def load_model(model_name):
    """
    제안 시스템과 동일한 방식으로 모델을 로드합니다.
    """
    print(f"Loading Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=2048):
    """
    LLM에게 텍스트 생성을 요청하는 함수
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.4, # 창의성 허용 (제안 모델과 비슷하게 유지)
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제거하고 응답만 추출
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "")
        
    return generated_text.strip()

def extract_xml(text):
    match = re.search(r'(<Scenario>.*?</Scenario>)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        # XML 태그가 깨졌을 경우 강제로 wrap
        return f"<Scenario>\n{text}\n</Scenario>"

# --- 3. Pure LLM용 프롬프트 ---
def create_baseline_prompt(config):
    """
    외부 데이터 없이 LLM에게 모든 상세 정보를 직접 생성하도록 요청하는 프롬프트
    """
    return f"""
            You are an expert Red Teamer and Threat Intelligence Analyst.
            Your task is to generate a realistic cyber attack scenario based on the following constraints.

            [Scenario Constraints]
            - Target Industry: {config['INDUSTRY']}
            - Attacker Level: {config['ATTACK_LEVEL']} (e.g., APT or Script Kiddie)
            - Complexity: {config['COMPLEXITY_LEVEL']}

            [Instructions]
            1. Plan a complete attack chain from 'Initial Access' to 'Impact' based on MITRE ATT&CK.
            2. YOU MUST GENERATE ALL TECHNICAL DETAILS YOURSELF using your internal knowledge. 
            - DO NOT use placeholders like [Insert CVE]. 
            - Provide specific real-world examples.
            3. For each stage, you must specify:
            - Technique Name & ID (e.g., T1566 Phishing)
            - Specific Vulnerability (KEV/CVE ID) if applicable.
            - Attack Surface (e.g., VPN, RDP, Outlook).
            - Attack Vector (e.g., Spearphishing Link).
            4. The output must be in the following XML format strictly:

            <Scenario>
                <metadata>
                    <industry>{config['INDUSTRY']}</industry>
                    <level>{config['ATTACK_LEVEL']}</level>
                    <complexity>{config['COMPLEXITY_LEVEL']}</complexity>
                </metadata>
                <stage name="Initial Access">
                    <technique_name>Txxxx: Technique Name</technique_name>
                    <kev>CVE-YYYY-NNNN</kev> 
                    <surface>Specific Asset (e.g., Apache Web Server)</surface>
                    <vector>Specific Method</vector>
                    <capec>CAPEC-XXX (Optional)</capec>
                    <description>Detailed narrative of the attack...</description>
                </stage>
                </Scenario>

            Generate the XML now.
            """

def run_baseline_pipeline(config, model, tokenizer):
    """
    RAG나 Enrichment 없이 한 번에 생성
    """
    print(f"\n[Baseline] Generating scenario for: {config['INDUSTRY']} ({config['ATTACK_LEVEL']})")
    
    # 1. 프롬프트 구성
    prompt = create_baseline_prompt(config)
    
    # 2. 생성 (Pure Generation)
    raw_output = generate_text(model, tokenizer, prompt)
    
    # 3. XML 추출
    final_xml = extract_xml(raw_output)
    
    # 4. 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Baseline_{config['INDUSTRY']}_{config['ATTACK_LEVEL']}_{timestamp}.xml"
    filepath = os.path.join(SAVE_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(final_xml)
        
    print(f"Saved: {filename}")

# --- 4. 메인 실행 루프 ---
def main():
    # 모델 로드
    model, tokenizer = load_model(MODEL_NAME)
    
    # 타겟 설정 (제안 모델과 동일하게 유지)
    TARGET_INDUSTRIES = [
                "Communications",
                "Energy", 
                "Healthcare and Public Health", 
                "Transportation Systems",
                "Water and Wastewater Systems"
                ]
    TARGET_COMPLEXITIES = ["Complex"] # 실험을 위해 'Complex'로 고정 추천
    TARGET_ATTACK_LEVELS = ["Expert"] # 실험을 위해 'Expert'로 고정 추천
    
    # 실험 횟수 (각 설정당 5개씩 생성하여 평균 품질 측정 추천)
    N_SAMPLES = 5 
    
    total_runs = len(TARGET_INDUSTRIES) * len(TARGET_COMPLEXITIES) * len(TARGET_ATTACK_LEVELS) * N_SAMPLES
    current_run = 0

    print(f"--- Pure LLM Baseline Generation Started (Total: {total_runs}) ---")
    
    for industry in TARGET_INDUSTRIES:
        for complexity in TARGET_COMPLEXITIES:
            for attack_level in TARGET_ATTACK_LEVELS:
                for i in range(N_SAMPLES):
                    current_run += 1
                    print(f"Processing [{current_run}/{total_runs}]...")
                    
                    config = {
                        "INDUSTRY": industry,
                        "COMPLEXITY_LEVEL": complexity,
                        "ATTACK_LEVEL": attack_level
                    }
                    
                    try:
                        run_baseline_pipeline(config, model, tokenizer)
                    except Exception as e:
                        print(f"Error generating {industry}: {e}")
                        
    print("--- All Baseline Scenarios Generated ---")

if __name__ == "__main__":
    main()