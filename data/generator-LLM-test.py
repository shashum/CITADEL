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
import sys
from datetime import datetime
import pandas as pd
import difflib # [NEW] 유사도 검사를 위한 라이브러리

# ==============================================================
# [설정 1] 모델 목록
# ==============================================================
# llama3= "meta-llama/Llama-3.1-8B-Instruct",
# mistral= "mistralai/Mistral-Nemo-Instruct-2407",
# qwen3 = "Qwen/Qwen3-14B",
# qwen3_think ="Qwen/Qwen3-4B-Thinking-2507",
# gemma2 = "google/gemma-2-9b-it",
# phi4 = "microsoft/phi-4",
# deepseek_d = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


# ==============================================================
# [설정 2] 파일 경로
# ==============================================================
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

# ==============================================================
# [설정 3] 환각 방지 시스템 프롬프트
# ==============================================================
SYSTEM_PROMPT_STRICT = """
You are a cybersecurity analyst.

### CRITICAL XML OUTPUT RULES

1. **Strict Data Copying (NO CREATIVITY)**:
   - For `<surface>`, `<vector>`, `<kev>`, and `<capec>`, **YOU MUST COPY** the exact string provided in the `attack_chain_plan` of the dossier.
   - **DO NOT** change, summarize, or invent these values. Use them exactly as they appear in the JSON.

2. **Stage Structure**:
   Inside EVERY `<stage>` block, include:
   - `<technique_name>`: ...
   - `<surface>`: Copy exact value from dossier.
   - `<vector>`: Copy exact value from dossier.
   - `<kev>`: ...
   - `<capec>`: ...
   - `<description>`: A detailed narrative connecting these elements.

**Output ONLY the XML block.**
"""

# --- 로거 ---
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- 모델 로드 ---
def load_local_llm(model_name):
    print(f"\n🔄 로컬 모델 로드: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit()

def remove_think_tags(text):
    """
    추론 모델(Reasoning Model)이 출력하는 <think>...</think> 구간을 제거합니다.
    """
    # <think> 태그와 그 사이의 모든 내용을 제거 (re.DOTALL로 줄바꿈 포함)
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def query_local_llm(model, tokenizer, messages, max_new_tokens=8192, temperature=0.1, top_p=0.9, top_k=50):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # [중요] 추론 모델은 말이 많으므로 max_new_tokens를 넉넉하게 늘려야 합니다.
            # 기존 2048/4096 -> 8192 이상 권장
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # [핵심] 생각(<think>) 부분 제거 후 반환
    return remove_think_tags(raw_text)

# --- 리소스 로드 ---
def load_resources(paths):
    print("📚 데이터 로드 중...")
    db = {}
    try:
        with open(paths["attack_levels"], 'r', encoding='utf-8') as f: db['levels'] = json.load(f)
        with open(paths["complexity"], 'r', encoding='utf-8') as f: db['complexity'] = json.load(f)
        with open(paths["surfaces"], 'r', encoding='utf-8') as f: db['surfaces'] = json.load(f)
        with open(paths["vectors"], 'r', encoding='utf-8') as f: db['vectors'] = json.load(f)
        with open(paths["campaign_data"], 'r', encoding='utf-8') as f: db['campaigns'] = json.load(f)
        
        if os.path.exists(paths["prompt_txt"]):
            with open(paths["prompt_txt"], 'r', encoding='utf-8') as f: db['system_prompt'] = f.read()
        else: db['system_prompt'] = "You are a cybersecurity analyst."

        if os.path.exists(paths["cti_data"]):
            with open(paths["cti_data"], 'r', encoding='utf-8') as f:
                cti = json.load(f)
                db['techniques'] = cti.get('techniques', {})
                db['name_to_id'] = {v['name'].lower(): k for k, v in db['techniques'].items()}
                # [NEW] 검색용 기술 이름 리스트
                db['tech_names_list'] = list(db['name_to_id'].keys())
        else: db['techniques'] = {}

        if os.path.exists(paths["kev_data"]):
            try: df = pd.read_csv(paths["kev_data"], encoding='utf-8')
            except: df = pd.read_csv(paths["kev_data"], encoding='latin-1')
            df.columns = df.columns.str.strip()
            key_col = 'cveID' if 'cveID' in df.columns else 'CVE ID'
            if key_col in df.columns:
                df[key_col] = df[key_col].astype(str).str.strip()
                db['kev'] = df.set_index(key_col).to_dict(orient='index')
                db['kev_set'] = set(db['kev'].keys())
            else: db['kev'] = {}
        
        if os.path.exists(paths["faiss_index"]):
            db['rag_index'] = faiss.read_index(paths["faiss_index"])
            with open(paths["capec_pkl"], "rb") as f: db['capec_data'] = pickle.load(f)
            db['rag_model'] = SentenceTransformer(paths["embed_model"])
            
        return db
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

def auto_fill_config(config, db):
    if not config.get("INDUSTRY"):
        config["INDUSTRY"] = random.choice(["Energy", "Healthcare", "Finance", "Defense"])
    if not config.get("COMPLEXITY_LEVEL"):
        config["COMPLEXITY_LEVEL"] = random.choice(["Simple", "Standard", "Complex"])
    if not config.get("ATTACK_LEVEL"):
        config["ATTACK_LEVEL"] = random.choice(["Basic", "Skilled", "Expert"])
    return config

# --- Step 1: 기획 (Planner) ---
def plan_attack_with_local_llm(config, db, model, tokenizer):
    print(f"\n🧠 [Step 1] 기획 ({config['INDUSTRY']} | {config['ATTACK_LEVEL']})...")
    req_tactics = db['complexity'].get(config['COMPLEXITY_LEVEL'], {}).get('tactics', [])
    
    avail_surfaces = [s['NAME'] for s in db['surfaces']]
    avail_vectors = [v['NAME'] for v in db['vectors']]

    prompt = f"""
    You are a Cyber Attack Architect. Plan a hypothetical attack chain.
    
    Constraints:
    1. Target: {config['INDUSTRY']}
    2. Complexity: {config['COMPLEXITY_LEVEL']} (Required Tactics: {json.dumps(req_tactics)})
    3. Output Format: JSON List of Objects.

    [Reference Options]
    - Surfaces: {json.dumps(avail_surfaces[:8])}...
    - Vectors: {json.dumps(avail_vectors[:8])}...

    Task:
    Return a JSON List. For EACH tactic in the required list, specify a technique, surface, and vector.
    Do NOT invent fake technique names. Use standard MITRE ATT&CK names.

    Example:
    [
      {{
        "tactic": "Initial Access",
        "technique_name": "Phishing",
        "surface": "Employee Email",
        "vector": "Social Engineering",
        "reason": "Entry point"
      }}
    ]
    """
    
    messages = [{"role": "system", "content": "Output ONLY valid JSON."}, {"role": "user", "content": prompt}]
    
    response = query_local_llm(model, tokenizer, messages, temperature=0.4)
    
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

# --- [핵심 수정] Step 2: 데이터 보강 및 팩트 체크 (Enricher) ---
def enrich_plan(plan, db):
    print("🔍 [Step 2] 데이터 보강 및 팩트 검증 (Surface/Vector 강제 교정)...")
    enriched = []
    used_surfaces = set()
    used_vectors = set()
    initial_access_info = {"surface": "Unknown", "vector": "Unknown"}
    
    # [핵심] 검증을 위한 정답지(Valid List) 생성
    valid_surface_names = [s['NAME'] for s in db['surfaces']]
    valid_vector_names = [v['NAME'] for v in db['vectors']]
    
    for step in plan:
        if isinstance(step, str) or not isinstance(step, dict): continue

        t_name = step.get('technique_name', '').strip()
        t_id = "Unknown"
        
        # (기존 기술 ID 매핑 로직 유지...)
        lookup = t_name.lower()
        if lookup in db.get('name_to_id', {}):
            t_id = db['name_to_id'][lookup]
        else:
            # 기술 이름 유사도 매칭 (기존 로직)
            matches = difflib.get_close_matches(lookup, db.get('tech_names_list', []), n=1, cutoff=0.6)
            if matches:
                t_id = db['name_to_id'][matches[0]]
                t_name = db['techniques'][t_id]['name']

        # ==================================================================
        # [NEW] Surface & Vector 강제 교정 로직 (Strict Validation)
        # ==================================================================
        raw_surface = step.get('surface', 'Internal Asset').strip()
        raw_vector = step.get('vector', 'Network').strip()

        # 1. Surface 교정
        if raw_surface in valid_surface_names:
            s_val = raw_surface
        else:
            # 유사한 것이 있으면 그걸로 대체, 없으면 리스트의 첫 번째 값이나 기본값 사용
            matches = difflib.get_close_matches(raw_surface, valid_surface_names, n=1, cutoff=0.4)
            if matches:
                s_val = matches[0]
                print(f"  🔧 Surface 교정: '{raw_surface}' -> '{s_val}'")
            else:
                s_val = "Vulnerability" # 기본값 (JSON에 있는 값 중 하나여야 함)
                print(f"  ⚠️ Surface 매칭 실패: '{raw_surface}' -> 기본값 '{s_val}' 적용")

        # 2. Vector 교정
        if raw_vector in valid_vector_names:
            v_val = raw_vector
        else:
            matches = difflib.get_close_matches(raw_vector, valid_vector_names, n=1, cutoff=0.4)
            if matches:
                v_val = matches[0]
                print(f"  🔧 Vector 교정: '{raw_vector}' -> '{v_val}'")
            else:
                v_val = "Technical" # 기본값
                print(f"  ⚠️ Vector 매칭 실패: '{raw_vector}' -> 기본값 '{v_val}' 적용")
        # ==================================================================

        used_surfaces.add(s_val)
        used_vectors.add(v_val)

        tactic_name = step.get('tactic', 'Unknown')
        if tactic_name.lower() == "initial access":
            initial_access_info['surface'] = s_val
            initial_access_info['vector'] = v_val

        # (KEV, CAPEC 매핑 로직은 기존 유지...)
        matched_kevs = []
        if t_id != "Unknown":
            assoc_cves = db['techniques'][t_id].get('associated_cves', [])
            valid_kevs = db.get('kev_set', set()).intersection(set(assoc_cves))
            for cve in list(valid_kevs)[:2]:
                cve_data = db['kev'].get(cve, {})
                cwe_val = cve_data.get('cwes', 'Unknown')
                short_desc = cve_data.get('shortDescription', 'No description')
                matched_kevs.append(f"{cve} [{cwe_val}]: {short_desc}")
        
        if not matched_kevs:
            if tactic_name in ["Initial Access", "Execution"] and db.get('kev'):
                rand_cve = random.choice(list(db['kev'].keys()))
                r_data = db['kev'].get(rand_cve, {})
                r_desc = r_data.get('shortDescription', 'No description')
                matched_kevs.append(f"{rand_cve} (Suggested): {r_desc}")
            else:
                matched_kevs = ["N/A"]

        capec_txt = "N/A"
        if 'rag_model' in db and t_id != "Unknown":
            q_vec = db['rag_model'].encode([f"{t_id} {t_name}"])
            _, idx = db['rag_index'].search(np.array(q_vec).astype('float32'), 1)
            doc = db['capec_data']['documents'][idx[0][0]]
            capec_txt = doc[:300]

        enriched.append({
            "tactic": tactic_name,
            "technique": t_name,
            "technique_id": t_id,
            "surface": s_val,   # 교정된 값
            "vector": v_val,    # 교정된 값
            "kev_info": matched_kevs,
            "capec_info": capec_txt,
            "rationale": step.get('reason', '')
        })
    
    return enriched, list(used_surfaces), list(used_vectors), initial_access_info

# --- Step 3: XML 생성 (Writer) ---
def generate_xml_with_local_llm(config, enriched_data, surfaces, vectors, initial_info, db, model, tokenizer):
    print("✍️ [Step 3] XML 작성 (Temperature 0.1로 고정)...")
    
    # Dossier 구성
    dossier = f"""
    <threat_intelligence_document>
        <metadata>
            <target_industry>{config['INDUSTRY']}</target_industry>
            <complexity>{config['COMPLEXITY_LEVEL']}</complexity>
            <attacker_skill>{config['ATTACK_LEVEL']}</attacker_skill>
            <campaign>{config['CAMPAIGN_NAME']}</campaign>
            <initial_access_summary>
                <surface>{initial_info['surface']}</surface>
                <vector>{initial_info['vector']}</vector>
            </initial_access_summary>
        </metadata>
        <attack_chain_plan>{json.dumps(enriched_data, indent=2)}</attack_chain_plan>
        <used_assets_summary>
            <surfaces>{json.dumps(surfaces)}</surfaces>
            <vectors>{json.dumps(vectors)}</vectors>
        </used_assets_summary>
    </threat_intelligence_document>
    """
    
    prompt = f"""
    Generate the <Scenario> XML based on the dossier.
    
    [STRICT RULES]
    1. For `<kev>`, use the exact string provided in the 'kev_info' field of the dossier.
       - If 'kev_info' says "N/A", write `<kev>N/A</kev>`.
       - DO NOT generate new CVEs.
    2. For `<capec>`, use the 'capec_info' string.
    3. Ensure `<target_and_entry>` matches `<initial_access_summary>`.
    
    Dossier:
    {dossier}
    """
    
    # [중요] 시스템 프롬프트를 SYSTEM_PROMPT_STRICT로 교체하여 환각 억제
    combined_system_prompt = db.get('system_prompt', "") + "\n\n" + SYSTEM_PROMPT_STRICT
    
    messages = [
        {"role": "system", "content": combined_system_prompt}, 
        {"role": "user", "content": prompt}
    ]
    
    # [설정] Temperature를 0.1로 아주 낮게 설정하여 팩트 유지
    return query_local_llm(model, tokenizer, messages, temperature=0.1)

# --- 저장 및 메인 ---
def save_scenario(content, config, current_model_name):
    def sanitize(name):
        return re.sub(r'[^a-zA-Z0-9_\-]', '', str(name).replace(" ", "_"))

    save_dir = os.path.join(
        "Output_test",
        sanitize(current_model_name.split('/')[-1]),
        sanitize(config['INDUSTRY']),
        sanitize(config['COMPLEXITY_LEVEL']),
    )
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{sanitize(config['ATTACK_LEVEL'])}_{timestamp}.xml"
    file_path = os.path.join(save_dir, filename)  # 🔹 폴더 + 파일 이름 합치기

    xml_match = re.search(r'(<Scenario.*?</Scenario>)', content, re.DOTALL | re.IGNORECASE)
    to_save = xml_match.group(1) if xml_match else content

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(to_save)

    print(f"저장 완료: {file_path}")

def run_pipeline(config, resources, model, tokenizer, current_model_name):
    try:
        config = auto_fill_config(config, resources)
        print(f"\n▶ 시작: {config['INDUSTRY']} | {config['COMPLEXITY_LEVEL']} | {config['ATTACK_LEVEL']}")
        
        plan = []
        for i in range(3):
            plan = plan_attack_with_local_llm(config, resources, model, tokenizer)
            if plan: break
            print("  -> 기획 재시도...")
        
        if plan:
            enriched, used_surfs, used_vecs, initial_info = enrich_plan(plan, resources)
            xml_output = generate_xml_with_local_llm(config, enriched, used_surfs, used_vecs, initial_info, resources, model, tokenizer)
            save_scenario(xml_output, config, current_model_name)
        else:
            print("❌ 기획 실패")
    except Exception as e:
        print(f"❌ 오류: {e}")
        traceback.print_exc()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # sys.stdout = DualLogger(f"execution_log_{timestamp}.txt")

    MODELS_TO_RUN = [
# "meta-llama/Llama-3.1-8B-Instruct",
# "mistralai/Mistral-Nemo-Instruct-2407",
# "Qwen/Qwen3-14B",
# "Qwen/Qwen3-4B-Thinking-2507",
"microsoft/phi-4",
# "microsoft/Phi-4-reasoning-plus",
# "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        ] # 원하는 모델 선택
    
    MODE = "SINGLE"
    # MODE = "BATCH"

    resources = load_resources(FILE_PATHS)
    if resources:
        if MODE == "SINGLE":
            target_model = MODELS_TO_RUN[0]
            llm_model, llm_tokenizer = load_local_llm(target_model)
            conf = {
                "INDUSTRY": "Energy",
                "COMPLEXITY_LEVEL": "Complex", 
                "ATTACK_LEVEL": "High", 
                "CAMPAIGN_NAME": ""}
            run_pipeline(conf, resources, llm_model, llm_tokenizer, target_model)
        elif MODE == "BATCH":
            # (배치 로직은 기존과 동일)
            pass