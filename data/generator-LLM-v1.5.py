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
import sys

# --- [설정] 로그 파일 저장용 클래스 ---
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

# --- 1. 모델 설정 ---
llama3= "meta-llama/Llama-3.1-8B-Instruct"
mistral_nemo= "mistralai/Mistral-Nemo-Instruct-2407"
qwen3 = "Qwen/Qwen3-14B"
gemma2 = "google/gemma-2-9b-it"
phi4 = "microsoft/Phi-4-reasoning-plus"
deepseek_d = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

MODEL_NAME = phi4  # 사용할 모델

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

# --- 태그 강제 시스템 프롬프트 ---
SYSTEM_PROMPT_FORCE_TAGS = """
You are a cybersecurity analyst.

### CRITICAL XML OUTPUT RULES

1. **Header Consistency**:
   - The `<target_and_entry>` section MUST match the Surface/Vector defined in the **'Initial Access'** stage.

2. **Stage Structure (Strict)**:
   For every step in the attack chain, create a `<stage>` block with the `name` attribute set to the **Tactic Name**.
   
   Inside the stage, you **MUST** include the following tags in this **EXACT ORDER**:
   
   - `<technique_name>`: The specific technique name (e.g., "Phishing").
   - `<surface>`: The Attack Surface used.
   - `<vector>`: The Attack Vector used.
   - `<kev>`: The CVE ID & Name. If none, output `<kev>N/A</kev>`.
   - `<capec>`: The CAPEC ID/Name. If none, output `<capec>N/A</capec>`.
   - `<description>`: A detailed narrative of the attack step.

   **Example:**
   <stage name="Reconnaissance">
       <technique_name>Gather Victim Organization Information</technique_name>
       <surface>Internet-facing assets</surface>
       <vector>Web</vector>
       <kev>CVE-xxxx-xxxx</kev>
       <capec>CAPEC-652: Identify Software/Services Used by Target</capec>
       <description>The attacker performs deep reconnaissance...</description>
   </stage>

3. **Reference Lists**:
   - In `<used_assets_summary>`, list ONLY the items actually used in the scenario.

**Output ONLY the XML block.**
"""

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

def extract_json_array(response: str):
    import re, json

    # 1) ```json ... ``` 블록이 있으면 그 안만 사용
    code_match = re.search(r"```json(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if code_match:
        text = code_match.group(1).strip()
        return json.loads(text)

    # 2) 아니면 JSON 배열처럼 보이는 것들을 '비탐욕'으로 전부 찾고, 마지막 것만 사용
    candidates = re.findall(r'\[\s*{.*?}\s*\]', response, re.DOTALL)
    if not candidates:
        return []

    for text in reversed(candidates):  # 뒤에서부터 시도
        try:
            return json.loads(text)
        except Exception:
            continue

    return []


def query_local_llm(model, tokenizer, messages, max_new_tokens=4096, temperature=0.1, top_p=0.9, top_k=50):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # [NEW] <think> 태그 제거 로직 (Phi-4, DeepSeek-R1 등 추론 모델 대응)
    # <think>...</think> 블록을 찾아서 공백으로 치환합니다.
    clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
    return clean_response

# --- 리소스 로드 (수정됨: 컬럼 공백 제거) ---
def load_resources(paths):
    print("데이터 로드 중...")
    db = {}
    try:
        with open(paths["attack_levels"], 'r', encoding='utf-8') as f: db['levels'] = json.load(f)
        with open(paths["complexity"], 'r', encoding='utf-8') as f: db['complexity'] = json.load(f)
        with open(paths["surfaces"], 'r', encoding='utf-8') as f: db['surfaces'] = json.load(f)
        with open(paths["vectors"], 'r', encoding='utf-8') as f: db['vectors'] = json.load(f)
        with open(paths["campaign_data"], 'r', encoding='utf-8') as f: db['campaigns'] = json.load(f)
        
        if os.path.exists(paths["prompt_txt"]):
            with open(paths["prompt_txt"], 'r', encoding='utf-8') as f:
                db['system_prompt'] = f.read()
            print("프롬프트 파일 로드 완료")
        else:
            db['system_prompt'] = "You are a cybersecurity analyst."
            print("프롬프트 파일을 찾을 수 없습니다.")

        if os.path.exists(paths["cti_data"]):
            with open(paths["cti_data"], 'r', encoding='utf-8') as f:
                cti = json.load(f)
                db['techniques'] = cti.get('techniques', {})
                db['name_to_id'] = {v['name'].lower(): k for k, v in db['techniques'].items()}
        else:
            db['techniques'] = {}

        # [중요] KEV 로드 시 컬럼명 공백 제거
        if os.path.exists(paths["kev_data"]):
            try: df = pd.read_csv(paths["kev_data"], encoding='utf-8')
            except: df = pd.read_csv(paths["kev_data"], encoding='latin-1')
            
            df.columns = df.columns.str.strip() # 컬럼명 공백 제거
            
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
        print(f"데이터 로드 실패: {e}")
        return None

# --- 설정 자동 완성 ---
def auto_fill_config(config, db):
    if not config.get("INDUSTRY"):
        industries = ["Communications", "Energy", "Healthcare and Public Health", "Transportation Systems", "Water and Wastewater Systems"]
        config["INDUSTRY"] = random.choice(industries)
        print(f"  -> [Auto] Industry: {config['INDUSTRY']}")

    if not config.get("COMPLEXITY_LEVEL"):
        options = list(db.get('complexity', {}).keys()) or ["Simple", "Intermediate", "Complex"]
        config["COMPLEXITY_LEVEL"] = random.choice(options)
        print(f"  -> [Auto] Complexity: {config['COMPLEXITY_LEVEL']}")

    if not config.get("ATTACK_LEVEL"):
        options = list(db.get('levels', {}).keys()) or ["Basic", "Skilled", "Expert"]
        config["ATTACK_LEVEL"] = random.choice(options)
        print(f"  -> [Auto] Attack Level: {config['ATTACK_LEVEL']}")

    if not config.get("CAMPAIGN_NAME"):
        campaign_names = [obj.get('name') for obj in db['campaigns'].get('objects', []) if obj.get('type') == 'campaign']
        if campaign_names:
            config["CAMPAIGN_NAME"] = random.choice(campaign_names)
            print(f"  -> [Auto] Campaign: {config['CAMPAIGN_NAME']}")
        else:
            config["CAMPAIGN_NAME"] = ""
    return config

# --- 기획 (Planner) ---
def plan_attack_with_local_llm(config, db, model, tokenizer):
    print(f"\n🧠 [Step 1] 기획 ({config['INDUSTRY']} | {config['ATTACK_LEVEL']})...")
    req_tactics = db['complexity'].get(config['COMPLEXITY_LEVEL'], {}).get('tactics', [])
    attacker_desc = db['levels'].get(config['ATTACK_LEVEL'], {}).get('description', '')
    
    avail_surfaces = [s['NAME'] for s in db['surfaces']]
    avail_vectors = [v['NAME'] for v in db['vectors']]
    
    campaign_context = ""
    if config['CAMPAIGN_NAME']:
        for obj in db['campaigns'].get('objects', []):
            if obj.get('name') == config['CAMPAIGN_NAME']:
                desc = obj.get('description', '')[:500]
                campaign_context = f"Reference Campaign: '{obj['name']}'. Style/TTPs: {desc}"
                break

    prompt = f"""
    You are a Creative Cyber Attack Architect. 
    Plan a **NEW, HYPOTHETICAL** attack chain.
    
    Constraints:
    1. Target: {config['INDUSTRY']}
    2. Level: {config['ATTACK_LEVEL']} ({attacker_desc})
    3. Required Tactics: {json.dumps(req_tactics)}
    4. Reference: {campaign_context if campaign_context else "Generic APT style"}
    
    [Rules]
    - Do NOT copy historical events exactly. Adapt TTPs to the new target.
    - Return a LIST of JSON OBJECTS. (No strings!)

    [Options]
    - Surfaces: {json.dumps(avail_surfaces[:10])}...
    - Vectors: {json.dumps(avail_vectors[:10])}...

    Example Output:
    [
        {{
            "tactic": "Initial Access", 
            "technique_name": "Spearphishing Link", 
            "surface": "Email", 
            "vector": "Social Engineering",
            "reason": "Initial entry"
        }}
    ]
    """
    messages = [{"role": "system", "content": "Output ONLY JSON."}, {"role": "user", "content": prompt}]
    
    # 창의성 높임 (온도 0.8)
    response = query_local_llm(model, tokenizer, messages, temperature=0.8, top_p=0.95, top_k=60)
    plan = extract_json_array(response)
    response = plan
    # print(response)
    return response    
    # try:
    #     match = re.search(r'\[.*\]', response, re.DOTALL)
    #     return json.loads(match.group(0)) if match else []
    # except: return []

# --- 데이터 보강 (Enricher) [수정됨: 방어 코드 및 .get 사용] ---
def enrich_plan(plan, db):
    print("🔍 [Step 2] 데이터 보강 및 필터링")
    enriched = [] 
    used_surfaces = set()
    used_vectors = set()
    initial_access_info = {"surface": "Unknown", "vector": "Unknown"}

    for step in plan:
        # [방어] 문자열이 들어오면 파싱 시도
        if isinstance(step, str):
            print(f"⚠️ 경고: 문자열 데이터 발견 -> {step}")
            # 임시 복구 (파싱 불가 시 기본값)
            step = {"technique_name": step, "tactic": "Unknown", "surface": "Unknown", "vector": "Unknown"}

        # [방어] 딕셔너리가 아니면 건너뜀
        if not isinstance(step, dict):
            continue

        t_name = step.get('technique_name', 'Unknown')
        t_id = ""
        
        lookup = t_name.lower()
        if lookup in db.get('name_to_id', {}):
            t_id = db['name_to_id'][lookup]
        
        s_val = step.get('surface', 'Internal Asset')
        v_val = step.get('vector', 'Network')
        used_surfaces.add(s_val)
        used_vectors.add(v_val)

        tactic_name = step.get('tactic', 'Unknown')
        if tactic_name.lower() == "initial access":
            initial_access_info['surface'] = s_val
            initial_access_info['vector'] = v_val

        # KEV 매칭 (안전하게 .get 사용)
        matched_kevs = []
        if t_id != "Unknown":
            assoc_cves = db['techniques'][t_id].get('associated_cves', [])
            valid_kevs = db.get('kev_set', set()).intersection(set(assoc_cves))
            
            for cve in list(valid_kevs)[:2]:
                cve_data = db['kev'].get(cve, {})
                cwe_val = cve_data.get('cwes', 'Unknown')
                vuln_name = cve_data.get('vulnerabilityName', 'Unknown')
                matched_kevs.append(f"{cve} [{cwe_val}] ({vuln_name})")
        
        if not matched_kevs:
            if tactic_name in ["Initial Access", "Execution"] and db.get('kev'):
                rand_cve = random.choice(list(db['kev'].keys()))
                matched_kevs.append(f"{rand_cve} (Suggested)")
            else:
                matched_kevs = [] 

        capec_txt = "N/A"
        if 'rag_model' in db:
            q_vec = db['rag_model'].encode([f"{t_id} {t_name}"])
            _, idx = db['rag_index'].search(np.array(q_vec).astype('float32'), 1)
            doc = db['capec_data']['documents'][idx[0][0]]
            capec_txt = doc[:300]

        enriched.append({
            "tactic": tactic_name,
            "technique": t_name,
            "technique_id": t_id,
            "surface": s_val,
            "vector": v_val,
            "kev_ids": matched_kevs,
            "capec_data": capec_txt,
            "rationale": step.get('reason', '')
        })
        
    return enriched, list(used_surfaces), list(used_vectors), initial_access_info

# --- XML 생성 (Writer) ---
def generate_xml_with_local_llm(config, enriched_data, surfaces, vectors, initial_info, db, model, tokenizer):
    print("✍️ [Step 3] XML 작성...")
    
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
    Dossier:
    {dossier}
    """
    
    # 프롬프트 파일 내용 + 강제 태그 규칙
    combined_system_prompt = db.get('system_prompt', "") + "\n\n" + SYSTEM_PROMPT_FORCE_TAGS
    
    messages = [
        {"role": "system", "content": combined_system_prompt}, 
        {"role": "user", "content": prompt}
    ]
    
    # 정확성 높임 (온도 0.1)
    return query_local_llm(model, tokenizer, messages, temperature=0.1, top_k=40, top_p=0.9)

# --- 저장 ---
def save_scenario(content, config, model_name):
    def sanitize(name): return re.sub(r'[^a-zA-Z0-9_\-]', '', str(name).replace(" ", "_"))
    save_dir = os.path.join("Output", sanitize(model_name.split('/')[-1]), sanitize(config['INDUSTRY']), sanitize(config['COMPLEXITY_LEVEL']))
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{sanitize(config['ATTACK_LEVEL'])}_{timestamp}.xml"
    
    xml_match = re.search(r'(<Scenario.*?</Scenario>)', content, re.DOTALL | re.IGNORECASE)
    to_save = xml_match.group(1) if xml_match else content
    
    with open(filepath, "w", encoding="utf-8") as f: 
        f.write(to_save)
    print(f"✅ 저장 완료: {filepath}")

# --- 실행 파이프라인 ---
def run_pipeline(config, resources, model, tokenizer):
    try:
        config = auto_fill_config(config, resources)
        print(f"\n▶ 시작: {config['INDUSTRY']} | {config['COMPLEXITY_LEVEL']} | {config['ATTACK_LEVEL']}")
        
        MAX_RETRIES = 3
        plan = []
        for attempt in range(MAX_RETRIES):
            print(f"🧠 [Step 1] 기획 시도 ({attempt+1}/{MAX_RETRIES})...")
            plan = plan_attack_with_local_llm(config, resources, model, tokenizer)
            if plan:
                print("  -> 기획 성공!")
                break
            else:
                print(f"  -> ⚠️ 기획 실패 (파싱 오류). 재시도...")
        
        if plan:
            # 4개 값 받기 (정상 작동)
            enriched, used_surfs, used_vecs, initial_info = enrich_plan(plan, resources)
            xml_output = generate_xml_with_local_llm(config, enriched, used_surfs, used_vecs, initial_info, resources, model, tokenizer)
            save_scenario(xml_output, config, MODEL_NAME)
        else:
            print("❌ 최종 기획 실패.")

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()

# --- 메인 ---
if __name__ == "__main__":
    # 로그 파일 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"execution_log_{timestamp}.txt"
    sys.stdout = DualLogger(log_file)
    print(f"✅ 로그 저장 시작: {log_file}")

    MODE = "SINGLE" # or "BATCH"

    llm_model, llm_tokenizer = load_local_llm(MODEL_NAME)
    resources = load_resources(FILE_PATHS)

    if resources:
        if MODE == "SINGLE":
            print("\n🚀 [SINGLE MODE]")
            single_config = {
                "INDUSTRY": "Communications",
                "COMPLEXITY_LEVEL": "Simple",
                "ATTACK_LEVEL": "Basic",
                "CAMPAIGN_NAME": ""
            }
            run_pipeline(single_config, resources, llm_model, llm_tokenizer)

        elif MODE == "BATCH":
            print("\n🚀 [BATCH MODE]")
            TARGET_INDUSTRIES = ["Communications", "Energy", "Healthcare and Public Health", "Transportation Systems", "Water and Wastewater Systems"]
            TARGET_COMPLEXITIES = ["Simple", "Standard", "Complex"]
            TARGET_ATTACK_LEVELS = ["Low", "Medium", "High"] # or Basic/Skilled/Expert (match your json)

            count = 0
            total = len(TARGET_INDUSTRIES) * len(TARGET_COMPLEXITIES) * len(TARGET_ATTACK_LEVELS)
            
            for ind in TARGET_INDUSTRIES:
                for comp in TARGET_COMPLEXITIES:
                    for lvl in TARGET_ATTACK_LEVELS:
                        count += 1
                        print(f"\n⏳ 진행률 [{count}/{total}]")
                        batch_config = {"INDUSTRY": ind, "COMPLEXITY_LEVEL": comp, "ATTACK_LEVEL": lvl, "CAMPAIGN_NAME": ""}
                        run_pipeline(batch_config, resources, llm_model, llm_tokenizer)
    
    print("\n✅ 모든 작업 완료.")