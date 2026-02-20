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
import difflib
from TD_IDF import ContextExtractor

# --- 1. 모델 설정 ---

# 8B 모델들
llama3= "meta-llama/Llama-3.1-8B-Instruct"
qwen3_8b = "Qwen/Qwen3-8B"
foundation = "fdtn-ai/Foundation-Sec-8B-Instruct" #  Foundation AI at Cisco
deepseek_8b = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# 12B 모델들
mistral_n= "mistralai/Mistral-Nemo-Instruct-2407" 
nvidia_nemo_nano = "nvidia/NVIDIA-Nemotron-Nano-12B-v2"
# gemma3 = "google/gemma-3-12b-it" # system prompt occurs

# 14B 모델들
qwen3 = "Qwen/Qwen3-14B"
deepseek_d = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
phi4 = "microsoft/phi-4" 

MODEL_NAME = deepseek_8b

# ==============================================================
FILE_PATHS = {
    "cti_data": "preprocessed_cti_data.json",
    "attack_db": "mitre_techniques_db.json",
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

context_extractor = None

SYSTEM_PROMPT_FORCE_TAGS = """
You are a cybersecurity analyst.

### CRITICAL XML OUTPUT RULES

1. **Root Structure**:
   - The output must start with `<Scenario>` and end with `</Scenario>`.
   - Include a simple `<metadata>` section at the top.

2. **Stage Structure (Strict)**:
   Inside `<attack_flow>`, create a `<stage>` block for every step.
   The `name` attribute of `<stage>` must be the **Tactic Name**.
   
   Inside each stage, you **MUST** include these tags in this order:
   
   - `<technique_name>`: The specific technique name.
   - `<surface>`: The target asset or object (External or Internal).
   - `<vector>`: The method or mechanism used.
   - `<kev>`: **CVE ID [CWE] : Short Description**. Use `<kev>N/A</kev>` if empty.
   - `<capec>`: CAPEC ID/Name. Use `<capec>N/A</capec>` if empty.
   - `<description>`: A detailed narrative of the attack step.

   **Example:**
   <stage name="Privilege Escalation">
       <technique_name>Txxxx</technique_name>
       <surface>OS Process</surface>
       <vector>Memory Injection</vector>
       <kev>N/A</kev>
       <capec>CAPEC-163: DLL Injection</capec>
       <description>The attacker injects malicious code...</description>
   </stage>

**Output ONLY the XML block.**
"""

# --- 모델 로드 ---
def load_local_llm(model_name):
    print(f"\n로컬 모델 로드: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()


def query_local_llm(model, tokenizer, messages, max_new_tokens=5000, temperature=0, top_p=0.9, top_k=50):
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
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    if "<think>" in response:
        # re.DOTALL: 줄바꿈(\n)이 포함된 내용까지 모두 매칭
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    return response

# --- 리소스 로드 ---
def load_resources(paths):
    print("데이터 로드 중...")
    global context_extractor
    db = {}
    try:
        # 설정 파일 로드
        with open(paths["attack_levels"], 'r', encoding='utf-8') as f: db['levels'] = json.load(f)
        with open(paths["complexity"], 'r', encoding='utf-8') as f: db['complexity'] = json.load(f)
        with open(paths["surfaces"], 'r', encoding='utf-8') as f: db['surfaces'] = json.load(f)
        with open(paths["vectors"], 'r', encoding='utf-8') as f: db['vectors'] = json.load(f)
        with open(paths["campaign_data"], 'r', encoding='utf-8') as f: db['campaigns'] = json.load(f)
        
        # prompt-gemini.txt 파일 읽기
        if os.path.exists(paths["prompt_txt"]):
            with open(paths["prompt_txt"], 'r', encoding='utf-8') as f:
                db['system_prompt'] = f.read()
            print("프롬프트 파일 로드 완료")
        else:
            print("프롬프트 파일을 찾을 수 없습니다.")

        db['techniques'] = {}

        # CTI 데이터
        if os.path.exists(paths["cti_data"]):
            with open(paths["cti_data"], 'r', encoding='utf-8') as f:
                cti = json.load(f)
                db['techniques'] = cti.get('techniques', {})
                db['name_to_id'] = {v['name'].lower(): k for k, v in db['techniques'].items()}
        else: db['techniques'] = {}

        # attack 데이터
        if os.path.exists(paths["attack_db"]):
            print(f"   -> MITRE DB 병합 중(Smart Merge): {paths['attack_db']}")
            with open(paths["attack_db"], 'r', encoding='utf-8') as f:
                mitre_db = json.load(f)
                
                # [중요] 덮어쓰지 않고 필요한 필드만 추가/갱신
                for t_id, new_data in mitre_db.items():
                    if t_id in db['techniques']:
                        # 이미 있는 기술이면: Data Source만 주입 (기존 CVE 보존)
                        db['techniques'][t_id]['data_sources'] = new_data.get('data_sources', [])
                        
                        # (선택) Description이나 Name도 최신 MITRE 것으로 업데이트하고 싶다면:
                        # db['techniques'][t_id]['description'] = new_data.get('description', '')
                    else:
                        # 없는 기술이면: 통째로 추가
                        db['techniques'][t_id] = new_data
        else:
            print(f"경고: {paths['attack_db']} 파일을 찾을 수 없습니다.")
        
        # C. Name-to-ID 매핑 생성
        db['name_to_id'] = {v['name'].lower(): k for k, v in db['techniques'].items()}
        db['tech_names_list'] = list(db['name_to_id'].keys())
        
        print(f"✅ 총 {len(db['techniques'])}개 기술(Technique) 로드 완료")
        
        # 검색용 이름 리스트 생성 (유사도 매칭용)
        db['tech_names_list'] = list(db['name_to_id'].keys())

        if os.path.exists(paths["kev_data"]):
            try: df = pd.read_csv(paths["kev_data"], encoding='utf-8')
            except: df = pd.read_csv(paths["kev_data"], encoding='latin-1')
            key_col = 'cveID' if 'cveID' in df.columns else 'CVE ID'
            if key_col in df.columns:
                df[key_col] = df[key_col].astype(str).str.strip()
                db['kev'] = df.set_index(key_col).to_dict(orient='index')
                db['kev_set'] = set(db['kev'].keys())
            else: db['kev'] = {}
        
        # RAG 데이터
        if os.path.exists(paths["faiss_index"]):
            db['rag_index'] = faiss.read_index(paths["faiss_index"])
            with open(paths["capec_pkl"], "rb") as f: db['capec_data'] = pickle.load(f)
            db['rag_model'] = SentenceTransformer(paths["embed_model"])

        global context_extractor
        if db['techniques']:
            try:
                context_extractor = ContextExtractor(db['techniques'])
                print("✅ TF-IDF ContextExtractor 초기화 완료")
            except Exception as e:
                print(f"TF-IDF 초기화 실패: {e}")

        return db
        
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None
    
# --- 캠페인 자동 선택 ---
def auto_fill_config(config, db):
    """설정값이 비어있으면('') 랜덤으로 채워넣음"""
    print("설정을 확인하고 빈 값을 자동으로 채웁니다...")

    # 1. 산업군 (목록이 없으면 기본 5대 인프라 사용)
    if not config.get("INDUSTRY"):
        industries = [
            "Communications", "Energy", "Healthcare and Public Health", 
            "Transportation Systems", "Water and Wastewater Systems"
        ]
        config["INDUSTRY"] = random.choice(industries)
        print(f"  -> [Auto] Industry: {config['INDUSTRY']}")

    # 2. 복잡도
    if not config.get("COMPLEXITY_LEVEL"):
        # complexity.json 키가 있으면 사용, 없으면 기본값
        options = list(db.get('complexity', {}).keys()) or ["Simple", "Intermediate", "Complex"]
        config["COMPLEXITY_LEVEL"] = random.choice(options)
        print(f"  -> [Auto] Complexity: {config['COMPLEXITY_LEVEL']}")

    # 3. 공격 수준
    if not config.get("ATTACK_LEVEL"):
        options = list(db.get('levels', {}).keys()) or ["Basic", "Skilled", "Expert"]
        config["ATTACK_LEVEL"] = random.choice(options)
        print(f"  -> [Auto] Attack Level: {config['ATTACK_LEVEL']}")

    # 4. 캠페인 이름
    if not config.get("CAMPAIGN_NAME"):
        campaign_names = [obj.get('name') for obj in db['campaigns'].get('objects', []) if obj.get('type') == 'campaign']
        if campaign_names:
            config["CAMPAIGN_NAME"] = random.choice(campaign_names)
            print(f"  -> [Auto] Campaign: {config['CAMPAIGN_NAME']}")
        else:
            config["CAMPAIGN_NAME"] = "" # 데이터 없으면 공란 유지

    return config

# --- 기획 (Planner) ---
def plan_attack_with_local_llm(config, db, model, tokenizer):
    print(f"[Step 1] 기획 ({config['INDUSTRY']} | {config['ATTACK_LEVEL']})...")
    req_tactics = db['complexity'].get(config['COMPLEXITY_LEVEL'], {}).get('tactics', [])
    attacker_desc = db['levels'].get(config['ATTACK_LEVEL'], {}).get('description', '')
    
    campaign_context = ""
    if config['CAMPAIGN_NAME']:
        for obj in db['campaigns'].get('objects', []):
            if obj.get('name') == config['CAMPAIGN_NAME']:
                campaign_context = f"Simulate campaign '{obj['name']}'. Desc: {obj.get('description', '')[:300]}..."
                break

    prompt = f"""
    You are a Cyber Attack Architect. Plan a realistic attack chain.
    
    Constraints:
    1. Target: {config['INDUSTRY']}
    2. Attacker Level: {config['ATTACK_LEVEL']} ({attacker_desc})
    3. Complexity: {config['COMPLEXITY_LEVEL']}
    4. Required Tactics: {json.dumps(req_tactics)}
    5. Context: {campaign_context}

    Output strictly a JSON List of objects.
    Example:
    [
        {{"tactic": "Initial Access", "technique_name": "Phishing", "reason": "Entry point"}}
    ]
    """
    messages = [{"role": "system", "content": "Output ONLY JSON."}, {"role": "user", "content": prompt}]
    response = query_local_llm(model, tokenizer, messages, max_new_tokens=8192, temperature=0.4, top_p=0.95 ,top_k=60) # 창의성 확보를 위한 유연성 조정
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

def infer_internal_context(description, technique_name):
    """
    기술 설명(Description)과 이름(Name)을 분석하여 내부 공격 표면(Surface)과 벡터(Vector)를 추론합니다.
    Ref: MITRE ATT&CK Data Sources, Software (S-ID), CAPEC Mechanisms
    """
    # 분석 대상 텍스트 (소문자 변환)
    text_to_analyze = (description + " " + technique_name).lower()
    
    # =========================================================
    # 1. Surface 추론 (공격 대상 객체)
    # =========================================================
    
    # [1-1] Pre-Attack Phase (정찰 및 자원 개발) 키워드
    pre_attack_map = {
        # Reconnaissance
        "repository": "Public Code Repository",
        "social media": "Social Media Platform",
        "employee": "Employee Identity",
        "search": "Open Source Intelligence (OSINT)",
        "scan": "Public IP/Network Block",
        "whois": "Domain Registry",
        "dns": "DNS Record",
        "public": "Public Facing Application",
        "blog": "Public Media",
        
        # Resource Development
        "domain": "Domain Name Registrar",
        "vps": "Virtual Private Server (VPS)",
        "server": "Command & Control Server",
        "certificate": "SSL/TLS Certificate",
        "account": "Compromised Account",
        "botnet": "Botnet Infrastructure",
        "malware": "Malware Artifact",
        "payload": "Malicious Payload",
        "tool": "Hacking Tool"
    }

    # [1-2] Post-Compromise Phase
    internal_map = {
        # 핵심 OS 객체
        "registry": "Windows Registry",
        "access token": "OS Access Token",
        "token": "OS Access Token",
        "service": "Windows Service",
        "process": "OS Process",
        "driver": "Kernel Driver",
        "firmware": "System Firmware",
        "wmi": "WMI Object",
        "scheduled task": "Scheduled Task",
        "job": "Scheduled Task",
        "boot": "Boot Sector",
        
        # 계정 및 권한
        "credential": "Credential Store",
        "password": "Credential Store",
        "group policy": "Active Directory",
        "ad ": "Active Directory",
        "domain controller": "Active Directory",
        "logon": "Logon Session",
        
        # 파일 및 스크립트
        "powershell": "Command Interpreter (PowerShell)",
        "cmd": "Command Interpreter (CMD)",
        "bash": "Command Interpreter (Bash)",
        "script": "Script File",
        "file": "File System",
        "directory": "File System",
        "dll": "Dynamic Link Library (DLL)",
        "binary": "Executable File",
        "image": "System Image",
        
        # 네트워크 및 기타
        "browser": "Web Browser",
        "network": "Network Traffic",
        "connection": "Network Connection",
        "remote": "Remote Service",
        "cloud": "Cloud Instance",
        "container": "Container"
    }
    
    # 맵 병합 (Pre-Attack 키워드가 우선순위를 갖도록 배치 가능)
    full_keyword_map = {**pre_attack_map, **internal_map}
    
    # Surface 매핑 실행
    inferred_surface = None
    for key, val in full_keyword_map.items():
        if key in text_to_analyze:
            inferred_surface = val
            break
            
    # 매칭 실패 시 기본값 처리 (단계별 일반화)
    if not inferred_surface:
        if any(x in text_to_analyze for x in ["recon", "gather", "identify"]):
            inferred_surface = "Public Information Source"
        elif any(x in text_to_analyze for x in ["development", "acquire", "buy"]):
            inferred_surface = "Attacker Infrastructure"
        else:
            inferred_surface = "Target System Artifact"

    # =========================================================
    # 2. Vector 추론 (공격 수행 수단/행위)
    # =========================================================
    
    # Pre-Attack 전용 행위 검색 (정찰/자원개발 특화)
    pre_attack_vectors = {
        "scan": "Active Scanning",
        "search": "OSINT Search",
        "buy": "Asset Acquisition",
        "purchase": "Asset Acquisition",
        "compromise": "System Compromise",
        "create": "Resource Development",
        "develop": "Malware Development",
        "register": "DNS/Domain Registration"
    }

    for key, val in pre_attack_vectors.items():
        if key in text_to_analyze:
             return inferred_surface, val

    # 구체적인 도구(Tools) 검색 (MITRE Software S-ID 기반)
    tools_map = { # preprocessed 참조
        "powershell": "PowerShell Script",
        "cmd": "Windows Command Shell",
        "bash": "Bash Script",
        "mimikatz": "Mimikatz Tool",
        "cobalt strike": "Cobalt Strike Beacon",
        "psexec": "PsExec Utility",
        "wmi": "WMI Query",
        "reg": "Reg.exe Utility",
        "schtasks": "Task Scheduler Utility",
        "dll": "DLL Sideloading",
        "macros": "Office Macros",
        "python": "Python Script"
    }
    
    for key, val in tools_map.items():
        if key in text_to_analyze:
            return inferred_surface, val  # 도구가 명확하면 즉시 반환
            
    # 일반 행위 메커니즘(Mechanism) 검색 (CAPEC 기반)
    mechanisms_map = {
        "injection": "Injection",
        "overflow": "Buffer Overflow",
        "brute force": "Brute Force Attempt",
        "modify": "Configuration Modification",
        "delete": "File Deletion",
        "encrypt": "Data Encryption",
        "steal": "Artifact Theft",
        "hook": "API Hooking",
        "masquerade": "Naming Masquerading",
        "impersonate": "Token Impersonation",
        "discovery": "System Discovery",
        "collection": "Data Collection"
    }
    
    for key, val in mechanisms_map.items():
        if key in text_to_analyze:
            return inferred_surface, val 
    
    inferred_vector = f"Execution via {technique_name.split(' ')[0]}"
    
    return inferred_surface, inferred_vector

# --- 데이터 보강 (Enricher) ---

def enrich_plan(plan, db):
    print("\n[Step 2] 데이터 보강 (Hybrid Semantic Mapping)...")
    enriched = []
    global context_extractor
    
    # 검증용 리스트
    valid_surfaces = [s['NAME'] for s in db.get('surfaces', [])]
    valid_vectors = [v['NAME'] for v in db.get('vectors', [])]
    all_tech_names = list(db.get('name_to_id', {}).keys())

    for step in plan:
        if isinstance(step, str) or not isinstance(step, dict): continue

        t_name = step.get('technique_name', '').strip()
        t_id = "Unknown"

        # 1. T-ID 매핑
        lookup = t_name.lower()
        if lookup in db.get('name_to_id', {}):
            t_id = db['name_to_id'][lookup]
        elif all_tech_names:
             matches = difflib.get_close_matches(lookup, all_tech_names, n=1, cutoff=0.5)
             if matches:
                 t_id = db['name_to_id'][matches[0]]
                 t_name = db['techniques'][t_id]['name']

        display_tech_name = f"{t_id}: {t_name}" if t_id != "Unknown" else t_name

        # -----------------------------------------------------------
        # 2. Surface & Vector 결정 (Hybrid Logic)
        # -----------------------------------------------------------
        tactic_name = step.get('tactic', 'Unknown')
        
        if tactic_name.lower() == "initial access":
            # [Case A] Initial Access: 기존 로직 (LLM + List Validation)
            llm_surface = step.get('surface', 'External Asset')
            llm_vector = step.get('vector', 'Delivery Method')
            
            # (이전 답변의 강제 보정 코드 삽입)
            s_match = difflib.get_close_matches(llm_surface, valid_surfaces, n=1, cutoff=0.4)
            s_val = s_match[0] if s_match else llm_surface
            
            v_match = difflib.get_close_matches(llm_vector, valid_vectors, n=1, cutoff=0.4)
            v_val = v_match[0] if v_match else llm_vector

            if s_val == llm_surface and llm_surface not in valid_surfaces:
                 # 너무 모호하면 추론
                 desc = db['techniques'].get(t_id, {}).get('description', '') if t_id != "Unknown" else ""
                 inferred_s, _ = infer_internal_context(desc, t_name) # 키워드 기반
                 if inferred_s != "Target System Artifact": s_val = inferred_s
                 else: s_val = "External Interface" # 최후의 수단

        else:
            # [Case B] Internal Stages: STIX + CAPEC + TF-IDF 결합
            
            # 기본값 설정
            s_val = "Target System Artifact"
            v_val = t_name

            # (1) 데이터 준비
            tech_info = db['techniques'].get(t_id, {})
            stix_data_sources = tech_info.get('data_sources', [])
            description = tech_info.get('description', '')
            
            # (2) Surface 결정 (STIX -> TF-IDF 보정)
            if stix_data_sources:
                # STIX Data Source 파싱 (예: "Process: Process Creation")
                raw_source = stix_data_sources[0]
                stix_surface = raw_source.split(':')[0].strip()
                
                # [논문 포인트] STIX가 너무 일반적이면 TF-IDF로 구체화 시도
                if stix_surface in ["File", "Process", "Command"] and context_extractor:
                    tfidf_s, _ = context_extractor.infer_surface_vector(t_id, t_name)
                    # TF-IDF 결과가 더 구체적이면 교체 (예: File -> Config File)
                    if "Artifact" not in tfidf_s: 
                        s_val = tfidf_s
                    else:
                        s_val = stix_surface # TF-IDF 실패 시 STIX 유지
                else:
                    s_val = stix_surface
            else:
                # STIX 없음 -> TF-IDF 전적으로 의존
                if context_extractor:
                    s_val, _ = context_extractor.infer_surface_vector(t_id, t_name)
                else:
                    s_val, _ = infer_internal_context(description, t_name)

            # (3) Vector 결정 (Tool Keywords -> CAPEC -> STIX Component)
            # 3-1. Description에서 구체적 도구(Tool) 검색 (최우선)
            _, tool_vector = infer_internal_context(description, t_name)
            if "Script" in tool_vector or "Tool" in tool_vector or "Utility" in tool_vector:
                v_val = tool_vector
            else:
                # 3-2. CAPEC Mechanism 활용 (Preprocessed 데이터 활용)
                # CAPEC ID가 있으면 메커니즘 가져오기 (DB 구조에 따라 다름)
                # 여기서는 Preprocessed 데이터에 'capec_id'가 있다고 가정하거나 
                # RAG로 찾은 CAPEC 이름을 활용
                if 'rag_model' in db: 
                     # (아래 CAPEC 검색 로직에서 가져온 이름 활용 가능)
                     pass 
                
                # 3-3. STIX Data Component 활용 (예: Process Creation)
                if stix_data_sources and v_val == t_name:
                    if ":" in stix_data_sources[0]:
                        v_val = stix_data_sources[0].split(':')[1].strip() + " (Mechanism)"
                    else:
                        v_val = f"{t_name} (Canonical Execution)"

        # -----------------------------------------------------------
        # 3. CAPEC & KEV 매핑 (기존 유지)
        # -----------------------------------------------------------
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
                matched_kevs = []

        capec_txt = "N/A"
        if 'rag_model' in db and t_id != "Unknown":
            # RAG 검색
            q_vec = db['rag_model'].encode([f"{t_id} {t_name}"])
            _, idx = db['rag_index'].search(np.array(q_vec).astype('float32'), 1)
            doc = db['capec_data']['documents'][idx[0][0]]
            capec_txt = doc[:300]

        enriched.append({
            "tactic": tactic_name,
            "technique": display_tech_name,
            "surface": s_val,
            "vector": v_val,
            "kev_ids": matched_kevs,
            "capec_data": capec_txt,
            "rationale": step.get('reason', '')
        })
    
    return enriched

# --- XML 생성 (Writer) ---
def generate_xml_with_local_llm(config, enriched_data, db, model, tokenizer):
    print("[Step 3] XML 작성...")
    
    dossier = f"""
    <threat_intelligence_document>
        <metadata>
            <target_industry>{config['INDUSTRY']}</target_industry>
            <complexity>{config['COMPLEXITY_LEVEL']}</complexity>
            <attacker_skill>{config['ATTACK_LEVEL']}</attacker_skill>
            <campaign>{config['CAMPAIGN_NAME']}</campaign>
        </metadata>
        
        <attack_chain_plan>
        {json.dumps(enriched_data, indent=2)}
        </attack_chain_plan>
    </threat_intelligence_document>
    """
    prompt = f"""
    Generate the <Scenario> XML based on the dossier.
    Focus on the <attack_flow> with correct Surface/Vector for each stage.
    
    Dossier:
    {dossier}
    """
    
    combined_system_prompt = db.get('system_prompt', "") + "\n\n" + SYSTEM_PROMPT_FORCE_TAGS
    
    messages = [
        {"role": "system", "content": combined_system_prompt}, 
        {"role": "user", "content": prompt}
    ]
    
    return query_local_llm(model, tokenizer, messages, temperature=0.1, top_p=0.9, top_k=50, max_new_tokens=8192)

# --- 저장 및 메인 ---
def save_scenario(content, config, model_name):
    def sanitize(name): 
        return re.sub(r'[^a-zA-Z0-9_\-]', '', str(name).replace(" ", "_"))

    # 1. 폴더 경로: Output / 모델명 / 산업 / 복잡도
    save_dir = os.path.join("Output_1212", 
                            sanitize(model_name.split('/')[-1]), 
                            sanitize(config['INDUSTRY']), 
                            sanitize(config['COMPLEXITY_LEVEL']))
    
    os.makedirs(save_dir, exist_ok=True)

    # 2. 파일 이름: 난이도_타임스탬프.xml
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{sanitize(config['ATTACK_LEVEL'])}_{timestamp}.xml"
    
    filepath = os.path.join(save_dir, filename)
    
    # XML 태그만 추출해서 저장
    xml_match = re.search(r'(<Scenario.*?</Scenario>)', content, re.DOTALL | re.IGNORECASE)
    to_save = xml_match.group(1) if xml_match else content
    
    with open(filepath, "w", encoding="utf-8") as f: 
        f.write(to_save)
    
    print(f"\n저장 완료: {filepath}")


def run_pipeline(config, resources, model, tokenizer, current_model_name):
    try:
        # 1. 정보 자동 선택 (비어있으면)
        config = auto_fill_config(config, resources)
        print(f"\n▶ 시작: {config['INDUSTRY']} | {config['COMPLEXITY_LEVEL']} | {config['ATTACK_LEVEL']}")
        
        # 2. 기획 (Plan)
        MAX_RETRIES = 3 # 최대 재시도 횟수
        plan = []
        
        for attempt in range(MAX_RETRIES):
            print(f"기획 시도 ({attempt+1}/{MAX_RETRIES})...")
            plan = plan_attack_with_local_llm(config, resources, model, tokenizer)
            
            if plan:
                print("성공. 다음 단계로 진행")
                break # 성공하면 루프 탈출
            else:
                print(f"  -> 기획 실패 (JSON 파싱 오류 등). 재시도 진행")
        
        if plan:
            # [수정] 반환값 1개만 받음
            enriched = enrich_plan(plan, resources)
            
            # [수정] 인자 줄여서 호출
            xml_output = generate_xml_with_local_llm(
                config, enriched, resources, model, tokenizer
            )
            
            save_scenario(xml_output, config, current_model_name)
        else:
            print("ERROR - 기획 단계에서 실패")
            
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        traceback.print_exc()
    
    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    MODE = "SINGLE"
    # MODE = "BATCH" # "BATCH": 45개 생성 모드
         
    # 1. 모델 및 리소스 로드 
    llm_model, llm_tokenizer = load_local_llm(MODEL_NAME)
    resources = load_resources(FILE_PATHS)

    if resources:
        if MODE == "SINGLE":
            print("\n[모드] 단일 시나리오 생성을 시작합니다.")
            
            # 원하는 설정을 여기에 직접 입력하세요
            single_config = {
                "INDUSTRY": "Energy",          # 타겟 산업
                "COMPLEXITY_LEVEL": "Complex", # Simple, Intermediate, Complex
                "ATTACK_LEVEL": "Expert",        # Basic, Skilled, Expert
                "CAMPAIGN_NAME": ""            # 비워두면 랜덤 선택
            }
            
            run_pipeline(single_config, resources, llm_model, llm_tokenizer, current_model_name=MODEL_NAME)
            print("\n단일 생성 완료.")

        elif MODE == "BATCH":
            TARGET_INDUSTRIES = [
                # "Communications",
                # "Energy", 
                "Healthcare and Public Health", 
                # "Transportation Systems",
                # "Water and Wastewater Systems"
            ]
            TARGET_COMPLEXITIES = [
                # "Simple",
                # "Intermediate",
                "Complex"
                ]
            TARGET_ATTACK_LEVELS = [
                "Basic", 
                # "Skilled", 
                "Expert"
                ]

            total_scenarios = len(TARGET_INDUSTRIES) * len(TARGET_COMPLEXITIES) * len(TARGET_ATTACK_LEVELS)
            count = 0

            print(f"\n[모드] 배치 생성을 시작합니다. (총 {total_scenarios}개)")

            for industry in TARGET_INDUSTRIES:
                for complexity in TARGET_COMPLEXITIES:
                    for attack_level in TARGET_ATTACK_LEVELS:
                        count += 1
                        print(f"진행률: [{count}/{total_scenarios}]")
                        
                        batch_config = {
                            "INDUSTRY": industry,
                            "COMPLEXITY_LEVEL": complexity,
                            "ATTACK_LEVEL": attack_level,
                            "CAMPAIGN_NAME": "" 
                        }
                        run_pipeline(batch_config, resources, llm_model, llm_tokenizer, current_model_name=MODEL_NAME)
            print("전체 배치 생성 완료.")
        else:
            print("잘못된 모드 설정. 'SINGLE' 또는 'BATCH'를 선택해주세요.")