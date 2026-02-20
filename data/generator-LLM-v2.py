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

# --- 1. 모델 설정 ---
llama3= "meta-llama/Llama-3.1-8B-Instruct"
mistral_n= "mistralai/Mistral-Nemo-Instruct-2407"
qwen3 = "Qwen/Qwen3-14B"
# gemma2 = "google/gemma-2-9b-it" # System role unsupported
# phi4 = "microsoft/phi-4"
deepseek_d = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

MODEL_NAME = deepseek_d  

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
# SYSTEM_PROMPT_FORCE_RULESt = """

# ### CRITICAL XML OUTPUT RULES

# 1. **Header Consistency**:
#    - The `<target_and_entry>` section MUST match the Surface/Vector defined in the **'Initial Access'** stage.

# 2. **Stage Structure (Strict)**:
#    Inside EVERY `<stage>` block, you **MUST** include exactly these tags in this order:
   
#    - `<technique_name>`: The specific technique name.
#    - `<surface>`: The Attack Surface used.
#    - `<vector>`: The Attack Vector used.
#    - `<kev>`: **CVE ID [CWE] : Short Description**. (e.g., "CVE-2021-44228 [CWE-502]: Apache Log4j2 JNDI features used in configuration..."). Use `<kev>N/A</kev>` if empty.
#    - `<capec>`: CAPEC ID/Name. Use `<capec>N/A</capec>` if empty.
#    - `<description>`: A detailed narrative of the attack step.

# 3. **Reference Lists**:
#    - In `<used_assets_summary>`, list ONLY the items actually used in the scenario.

# **Output ONLY the XML block.**
# """
SYSTEM_PROMPT_FORCE_RULES = """
You are a cybersecurity analyst.

### CRITICAL XML OUTPUT RULES
Inside every `<stage>` block, you **MUST** include exactly these three tags:

1. **`<kev>`**: The CVE ID and Name. Use `<kev></kev>` if empty.
2.  - `<technique_name>`: The specific technique name. (e.g., "Txxx": Spearphishing Attatchment) 
    - `<surface>`: Use data from "surfaces",
    - `<vector>`: Use data from "vectors",
    - `<kev>`: **CVE ID [CWE] : Short Description**. (e.g., "CVE-2021-44228 [CWE-502]: Apache Log4j2 JNDI features used in configuration..."). Use `<kev>N/A</kev>` if empty.
    - `<capec>`: CAPEC ID/Name. Use `<capec>N/A</capec>` if empty.
3. **`<description>`**: Detailed explanation of the attack action.

### OUTPUT STRUCTURE:
<Scenario>
    <overview>...</overview>
    <target_and_entry>...</target_and_entry>
    <attack_flow>
        <stage name="Tactic Name">
             <technique_name>...</technique_name>
             <surface>...</surface>
             <vector>...</vector>
             <kev>...</kev>
             <capec>...</capec>
             <description>...</description>
        </stage>
    </attack_flow>
    <summary>...</summary>
</Scenario>

**NEVER omit `<description>`, `<kev>`, or `<capec>`.**
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

def query_local_llm(model, tokenizer, messages, max_new_tokens=3000, temperature=0, top_p=0.9, top_k=50):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,top_p=top_p, top_k=top_k,
            do_sample=True if temperature > 0 else False, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1
            )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# --- 리소스 로드 ---
def load_resources(paths):
    print("데이터 로드 중...")
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

        # CTI 데이터
        if os.path.exists(paths["cti_data"]):
            with open(paths["cti_data"], 'r', encoding='utf-8') as f:
                cti = json.load(f)
                db['techniques'] = cti.get('techniques', {})
                db['name_to_id'] = {v['name'].lower(): k for k, v in db['techniques'].items()}
        else: db['techniques'] = {}

        # KEV 데이터
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
    response = query_local_llm(model, tokenizer, messages, max_new_tokens=3000, temperature=0.3, top_p=0.9 ,top_k=60)
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

# --- 데이터 보강 (Enricher) ---
def enrich_plan(plan, db):
    """
    [Step 2] 데이터 보강 함수
    LLM이 기획한 공격 단계(plan)에 실제 데이터(KEV, CAPEC)를 매핑하여 상세 정보를 추가합니다.
    """
    print("[Step 2] 데이터 보강 (빈 데이터 처리 및 RAG 검색 포함)...")
    
    enriched = []
    
    # 나중에 LLM이 XML을 작성할 때 "이 중에서 골라라"고 던져줄 전체 메뉴판(Reference List) 추출
    avail_surfaces = [s['NAME'] for s in db['surfaces']]
    avail_vectors = [v['NAME'] for v in db['vectors']]
    
    # 기획된 공격 체인의 각 단계(Step)를 순회
    for step in plan:
        t_name = step.get('technique_name', '') # 예: "Phishing"
        t_id = "Unknown"
        
        # -------------------------------------------------------
        # 1. 기술 이름 -> ID 매핑
        # -------------------------------------------------------
        # 대소문자 구분 없이 매칭하기 위해 소문자로 변환하여 검색
        lookup = t_name.lower()
        if lookup in db.get('name_to_id', {}):
            t_id = db['name_to_id'][lookup] # 예: "T1566"
        
        # -------------------------------------------------------
        # 2. KEV(알려진 악용 취약점) 매칭 로직
        # -------------------------------------------------------
        matched_kevs = []
        if t_id != "Unknown":
            # CTI 데이터에서 해당 기술(Technique)과 연관된 CVE 목록 가져오기
            assoc_cves = db['techniques'][t_id].get('associated_cves', [])
            
            # 전체 KEV 리스트와 교집합을 찾아 '실제 해커가 악용 중인 취약점'만 필터링
            valid_kevs = db.get('kev_set', set()).intersection(set(assoc_cves))
            
            # 결과가 너무 많으면 토큰 낭비이므로 최대 2개까지만 포맷팅하여 저장
            for cve in list(valid_kevs)[:2]:
                matched_kevs.append(f"{cve} ({db['kev'][cve].get('vulnerabilityName')})")
        
        # -------------------------------------------------------
        # 3. KEV 데이터가 없을 때의 처리 (중요!)
        # -------------------------------------------------------
        if not matched_kevs:
            # '초기 침투(Initial Access)'나 '실행(Execution)' 단계는 취약점이 중요한 단계임.
            # 따라서 매칭된 게 없으면 랜덤으로 KEV 하나를 '제안(Suggested)'하여
            # LLM이 시나리오를 쓸 때 참고할 수 있도록 강제 주입함.
            if step['tactic'] in ["Initial Access", "Execution"] and db.get('kev'):
                rand_cve = random.choice(list(db['kev'].keys()))
                matched_kevs.append(f"{rand_cve} (Suggested)")
            else:
                # 마땅한 방법이 없을 경우 빈 리스트 유지
                matched_kevs = [] 

        # -------------------------------------------------------
        # 4. CAPEC(공격 패턴) 검색 (RAG: 벡터 검색 활용)
        # -------------------------------------------------------
        capec_txt = ""
        if 'rag_model' in db:
            # 기술 ID와 이름을 쿼리로 변환하여 벡터 임베딩 생성
            q_vec = db['rag_model'].encode([f"{t_id} {t_name}"])
            
            # FAISS 인덱스에서 가장 유사도가 높은 문서 1개 검색
            _, idx = db['rag_index'].search(np.array(q_vec).astype('float32'), 1)
            
            # 인덱스로 실제 문서 내용 조회
            doc = db['capec_data']['documents'][idx[0][0]]
            
            # 텍스트가 너무 길면 LLM 입력 제한을 고려해 앞부분 300자만 추출
            capec_txt = doc[:300] 

        # -------------------------------------------------------
        # 5. 최종 보강된 데이터 조립
        # -------------------------------------------------------
        enriched.append({
            "tactic": step['tactic'],          # 전술명 (예: Initial Access)
            "technique": t_name,               # 기술명 (예: Phishing)
            "technique_id": t_id,              # 기술 ID (예: T1566)
            "kev_ids": matched_kevs,           # 매칭된 취약점 리스트 (없으면 빈 리스트)
            "capec_data": capec_txt,           # 검색된 공격 패턴 설명 텍스트
            "rationale": step.get('reason', '') # LLM이 기획 단계에서 적은 '이 기술을 쓴 이유'
        })
        
    # 보강된 계획 리스트와, LLM이 나중에 선택할 표면/벡터 리스트 반환
    return enriched, avail_surfaces, avail_vectors

# --- XML 생성 (Writer) ---
def generate_xml_with_local_llm(config, enriched_data, surfaces, vectors, db, model, tokenizer):
    print("[Step 3] XML 작성 및 프롬프트 조합")
    
    dossier = f"""
    <threat_intelligence_document>
        <metadata>
            <target_industry>{config['INDUSTRY']}</target_industry>
            <complexity>{config['COMPLEXITY_LEVEL']}</complexity>
            <attacker_skill>{config['ATTACK_LEVEL']}</attacker_skill>
            <campaign>{config['CAMPAIGN_NAME']}</campaign>
        </metadata>
        <attack_chain_plan>{json.dumps(enriched_data, indent=2)}</attack_chain_plan>
        <reference_lists>
            <surfaces>{json.dumps(surfaces)}</surfaces>
            <vectors>{json.dumps(vectors)}</vectors>
        </reference_lists>
    </threat_intelligence_document>
    """
    
    # 파일에서 읽은 프롬프트 + 수정된 강제 규칙 결합
    combined_system_prompt = db.get('system_prompt', "") + "\n\n" + SYSTEM_PROMPT_FORCE_RULES
    
    prompt = f"""
    Generate the <Scenario> XML based on the dossier.
    
    [SELECTION RULE]
    The <reference_lists> contains ALL possible options.
    You must **SELECT ONLY** the items relevant to the 'Initial Access' technique in this scenario.
    Do **NOT** output the full list.
    
    Dossier:
    {dossier}
    """
    file_prompt_content = db.get('system_prompt', "")
    combined_system_prompt = file_prompt_content + SYSTEM_PROMPT_FORCE_RULES
    
    messages = [
        {"role": "system", "content": combined_system_prompt}, 
        {"role": "user", "content": prompt}
    ]
    
    return query_local_llm(model, tokenizer, messages, max_new_tokens=5000, temperature=0.1, top_k=40, top_p=0.9)

# --- 저장 및 메인 ---
def save_scenario(content, config, model_name):
    def sanitize(name): 
        return re.sub(r'[^a-zA-Z0-9_\-]', '', str(name).replace(" ", "_"))

    # 1. 폴더 경로: Output / 모델명 / 산업 / 복잡도
    # (기존에는 여기에 ATTACK_LEVEL이 포함되어 있었으나 제거함)
    save_dir = os.path.join("Output_1203", 
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
    
    print(f"저장 완료: {filepath}")

# --- [NEW] 공통 실행 파이프라인 함수 (중복 코드 제거) ---
def run_pipeline(config, resources, model, tokenizer):
    try:
        
        # 1. 정보 자동 선택 (비어있으면)
        config = auto_fill_config(config, resources)
        print(f"\n▶ 시작: {config['INDUSTRY']} | {config['COMPLEXITY_LEVEL']} | {config['ATTACK_LEVEL']}")
        
        # 2. 기획 (Plan)
        MAX_RETRIES = 3
        plan = []
        
        for attempt in range(MAX_RETRIES):
            print(f"[Step 1] 기획 시도 ({attempt+1}/{MAX_RETRIES})...")
            plan = plan_attack_with_local_llm(config, resources, model, tokenizer)
            
            if plan:
                print("  -> 기획 성공!")
                break # 성공하면 루프 탈출
            else:
                print(f"  -> 기획 실패 (JSON 파싱 오류 등). 재시도 진행")
        
        if plan:
            # 3. 보강 (Enrich)
            enriched, surfs, vecs = enrich_plan(plan, resources)
            # 4. XML 생성 (Generate)
            xml_output = generate_xml_with_local_llm(config, enriched, surfs, vecs, resources, model, tokenizer)
            # 5. 저장 (Save)
            save_scenario(xml_output, config, MODEL_NAME)
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
    # MODE = "SINGLE"
    MODE = "BATCH" # "BATCH": 45개 생성 모드
         
    # 1. 모델 및 리소스 로드 
    llm_model, llm_tokenizer = load_local_llm(MODEL_NAME)
    resources = load_resources(FILE_PATHS)

    if resources:
        if MODE == "SINGLE":
            print("\n[모드] 단일 시나리오 생성을 시작합니다.")
            
            # 원하는 설정을 여기에 직접 입력하세요
            single_config = {
                "INDUSTRY": "",          # 타겟 산업
                "COMPLEXITY_LEVEL": "Intermediate", # Simple, Intermediate, Complex
                "ATTACK_LEVEL": "Expert",        # Basic, Skilled, Expert
                "CAMPAIGN_NAME": ""            # 비워두면 랜덤 선택
            }
            
            run_pipeline(single_config, resources, llm_model, llm_tokenizer)
            print("\n단일 생성 완료.")

        elif MODE == "BATCH":
            TARGET_INDUSTRIES = [
                "Communications",
                "Energy", 
                "Healthcare and Public Health", 
                "Transportation Systems",
                "Water and Wastewater Systems"
            ]
            TARGET_COMPLEXITIES = [
                "Simple",
                "Intermediate",
                "Complex"
                ]
            TARGET_ATTACK_LEVELS = [
                "Basic", 
                "Skilled", 
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
                        run_pipeline(batch_config, resources, llm_model, llm_tokenizer)
            
            print("전체 배치 생성 완료.")
        
        else:
            print("잘못된 모드 설정. 'SINGLE' 또는 'BATCH'를 선택해주세요.")



# if __name__ == "__main__":
#     TARGET_INDUSTRIES = ["Communications", "Energy", "Healthcare and Public Health", "Transportation Systems", "Water and Wastewater Systems"]
#     TARGET_COMPLEXITIES = ["Simple", "Intermediate", "Complex"]
#     TARGET_ATTACK_LEVELS = ["Low", "Medium", "High"]

#     llm_model, llm_tokenizer = load_local_llm(MODEL_NAME)
#     resources = load_resources(FILE_PATHS)

#     if resources:
#         total_scenarios = len(TARGET_INDUSTRIES) * len(TARGET_COMPLEXITIES) * len(TARGET_ATTACK_LEVELS)
#         count = 0
#         print(f"\n 시나리오 생성 시작 (총 {total_scenarios}개)")

#         for ind in TARGET_INDUSTRIES:
#             for comp in TARGET_COMPLEXITIES:
#                 for lvl in TARGET_ATTACK_LEVELS:
#                     count += 1
                    # print(f"\n⏳ [{count}/{total_scenarios}] {ind} | {comp} | {lvl}")
#                     conf = {"INDUSTRY": ind, "COMPLEXITY_LEVEL": comp, "ATTACK_LEVEL": lvl, "CAMPAIGN_NAME": ""}
#                     try:
#                         conf = auto_fill_campaign(conf, resources)
#                         plan = plan_attack_with_local_llm(conf, resources, llm_model, llm_tokenizer)
#                         if plan:
#                             enr, sur, vec = enrich_plan(plan, resources)
#                             xml = generate_xml_with_local_llm(conf, enr, sur, vec, resources, llm_model, llm_tokenizer)
#                             save_scenario(xml, conf, MODEL_NAME)
#                     except Exception as e:
#                         traceback.print_exc()
#                     gc.collect()
#                     torch.cuda.empty_cache()
#     print("시나리오 생성 완료.")