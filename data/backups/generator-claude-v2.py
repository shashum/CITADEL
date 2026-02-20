import os
import json
import re
import pandas as pd
import pickle
import faiss
import numpy as np
import anthropic
import random  # 랜덤 선택을 위해 추가
from sentence_transformers import SentenceTransformer
from datetime import datetime

# --- API 설정 ---
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
client = anthropic.Anthropic(api_key=api_key)
MODEL_NAME = "claude-opus-4-1-20250805" # 사용할 LLM 모델 이름 설정
            # claude-sonnet-4-5-20250929

# --- 파일 경로 ---
FILE_PATHS = {
    "cti_data": "preprocessed_cti_data.json",
    "kev_data": "known_exploited_vulnerabilities.csv",
    "campaign_data": "campaign-merge.json",
    "attack_levels": "attack_levels.json",
    "complexity": "complexity.json",
    "surfaces": "attack_surface_en.json",
    "vectors": "attack_vector_en.json",
    "prompt_txt": "prompt-gemini.txt",
    "faiss_index": "capec_index.faiss",
    "capec_pkl": "capec_data.pkl",
    "embed_model": "all-MiniLM-L6-v2"
}

# --- 1. 리소스 로드 ---
def load_resources(paths):
    print("📚 [System] 데이터 파일을 로드 중입니다...")
    db = {}
    try:
        with open(paths["attack_levels"], 'r', encoding='utf-8') as f: db['levels'] = json.load(f)
        with open(paths["complexity"], 'r', encoding='utf-8') as f: db['complexity'] = json.load(f)
        with open(paths["surfaces"], 'r', encoding='utf-8') as f: db['surfaces'] = json.load(f)
        with open(paths["vectors"], 'r', encoding='utf-8') as f: db['vectors'] = json.load(f)
        with open(paths["campaign_data"], 'r', encoding='utf-8') as f: db['campaigns'] = json.load(f)
        with open(paths["prompt_txt"], 'r', encoding='utf-8') as f: db['system_prompt'] = f.read()
        
        if os.path.exists(paths["cti_data"]):
            with open(paths["cti_data"], 'r', encoding='utf-8') as f:
                cti = json.load(f)
                db['techniques'] = cti.get('techniques', {})
                db['name_to_id'] = {v['name'].lower(): k for k, v in db['techniques'].items()}
        else: db['techniques'] = {}

        if os.path.exists(paths["kev_data"]):
            try: df = pd.read_csv(paths["kev_data"], encoding='utf-8')
            except: df = pd.read_csv(paths["kev_data"], encoding='latin-1')
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

# --- [NEW] 설정 자동 완성 함수 ---
def auto_fill_config(config, db):
    print("\n [Config] 설정값을 확인하고 빈 항목을 자동으로 채웁니다...")
    
    # 1. 산업군 (목록이 따로 없으면 기본 리스트 사용)
    if not config.get("INDUSTRY"):
        industries = [
            "Communications",
            "Energy",
            "Healthcare and Public Health",
            "Transportation Systems",
            "Water and Wastewater Systems"
        ]
        config["INDUSTRY"] = random.choice(industries)
        print(f"  -> Industry: {config['INDUSTRY']}")

    # 2. 복잡도 (complexity.json 키 기반)
    if not config.get("COMPLEXITY_LEVEL"):
        options = list(db['complexity'].keys())
        config["COMPLEXITY_LEVEL"] = random.choice(options)
        print(f"  -> Complexity 선택됨: {config['COMPLEXITY_LEVEL']}")

    # 3. 공격 난이도 (attack_levels.json 키 기반)
    if not config.get("ATTACK_LEVEL"):
        options = list(db['levels'].keys())
        config["ATTACK_LEVEL"] = random.choice(options)
        print(f"  -> Attack Level 선택됨: {config['ATTACK_LEVEL']}")

    # 4. 캠페인 이름 (campaign-merge.json 기반)
    if not config.get("CAMPAIGN_NAME"):
        # type이 campaign인 객체들의 이름만 추출
        campaign_names = [
            obj.get('name') for obj in db['campaigns'].get('objects', []) 
            if obj.get('type') == 'campaign'
        ]
        if campaign_names:
            config["CAMPAIGN_NAME"] = random.choice(campaign_names)
            print(f"  -> Campaign 선택됨: {config['CAMPAIGN_NAME']}")
        else:
            config["CAMPAIGN_NAME"] = "" # 데이터 없으면 빈칸 유지

    return config

# --- 2. LLM 기획 ---
def plan_attack_with_llm(config, db):
    print(f"\n🧠 [Step 1] '{config['CAMPAIGN_NAME']}' 스타일로 공격 기획 중...")
    
    req_tactics = db['complexity'].get(config['COMPLEXITY_LEVEL'], {}).get('tactics', [])
    attacker_desc = db['levels'].get(config['ATTACK_LEVEL'], {}).get('description', '')
    
    # 캠페인 컨텍스트 생성
    campaign_context = ""
    if config['CAMPAIGN_NAME']:
        for obj in db['campaigns'].get('objects', []):
            if obj.get('name') == config['CAMPAIGN_NAME']:
                campaign_context = f"Emulate the specific behaviors/TTPs of the campaign '{obj['name']}'. Description: {obj.get('description', '')}"
                break

    prompt = f"""
    You are a Cyber Attack Architect. Plan a realistic attack chain.

    Constraints:
    1. Target: {config['INDUSTRY']}
    2. Attacker Level: {config['ATTACK_LEVEL']} ({attacker_desc})
    3. Complexity: {config['COMPLEXITY_LEVEL']}
    4. Required Tactics: {json.dumps(req_tactics)}
    5. Campaign Style: {campaign_context}

    Task:
    Select specific MITRE ATT&CK techniques for each tactic.
    Return a JSON List of objects: [{{"tactic": "...", "technique_name": "...", "reason": "..."}}]
    """

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=5000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        match = re.search(r'\[.*\]', message.content[0].text, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except Exception as e:
        print(f"❌ Planning Error: {e}")
        return []

# --- 3. 데이터 보강 ---
def enrich_plan(plan, db):
    print("\n🔍 [Step 2] 세부 데이터(KEV, CAPEC) 매핑 중...")
    enriched = []
    avail_surfaces = [s['NAME'] for s in db['surfaces']]
    avail_vectors = [v['NAME'] for v in db['vectors']]
    
    for step in plan:
        t_name = step.get('technique_name', '')
        t_id = "Unknown"
        
        lookup = t_name.lower()
        if lookup in db.get('name_to_id', {}):
            t_id = db['name_to_id'][lookup]
        
        # KEV 매칭
        matched_kevs = []
        if t_id != "Unknown":
            assoc_cves = db['techniques'][t_id].get('associated_cves', [])
            valid_kevs = db.get('kev_set', set()).intersection(set(assoc_cves))
            for cve in list(valid_kevs)[:2]:
                matched_kevs.append(f"ID: {cve}, Name: {db['kev'][cve].get('vulnerabilityName')}")
        
        # KEV 랜덤 추천 (중요 단계인데 없을 경우)
        if not matched_kevs and step['tactic'] in ["Initial Access", "Execution"] and db.get('kev'):
            rand_cve = random.choice(list(db['kev'].keys()))
            matched_kevs.append(f"Suggested: {rand_cve} (Use if fits)")

        # CAPEC (RAG)
        capec_txt = "No specific pattern."
        if 'rag_model' in db:
            q_vec = db['rag_model'].encode([f"{t_id} {t_name}"])
            _, idx = db['rag_index'].search(np.array(q_vec).astype('float32'), 1)
            doc = db['capec_data']['documents'][idx[0][0]]
            match = re.search(r"Description:(.*?)(?=\n\w+:)", doc, re.DOTALL)
            capec_txt = match.group(1).strip() if match else doc[:300]

        enriched.append({
            "tactic": step['tactic'],
            "technique": t_name,
            "technique_id": t_id,
            "kev_info": matched_kevs,
            "capec_info": capec_txt,
            "rationale": step.get('reason', '')
        })
    return enriched, avail_surfaces, avail_vectors

# --- 4. 시나리오 생성 ---
def generate_xml_scenario(config, enriched_data, surfaces, vectors, db):
    print("\n✍️ [Step 3] 시나리오 XML 작성 중...")
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
    try:
        msg = client.messages.create(
            model=MODEL_NAME,
            max_tokens=5000,
            temperature=0.1,
            system=db['system_prompt'],
            messages=[{"role": "user", "content": f"Document:\n{dossier}"}]
        )
        return msg.content[0].text
    except Exception as e: return f"Error: {e}"

# --- [NEW] 결과 파일 저장 함수 ---
def save_scenario(content, config, model_name):
    # 1. 경로에 사용할 이름들을 안전하게 변환 (공백 -> 언더바, 특수문자 제거)
    def sanitize(name):
        return re.sub(r'[^a-zA-Z0-9_\-]', '', name.replace(" ", "_"))

    safe_model = sanitize(model_name)
    safe_industry = sanitize(config['INDUSTRY'])
    safe_complexity = sanitize(config['COMPLEXITY_LEVEL'])
    safe_level = sanitize(config['ATTACK_LEVEL'])
    
    # 2. 폴더 구조 생성: Output / 모델명 / 산업 / 복잡도 / 난이도
    save_dir = os.path.join("Output", safe_model, safe_industry, safe_complexity, safe_level)
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Scenario_{timestamp}.xml"
    filepath = os.path.join(save_dir, filename)
    
    # 4. 저장
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ 파일 저장 완료!")
    print(f"   📂 경로: {filepath}")

# --- 메인 실행 ---
if __name__ == "__main__":
    # 사용자가 원하는 설정 (비워두면 자동 선택됨)
    USER_CONFIG = {
        "INDUSTRY": "Healthcare and Public Health",          # 예: "Healthcare" (비워두면 랜덤)
                # "Communications",
                # "Energy", 
                # "Healthcare and Public Health", 
                # "Transportation Systems",
                # "Water and Wastewater Systems"

        "COMPLEXITY_LEVEL": "Simple",  # 예: "Standard" (비워두면 랜덤)
                # "Simple",
                # "Standard",
                # "Complex"

        "ATTACK_LEVEL": "High",      # 예: "High" (비워두면 랜덤)
                # "Low",
                # "Medium",
                # "High"

        "CAMPAIGN_NAME": ""      # 예: "Lazarus Group" (비워두면 랜덤)
    }

    # 1. 데이터 로드
    resources = load_resources(FILE_PATHS)
    
    if resources:
        # 2. 설정 자동 완성 (빈 값 채우기)
        final_config = auto_fill_config(USER_CONFIG, resources)
        
        # 3. 기획 -> 보강 -> 생성 프로세스
        plan = plan_attack_with_llm(final_config, resources)
        if plan:
            enriched, surfs, vecs = enrich_plan(plan, resources)
            scenario_xml = generate_xml_scenario(final_config, enriched, surfs, vecs, resources)
            
            # 4. 폴더 구조에 맞춰 저장
            save_scenario(scenario_xml, final_config, MODEL_NAME)