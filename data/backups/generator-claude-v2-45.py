import os
import json
import re
import pandas as pd
import pickle
import faiss
import numpy as np
import anthropic
import random
import time
import traceback
from sentence_transformers import SentenceTransformer
from datetime import datetime

# --- API 설정 ---
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
client = anthropic.Anthropic(api_key=api_key)
MODEL_NAME = "claude-opus-4-1-20250805" # 사용할 LLM 모델 이름 설정
# MODEL_NAME = "claude-sonnet-4-5-20250929"

# --- 파일 경로 ---
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

# --- [수정됨] 태그 강제 시스템 프롬프트 ---
SYSTEM_PROMPT_FORCE_TAGS = """
                        You are a cybersecurity analyst. Your goal is to generate a realistic XML attack scenario based on the provided dossier.

                        ### CRITICAL XML OUTPUT RULES

                        1. **Attack Surface & Vector Selection**:
                        - You are provided with lists of `Available Surfaces` and `Available Vectors` in the dossier.
                        - **DO NOT list all of them.**
                        - **SELECT ONLY 1 or 2 items** that strictly match the **'Initial Access'** technique used in your scenario.
                        - Example:
                            <target_and_entry>
                                <initial_attack_surface>Internet-facing assets</initial_attack_surface>
                                <initial_attack_vector>Web</initial_attack_vector>
                            </target_and_entry>

                        2. **Stage Tags (Mandatory)**:
                        Inside every `<stage>` block, you **MUST** include:
                        - `<kev>`: CVE ID & Name. Use `<kev></kev>` if empty.
                        - `<capec>`: CAPEC ID/Name. Use `<capec></capec>` if empty.
                        - `<description>`: Narrative each of the attack step.

                        3. **General**:
                        - Output ONLY the XML block.
                        """

# --- 리소스 로드 함수 (기존 동일) ---
def load_resources(paths):
    print("📚 [System] 데이터 로드 중...")
    db = {}
    try:
        with open(paths["attack_levels"], 'r', encoding='utf-8') as f: db['levels'] = json.load(f)
        with open(paths["complexity"], 'r', encoding='utf-8') as f: db['complexity'] = json.load(f)
        with open(paths["surfaces"], 'r', encoding='utf-8') as f: db['surfaces'] = json.load(f)
        with open(paths["vectors"], 'r', encoding='utf-8') as f: db['vectors'] = json.load(f)
        with open(paths["campaign_data"], 'r', encoding='utf-8') as f: db['campaigns'] = json.load(f)
        
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

def auto_fill_campaign(config, db):
    if not config.get("CAMPAIGN_NAME"):
        campaign_names = [obj.get('name') for obj in db['campaigns'].get('objects', []) if obj.get('type') == 'campaign']
        if campaign_names:
            config["CAMPAIGN_NAME"] = random.choice(campaign_names)
            print(f"  -> Campaign Auto-Selected: {config['CAMPAIGN_NAME']}")
    return config

# --- 3. Claude 기획 단계 (Planner) ---
def plan_attack_with_claude(config, db):
    print(f"\n🧠 [Step 1] Claude가 공격을 기획합니다 (Target: {config['INDUSTRY']} | Level: {config['ATTACK_LEVEL']})...")
    
    req_tactics = db['complexity'].get(config['COMPLEXITY_LEVEL'], {}).get('tactics', [])
    attacker_desc = db['levels'].get(config['ATTACK_LEVEL'], {}).get('description', '')
    
    campaign_context = ""
    if config['CAMPAIGN_NAME']:
        for obj in db['campaigns'].get('objects', []):
            if obj.get('name') == config['CAMPAIGN_NAME']:
                campaign_context = f"Simulate the campaign '{obj['name']}'. Description: {obj.get('description', '')[:300]}..."
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
        {{"tactic": "Initial Access", "technique_name": "Phishing", "reason": "Common entry point"}},
        {{"tactic": "Execution", "technique_name": "PowerShell", "reason": "Fileless attack"}}
    ]
    """

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=3000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        
        match = re.search(r'\[.*\]', response, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except Exception as e:
        print(f"❌ Planning Error: {e}")
        return []

# --- 4. 데이터 보강 (Enricher - Python 로직) ---
def enrich_plan(plan, db):
    print("🔍 [Step 2] 세부 데이터(KEV, CAPEC) 매핑 중...")
    enriched = []
    avail_surfaces = [s['NAME'] for s in db['surfaces']]
    avail_vectors = [v['NAME'] for v in db['vectors']]
    
    for step in plan:
        t_name = step.get('technique_name', '')
        t_id = "Unknown"
        
        lookup = t_name.lower()
        if lookup in db.get('name_to_id', {}):
            t_id = db['name_to_id'][lookup]
        
        matched_kevs = []
        if t_id != "Unknown":
            assoc_cves = db['techniques'][t_id].get('associated_cves', [])
            valid_kevs = db.get('kev_set', set()).intersection(set(assoc_cves))
            for cve in list(valid_kevs)[:2]:
                matched_kevs.append(f"ID: {cve}, Name: {db['kev'][cve].get('vulnerabilityName')}")
        
        # 중요 단계인데 KEV가 없으면 랜덤 추천
        if not matched_kevs and step['tactic'] in ["Initial Access", "Execution"] and db.get('kev'):
            rand_cve = random.choice(list(db['kev'].keys()))
            matched_kevs.append(f"Suggested: {rand_cve} (Use if applicable)")

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

# --- 5. Claude 시나리오 생성 (Writer) ---
def generate_xml_with_claude(config, enriched_data, surfaces, vectors, db):
    print("✍️  [Step 3] Claude가 XML 작성 (Surface/Vector 선별 적용)...")
    
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

    prompt = f"""
    Generate the <Scenario> XML based on the dossier.
    
    [IMPORTANT INSTRUCTION]
    - Look at the 'Initial Access' stage in the <attack_chain_plan>.
    - From the <reference_lists>, **PICK ONLY** the Surface and Vector that logically allow that Initial Access technique.
    - **DO NOT output the entire list.** Only the selected 1-2 items.
    - please write scenario description only in Korean.
    
    Dossier:
    {dossier}
    """

    file_prompt_content = db.get('system_prompt', "")
    combined_system_prompt = file_prompt_content + SYSTEM_PROMPT_FORCE_TAGS

    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            temperature=0.1,
            system=combined_system_prompt, # 수정된 시스템 프롬프트 적용
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {e}"

# --- 6. 저장 함수 ---
def save_scenario(content, config, model_name):
    def sanitize(name):
        return re.sub(r'[^a-zA-Z0-9_\-]', '', str(name).replace(" ", "_"))

    safe_model = sanitize(model_name.replace(":", ""))
    safe_industry = sanitize(config['INDUSTRY'])
    safe_complexity = sanitize(config['COMPLEXITY_LEVEL'])
    safe_level = sanitize(config['ATTACK_LEVEL'])
    
    # Output/claude-3-5.../Industry/Complexity/Level/
    save_dir = os.path.join("Output", safe_model, safe_industry, safe_complexity, safe_level)
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Scenario_{timestamp}.xml"
    filepath = os.path.join(save_dir, filename)
    
    # XML 태그만 추출
    xml_match = re.search(r'(<Scenario.*?</Scenario>)', content, re.DOTALL | re.IGNORECASE)
    to_save = xml_match.group(1) if xml_match else content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(to_save)
    
    print(f"✅ 저장 완료: {filepath}")

# --- 메인 실행 (45개 배치) ---
if __name__ == "__main__":
    # 1. 목표 설정 (45개 조합)
    TARGET_INDUSTRIES = [
        # "Communications",
        # "Energy", 
        "Healthcare and Public Health", 
        # "Transportation Systems",
        # "Water and Wastewater Systems"
    ]
    TARGET_COMPLEXITIES = [
                        #    "Simple", 
                        #    "Standard", 
                           "Complex"
                           ]
    TARGET_ATTACK_LEVELS = [
                            # "Low", 
                            # "Medium", 
                            "High"
                            ]

    # 2. 데이터 로드
    resources = load_resources(FILE_PATHS)

    if resources:
        total_scenarios = len(TARGET_INDUSTRIES) * len(TARGET_COMPLEXITIES) * len(TARGET_ATTACK_LEVELS)
        current_count = 0

        print(f"\n🚀 Claude API를 사용하여 총 {total_scenarios}개의 시나리오 생성을 시작합니다.")

        for industry in TARGET_INDUSTRIES:
            for complexity in TARGET_COMPLEXITIES:
                for attack_level in TARGET_ATTACK_LEVELS:
                    current_count += 1
                    print(f"\n" + "="*60)
                    print(f"⏳ 진행률: [{current_count}/{total_scenarios}]")
                    print(f"🎯 타겟: {industry} | 복잡도: {complexity} | 수준: {attack_level}")
                    print("="*60)

                    # 설정 구성
                    config = {
                        "INDUSTRY": industry,
                        "COMPLEXITY_LEVEL": complexity,
                        "ATTACK_LEVEL": attack_level,
                        "CAMPAIGN_NAME": "" # 아래 함수에서 랜덤 선택
                    }

                    try:
                        # 캠페인 랜덤 선택
                        config = auto_fill_campaign(config, resources)

                        # 1. 기획
                        plan = plan_attack_with_claude(config, resources)
                        
                        if plan:
                            # 2. 보강
                            enriched, surfs, vecs = enrich_plan(plan, resources)
                            
                            # 3. 생성
                            xml_output = generate_xml_with_claude(config, enriched, surfs, vecs, resources)
                            
                            # 4. 저장
                            save_scenario(xml_output, config, MODEL_NAME)
                            
                            # API 속도 제한 고려하여 잠시 대기
                            time.sleep(1)
                        else:
                            print("❌ 기획 단계 실패로 건너뜀")

                    except Exception as e:
                        print(f"❌ 생성 중 오류 발생: {e}")
                        traceback.print_exc()

    print("\n✅ 모든 시나리오 생성이 완료되었습니다.")