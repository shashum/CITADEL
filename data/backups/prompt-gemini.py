import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os
import json
import re
import gc
import random
from collections import defaultdict
from datetime import datetime
import copy
import pandas as pd

# --- 1. 설정 (Configuration) ---
llama3= "meta-llama/Llama-3.1-8B-Instruct"
mistral= "mistralai/Mistral-Nemo-Instruct-2407"
exaone = "LGAI-EXAONE/EXAONE-Deep-7.8B"
qwen3 = "Qwen/Qwen3-14B"
qwen3_think ="Qwen/Qwen3-4B-Thinking-2507"
gemma2 = "google/gemma-2-9b-it"
# phi4 = "microsoft/phi-4"
# deepseek_d = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

MODEL_NAME = mistral  # 사용할 LLM 모델 이름 설정

# --- 생성할 시나리오의 목표를 설정하세요 ---
SCENARIO_CONFIG = {
    "INDUSTRY": "", 
    "COMPLEXITY_LEVEL": "", 
    "ATTACK_LEVEL": "", 

    # - 값을 지정하면 해당 캠페인(들)을 기반으로 시나리오를 생성합니다. (예: ["Sandworm Team", "APT28"])
    # - 빈 리스트([])로 두면 LLM이 자동으로 캠페인을 추천하거나 기술 중심으로 생성합니다.
    "CAMPAIGN_NAMES": [],

    "FOCUS_TACTIC_NAME": "",
    "FOCUS_TECHNIQUE_ID": "",

    "ATTACK_SURFACE_NAME": "",
    "ATTACK_VECTOR_NAME": ""   
}

# --- 파일 경로 설정 ---
FILE_PATHS = {
    "prompt_template": "prompt-gemini.txt",
    "cti_data": "preprocessed_cti_data.json",
    "faiss_index": "capec_index.faiss",
    "capec_data": "capec_data.pkl",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "attack_levels": "attack_levels.json",
    "attack_surface_en": "attack_surface_en.json",
    "attack_vector_en": "attack_vector_en.json",
    "campaign-merge": "campaign-merge.json",
    "complexity": "complexity.json",
    "critical-infra-description": "critical-infra-description.json",
    "kev_data": "known_exploited_vulnerabilities.csv"
}

# --- 결과물 저장 폴더 ---
OUTPUT_DIR = "Output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


## [수정됨] 다중 캠페인 추천을 지원하도록 LLM 자동 완성 함수 수정
def complete_config_with_llm(config, all_data, model_name):
    """LLM을 호출하여 비어있는 설정 값을 자동으로 채우는 함수"""
    print("\n[Sub-Step] 비어있는 설정을 LLM을 통해 자동으로 완성합니다...")
    
    available_campaigns = list(all_data['groups_by_name'].keys())
    available_surfaces = [s['NAME'] for s in all_data['attack_surface_en']]
    available_vectors = [v['NAME'] for v in all_data['attack_vector_en']]
    
    prompt = f"""
            You are a cybersecurity intelligence expert. Your task is to complete a JSON configuration for a cyber attack scenario.
            Based on the user's required inputs, fill in the empty ("") or empty list ([]) values.

            **Required User Inputs:**
            - Target Industry: "{config['INDUSTRY']}"
            - Scenario Complexity: "{config['COMPLEXITY_LEVEL']}"
            - Attacker Skill Level: "{config['ATTACK_LEVEL']}"

            **Available Data Options (sample):**
            - Campaigns: {random.sample(available_campaigns, min(15, len(available_campaigns)))}
            - Attack Surfaces: {random.sample(available_surfaces, min(15, len(available_surfaces)))}
            - Attack Vectors: {random.sample(available_vectors, min(15, len(available_vectors)))}

            **User's Incomplete Configuration:**
            {json.dumps(config, indent=4)}

            **Instructions:**
            1. Analyze the inputs and choose the most plausible values to complete the configuration.
            2. For "CAMPAIGN_NAMES", select one or more relevant campaigns from the available options and return them as a JSON list of strings. If no specific campaign is suitable, leave it as an empty list [].
            3. If "CAMPAIGN_NAMES" is an empty list, you MUST choose a relevant "FOCUS_TECHNIQUE_ID" and its "FOCUS_TACTIC_NAME".
            4. Your output must be ONLY the completed JSON object.

            **Completed JSON Output:**
            """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
    
    del model, tokenizer, inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match: raise ValueError("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")
        completed_json = json.loads(json_match.group(0))
        
        original_keys = config.keys()
        updated_config = {key: completed_json.get(key, config[key]) for key in original_keys}
        
        print("  - ✔ LLM이 추천한 설정이 적용되었습니다.")
        return updated_config
    except Exception as e:
        print(f"  - ❌ LLM 설정 자동 완성 실패: {e}. 기본 설정으로 계속합니다.")
        return config

def load_all_data(paths):
    # 모든 필요한 데이터 파일을 로드하는 함수
    
    data = {}
    print("모든 데이터 소스를 로드합니다...")
    try:
        # 1. 파일 경로만 순회하며 로드
        for key, path in paths.items():
            # embedding_model_name은 모델 이름이므로 파일 존재 여부 검사에서 제외
            if key == "embedding_model_name":
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"필수 파일 '{path}'를 찾을 수 없습니다.")
            if key == "prompt_template":
                with open(path, 'r', encoding='utf-8') as f: data[key] = f.read()
            elif path.endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f: data[key] = json.load(f)
            elif path.endswith(".csv"):
                try:
                    # 'utf-8'이 표준이므로 먼저 시도
                    kev_df = pd.read_csv(path, encoding='utf-8')
                except UnicodeDecodeError:
                    # 실패 시 다른 인코딩으로 시도
                    kev_df = pd.read_csv(path, encoding='latin-1')
                print(f"  - KEV 데이터를 로드합니다: {path}")
                if 'cveID' in kev_df.columns:
                    data['kev_set'] = set(kev_df['cveID'])
                    print("  - KEV 세부 정보 딕셔너리를 생성합니다...")
                    kev_df_indexed = kev_df.set_index('cveID')
                    data['kev_details_dict'] = kev_df_indexed.to_dict(orient='index')
                else:
                    print(f"  - [경고] KEV.csv 파일에 'cveID' 컬럼이 없습니다.")
                    data['kev_set'] = set()

        # 2. 바이너리 파일 로드
        print("  - 벡터DB 인덱스를 로드합니다...")
        data['rag_index'] = faiss.read_index(paths["faiss_index"])
        with open(paths["capec_data"], "rb") as f: data['capec_data'] = pickle.load(f)

        # 3. 임베딩 모델은 이름으로 직접 로드 (파일 경로가 아님)
        embedding_model_name = paths["embedding_model_name"]
        print(f"  - 임베딩 모델 '{embedding_model_name}'을 로드합니다...")
        data['rag_model'] = SentenceTransformer(embedding_model_name)

        # 4. 이름 기반 조회를 위한 데이터 구조 변환 (기존과 동일)
        data['surfaces_by_name'] = {item['NAME']: item for item in data['attack_surface_en']}
        data['vectors_by_name'] = {item['NAME']: item for item in data['attack_vector_en']}
        cti_groups = data.get('cti_data', {}).get('groups', {})
        data['groups_by_name'] = cti_groups
        campaign_objects = data.get('campaign-merge', {}).get('objects', [])
        data['campaigns_by_name'] = {c.get('name'): c for c in campaign_objects if c.get('type') == 'campaign'}

        print("✔ 모든 데이터 소스를 성공적으로 로드했습니다.")
        return data
    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류 발생: {e}")
        return None

def search_capec(query, top_k=1, rag_model=None, rag_index=None, capec_data=None):
    """RAG를 통해 CAPEC 문서를 검색하고 'Description' 부분만 추출하는 함수"""
    if not all([rag_model, rag_index, capec_data]):
        return "No CAPEC context available."
    
    query_vector = rag_model.encode([query])
    _, indices = rag_index.search(np.array(query_vector).astype('float32'), top_k)
    
    full_text = capec_data['documents'][indices[0][0]]
    
    # 정규식을 사용하여 "Description:"으로 시작하고 다음 주요 섹션 전까지의 내용을 추출
    # (re.DOTALL 플래그는 '.'이 줄바꿈 문자도 포함하도록 함)
    match = re.search(r"Description:(.*?)(?=\n\w+:)", full_text, re.DOTALL)
    
    if match:
        # 추출된 텍스트의 앞뒤 공백을 제거하여 반환
        return match.group(1).strip()
    else:
        # 'Description'을 찾지 못한 경우, 문서의 앞부분 일부를 반환
        return full_text[:300] + "..."

## [수정됨] build_attack_chain 함수
def build_attack_chain(config, cti_data, groups_by_name, kev_set):
    """[KEV 수정] KEV에 해당하는 CVE ID를 함께 기록하여 TTP 체인을 구성하는 함수"""

    if config['COMPLEXITY_LEVEL'] == 'Simple':
        selected_tactics = ["Initial Access", "Execution", "Impact"]
    elif config['COMPLEXITY_LEVEL'] == 'Standard':
        selected_tactics = ["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Impact"]
    else: # Complex
        selected_tactics = ["Reconnaissance", "Resource Development", "Initial Access", "Execution", "Persistence", "Privilege Escalation",
            "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement", "Collection", "Exfiltration", "Impact"]
        
    print(f"\n[Step 1] '{config['COMPLEXITY_LEVEL']}' 복잡도 수준으로 공격 체인을 구성합니다...")
    tactic_order = ["Reconnaissance", "Resource Development", "Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement", "Collection", "Command and Control", "Exfiltration", "Impact"]

    source_technique_ids = set()
    campaign_names = config.get('CAMPAIGN_NAMES', [])
    if campaign_names:
        print(f"  - [캠페인 기반] {campaign_names} 그룹(들)의 기술을 종합합니다.")
        for group_name in campaign_names:
            group_data = groups_by_name.get(group_name)
            if group_data:
                source_technique_ids.update(group_data.get('used_techniques', []))
        if not source_technique_ids: raise ValueError(f"'{campaign_names}' 그룹들의 기술 정보를 찾을 수 없습니다.")
    else:
        print(f"  - [기술 중심] '{config['FOCUS_TECHNIQUE_ID']}' 기술을 중심으로 범용 체인을 구성합니다.")
        source_technique_ids = set(cti_data['techniques'].keys())

    tactic_to_techniques = defaultdict(list)
    for tech_id in list(source_technique_ids):
        tech_info = cti_data['techniques'].get(tech_id)
        if tech_info and 'tactics' in tech_info:
            for tactic in tech_info['tactics']:
                tactic_to_techniques[tactic].append(tech_id)

    attack_chain = []
    for tactic_name in selected_tactics:
        possible_techs = tactic_to_techniques.get(tactic_name)
        if possible_techs:
            kev_related_techs = []
            tech_to_kev_cves = {} # 기술 ID별 KEV CVE를 저장할 딕셔너리

            for tech_id in possible_techs:
                tech_info = cti_data['techniques'].get(tech_id, {})
                associated_cves = set(tech_info.get('associated_cves', []))
                found_kev_cves = kev_set.intersection(associated_cves)
                if found_kev_cves:
                    kev_related_techs.append(tech_id)
                    tech_to_kev_cves[tech_id] = list(found_kev_cves)

            chosen_technique_id, is_kev, related_cves = None, False, []
            if kev_related_techs:
                chosen_technique_id = random.choice(kev_related_techs)
                is_kev = True
                related_cves = tech_to_kev_cves[chosen_technique_id] # 해당 기술의 CVE 목록 가져오기
            elif possible_techs:
                chosen_technique_id = random.choice(possible_techs)

            if chosen_technique_id:
                technique_info = cti_data['techniques'].get(chosen_technique_id)
                tactic_id = "TA" + str(tactic_order.index(tactic_name) + 1).zfill(4) if tactic_name in tactic_order else "N/A"
                attack_chain.append({
                    "tactic_name": tactic_name, "tactic_id": tactic_id,
                    "technique_name": technique_info['name'], "technique_id": chosen_technique_id,
                    "is_kev": is_kev,
                    "kev_cves": related_cves  # KEV에 해당하는 CVE ID 리스트를 저장
                })

    if not attack_chain: raise ValueError("유효한 공격 체인을 구성하지 못했습니다.")
    print(f"  - ✔ 총 {len(attack_chain)} 단계의 공격 체인 구성 완료 (KEV 우선 적용).")
    return attack_chain

## [수정됨] 다중 캠페인 정보를 프롬프트에 반영하도록 수정
def assemble_final_prompt(config, data, attack_chain):
    print("\n[Step 2] 최종 프롬프트를 조합합니다...")
    prompt = data['prompt_template']
    
    # LLM에게 선택지를 주기 위해 유효한 목록을 프롬프트에 주입
    available_surfaces = [s['NAME'] for s in data.get('attack_surface_en', [])]
    available_vectors = [v['NAME'] for v in data.get('attack_vector_en', [])]
    prompt = prompt.replace("{{AVAILABLE_SURFACES_LIST}}", json.dumps(available_surfaces))
    prompt = prompt.replace("{{AVAILABLE_VECTORS_LIST}}", json.dumps(available_vectors))
    
    surface_info = data['surfaces_by_name'].get(config['ATTACK_SURFACE_NAME'], {})
    vector_info = data['vectors_by_name'].get(config['ATTACK_VECTOR_NAME'], {})
    level_info = data['attack_levels'].get(config['ATTACK_LEVEL'], {})
    complexity_info = data['complexity'].get(config['COMPLEXITY_LEVEL'], {})
    replacements = {
        "{{INDUSTRY}}": config['INDUSTRY'],
        "{{ATTACK_SURFACE}}": surface_info.get('DESCRIPTION', 'N/A'),
        "{{ATTACK_VECTOR}}": vector_info.get('DESCRIPTION', 'N/A'),
        "{{ATTACK_LEVEL}}": f"{config['ATTACK_LEVEL']} - {level_info.get('description', '')}",
        "{{COMPLEXITY_LEVEL}}": config['COMPLEXITY_LEVEL'],
    }
    campaign_names = config.get('CAMPAIGN_NAMES', [])
    if campaign_names:
        campaign_details_str = ""
        for name in campaign_names:
            campaign_info = data['campaigns_by_name'].get(name)
            if campaign_info:
                ext_id = next((ref['external_id'] for ref in campaign_info.get('external_references', []) if ref.get('source_name') == 'mitre-attack'), 'N/A')
                campaign_details_str += f"""
        <campaign name="{name}" id="{ext_id}">
            <description>{campaign_info.get('description', 'N/A')}</description>
            <primary_motivation>{campaign_info.get('primary_motivation', 'Not specified')}</primary_motivation>
        </campaign>"""
        replacements["{{THREAT_PROFILE_DETAILS}}"] = campaign_details_str
        replacements["{{CAMPAIGN_NAME_OR_GENERIC}}"] = ", ".join(campaign_names)
        replacements["{{CAMPAIGN_GOAL_OR_FOCUS}}"] = "Multiple campaign objectives"
    else:
        tech_info = data['cti_data']['techniques'].get(config['FOCUS_TECHNIQUE_ID'], {})
        focus_details_str = f"""
        <focus>
            <tactic id="{next((t['tactic_id'] for t in attack_chain if t['technique_id'] == config['FOCUS_TECHNIQUE_ID']), 'N/A')}">{tech_info.get('tactics', ['N/A'])[0]}</tactic>
            <technique id="{config['FOCUS_TECHNIQUE_ID']}">{tech_info.get('name', 'N/A')}</technique>
        </focus>"""
        replacements["{{THREAT_PROFILE_DETAILS}}"] = focus_details_str
        replacements["{{CAMPAIGN_NAME_OR_GENERIC}}"] = "Generic Threat Profile"
        replacements["{{CAMPAIGN_GOAL_OR_FOCUS}}"] = f"Exploitation of {tech_info.get('name', 'N/A')}"

    attack_chain_str = ""
    for stage in attack_chain:
        query = f"attack pattern for ATT&CK Technique {stage['technique_id']} {stage['technique_name']}"
        capec_context = search_capec(query, 1, data['rag_model'], data['rag_index'], data['capec_data'])
        kev_details_str = ""
        if stage.get("is_kev"):
            for cve_id in stage.get("kev_cves", []):
                kev_info = data['kev_details_dict'].get(cve_id, {})
                kev_name = kev_info.get('name', 'N/A')
                kev_desc = kev_info.get('description', 'N/A')
                kev_details_str += f"<kev id='{cve_id}' name='{kev_name}'>{kev_desc}</kev>"

        attack_chain_str += f"""
        <tactic id="{stage['tactic_id']}" name="{stage['tactic_name']}">
            <technique id="{stage['technique_id']}">{stage['technique_name']}</technique>
            <capec id="[CAPEC_ID]" name="[CAPEC_NAME]">{capec_context}</capec>
            {kev_details_str}
        </tactic>"""
    replacements["{{ATTACK_CHAIN_PLACEHOLDER}}"] = attack_chain_str
    
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, str(value))
    print("  - ✔ 최종 프롬프트 조합 완료.")
    return prompt

def generate_scenario(prompt, model_name):
    """LLM을 호출하여 시나리오를 생성하는 함수"""
    print("\n[Step 3] LLM을 로드하고 시나리오 생성을 시작합니다...")
    print(f"  - 모델: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=8192, do_sample=True, temperature=0.1, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    scenario_match = re.search(r'<scenario>.*</scenario>', full_output, re.DOTALL)
    if scenario_match:
        final_scenario = scenario_match.group(0)
    else:
        print("  - [경고] 모델 응답에서 <scenario> 블록을 찾지 못했습니다.")
        final_scenario = re.split(r'\[/INST\]', full_output)[-1].strip()

    del model, tokenizer, inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return final_scenario

## [수정됨] 모델 이름 폴더를 생성하도록 경로 로직 수정
def run_single_scenario(base_config, all_data, industry, complexity, attack_level):
    active_config = copy.deepcopy(base_config)
    active_config.update({"INDUSTRY": industry, "COMPLEXITY_LEVEL": complexity, "ATTACK_LEVEL": attack_level})
    print("\n" + "="*80)
    print(f"Executing: Industry='{industry}', Complexity='{complexity}', Attack Level='{attack_level}'")
    print("="*80)
    is_incomplete = any(not active_config.get(key) for key in ["CAMPAIGN_NAMES", "ATTACK_SURFACE_NAME", "ATTACK_VECTOR_NAME"]) or not active_config.get("CAMPAIGN_NAMES")
    if is_incomplete:
        active_config = complete_config_with_llm(active_config, all_data, MODEL_NAME)
    print("\n--- 최종 시나리오 설정 ---")
    print(json.dumps(active_config, indent=2))
    print("--------------------------")
    try:
        attack_chain = build_attack_chain(active_config, all_data['cti_data'], all_data['groups_by_name'], all_data.get('kev_set', set()))
        final_prompt = assemble_final_prompt(active_config, all_data, attack_chain)
        # debug_filename = f"debug_prompt_{industry}_{complexity}_{attack_level}.txt"
        # with open(debug_filename, "w", encoding="utf-8") as f: f.write(final_prompt)
        # print(f"  - ✔ 디버깅용 최종 프롬프트가 '{debug_filename}'에 저장되었습니다.")
        final_scenario = generate_scenario(final_prompt, MODEL_NAME)
        
        model_name_short = MODEL_NAME.split('/')[-1]
        scenario_output_dir = os.path.join(OUTPUT_DIR, model_name_short, industry.replace(" ", "_"), complexity.replace(" ", "_"))
        os.makedirs(scenario_output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{attack_level}_{timestamp}.xml" # 파일 이름에서 모델 이름은 폴더로 갔으므로 제외
        
        filepath = os.path.join(scenario_output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(final_scenario)
        print(f"✔ 시나리오가 '{filepath}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        import traceback
        print(f"\n❌ 파이프라인 실행 중 심각한 오류가 발생했습니다: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    INDUSTRIES = [
                "Communications",
                # "Energy", 
                # "Healthcare and Public Health", 
                # "Transportation Systems",
                # "Water and Wastewater Systems"
                ]
    COMPLEXITY_LEVEL = [
                "Simple",
                # "Standard",
                # "Complex"
                ]
    ATTACK_LEVELS = ["High", "Medium", "Low"]

    print("실험을 시작하기 전, 모든 데이터를 미리 로드합니다...")
    all_data = load_all_data(FILE_PATHS)
    
    if all_data:
        total_runs = len(INDUSTRIES) * len(COMPLEXITY_LEVEL) * len(ATTACK_LEVELS)
        current_run = 0
        for industry in INDUSTRIES:
            for complexity in COMPLEXITY_LEVEL:
                for attack_level in ATTACK_LEVELS:
                    current_run += 1
                    print(f"\n\n--- 전체 진행률: {current_run}/{total_runs} ---")
                    run_single_scenario(
                        base_config=SCENARIO_CONFIG,
                        all_data=all_data,
                        industry=industry,
                        complexity=complexity,
                        attack_level=attack_level
                    )
        print("\n\n✅ 모든 시나리오 생성이 완료되었습니다.")
    else:
        print("❌ 데이터 로딩에 실패하여 실험을 시작할 수 없습니다.")