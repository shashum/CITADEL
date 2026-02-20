import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

# --- 1. 설정 (Configuration) ---
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# --- [중요] 여기서 생성할 시나리오의 목표를 설정하세요 ---
SCENARIO_CONFIG = {
    "INDUSTRY": "Energy",
    "GENERATION_MODE": "Actor-Emulation",  # "Actor-Emulation" 또는 "Capability-Centric"
    "COMPLEXITY_LEVEL": "Standard",      # "Simple", "Standard", "Complex"
    
    # Actor-Emulation 모드일 때 사용
    "CAMPAIGN_NAME": "Sandworm Team", 
    
    # Capability-Centric 모드일 때 사용 (예시)
    "FOCUS_TACTIC_NAME": "Impact",
    "FOCUS_TECHNIQUE_ID": "T1486",

    # 사용자 JSON 파일에서 가져올 정보 (이름으로 매칭)
    "ATTACK_SURFACE_NAME": "Remote Access Services",
    "ATTACK_VECTOR_NAME": "Exploitation of public-facing application",
    "ATTACK_LEVEL": "High"
}
# ---------------------------------------------------------

# --- 파일 경로 설정 (키 이름을 파일명과 일치시켜 오류 수정) ---
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
    "critical-infra-description": "critical-infra-description.json"
}

# --- 결과물 저장 폴더 ---
OUTPUT_DIR = "final_generated_scenarios"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 2. 데이터 로더 및 헬퍼 함수 (최종 수정 버전) ---

def load_all_data(paths):
    """[최종 수정됨] 모든 필요한 데이터 파일을 로드하는 함수"""
    data = {}
    print("모든 데이터 소스를 로드합니다...")
    try:
        # 텍스트, JSON 파일 로드
        for key, path in paths.items():
            if key == "embedding_model_name": continue
            if not os.path.exists(path): raise FileNotFoundError(f"필수 파일 '{path}'를 찾을 수 없습니다.")
            
            if key == "prompt_template":
                with open(path, 'r', encoding='utf-8') as f: data[key] = f.read()
            elif path.endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f: data[key] = json.load(f)

        # 바이너리, 모델 파일 로드
        print("  - 벡터DB 인덱스를 로드합니다...")
        data['rag_index'] = faiss.read_index(paths["faiss_index"])
        with open(paths["capec_data"], "rb") as f: data['capec_data'] = pickle.load(f) # 'capec_data' 키 사용
        
        print(f"  - 임베딩 모델 '{paths['embedding_model_name']}'을 로드합니다...")
        data['rag_model'] = SentenceTransformer(paths["embedding_model_name"])
        
        # 이름 기반 조회를 위해 데이터 구조 변환
        data['surfaces_by_name'] = {item['NAME']: item for item in data['attack_surface_en']}
        data['vectors_by_name'] = {item['NAME']: item for item in data['attack_vector_en']}
        data['campaigns_by_name'] = data['campaign-merge']['objects']

        print("✔ 모든 데이터 소스를 성공적으로 로드했습니다.")
        return data
    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류 발생: {e}")
        return None

def search_capec(query, top_k=1, rag_model=None, rag_index=None, capec_data=None):
    if not all([rag_model, rag_index, capec_data]): return "No CAPEC context available."
    query_vector = rag_model.encode([query])
    _, indices = rag_index.search(np.array(query_vector).astype('float32'), top_k)
    return capec_data['documents'][indices[0][0]]

# --- 3. TTP 공격 체인 구성 함수 ---
def build_attack_chain(config, cti_data):
    print(f"\n[Step 1] '{config['COMPLEXITY_LEVEL']}' 복잡도 수준으로 공격 체인을 구성합니다...")
    tactic_order = ["Reconnaissance", "Resource Development", "Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement", "Collection", "Command and Control", "Exfiltration", "Impact"]
    
    source_technique_ids = []
    if config['GENERATION_MODE'] == 'Actor-Emulation':
        group_name = config['CAMPAIGN_NAME']
        group_data = cti_data['groups'].get(group_name)
        if not group_data: raise ValueError(f"'{group_name}' 그룹을 CTI 데이터에서 찾을 수 없습니다.")
        source_technique_ids = group_data['used_techniques']
        print(f"  - '{group_name}' 그룹이 사용하는 {len(source_technique_ids)}개의 기술을 기반으로 합니다.")
    else: # Capability-Centric
        source_technique_ids = list(cti_data['techniques'].keys())
        print(f"  - '{config['FOCUS_TECHNIQUE_ID']}' 기술을 중심으로 범용적인 체인을 구성합니다.")

    tactic_to_techniques = defaultdict(list)
    for tech_id in source_technique_ids:
        tech_info = cti_data['techniques'].get(tech_id)
        if tech_info and 'tactics' in tech_info:
            for tactic in tech_info['tactics']:
                tactic_to_techniques[tactic].append(tech_id)

    if config['COMPLEXITY_LEVEL'] == 'Simple':
        selected_tactics = ["Initial Access", "Execution", "Impact"]
    elif config['COMPLEXITY_LEVEL'] == 'Standard':
        selected_tactics = ["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Impact"]
    else: # Complex
        used_tactics_in_order = [t for t in tactic_order if t in tactic_to_techniques and tactic_to_techniques[t]]
        selected_tactics = used_tactics_in_order if used_tactics_in_order else tactic_order

    attack_chain = []
    for tactic_name in selected_tactics:
        if tactic_to_techniques.get(tactic_name):
            chosen_technique_id = random.choice(tactic_to_techniques[tactic_name])
            technique_info = cti_data['techniques'].get(chosen_technique_id)
            if technique_info:
                tactic_id = "TA" + str(tactic_order.index(tactic_name) + 1).zfill(4) if tactic_name in tactic_order else "N/A"
                attack_chain.append({
                    "tactic_name": tactic_name, "tactic_id": tactic_id,
                    "technique_name": technique_info['name'], "technique_id": chosen_technique_id
                })
    
    if not attack_chain: raise ValueError("유효한 공격 체인을 구성하지 못했습니다. 사전 처리된 CTI 데이터에 'tactics' 정보가 포함되어 있는지 확인하세요.")
    print(f"  - ✔ 총 {len(attack_chain)} 단계의 공격 체인 구성 완료.")
    return attack_chain

# --- 4. 최종 프롬프트 조합 함수 (최종 수정 버전) ---
def assemble_final_prompt(config, data, attack_chain):
    """[최종 수정됨] 모든 정보를 최종 마스터 프롬프트 템플릿에 삽입하는 함수"""
    print("\n[Step 2] 최종 프롬프트를 조합합니다...")
    
    prompt = data['prompt_template']
    
    # [핵심 수정] 올바른 키 이름을 사용하여 데이터 조회
    surface_info = data['surfaces_by_name'].get(config['ATTACK_SURFACE_NAME'], {})
    vector_info = data['vectors_by_name'].get(config['ATTACK_VECTOR_NAME'], {})
    level_info = data['attack_levels'].get(config['ATTACK_LEVEL'], {})
    complexity_info = data['complexity'].get(config['COMPLEXITY_LEVEL'], {})

    replacements = {
        "{{GENERATION_MODE}}": config['GENERATION_MODE'], "{{INDUSTRY}}": config['INDUSTRY'],
        "{{ATTACK_SURFACE}}": surface_info.get('DESCRIPTION', 'N/A'),
        "{{ATTACK_VECTOR}}": vector_info.get('DESCRIPTION', 'N/A'),
        "{{ATTACK_LEVEL}}": f"{config['ATTACK_LEVEL']} - {level_info.get('description', '')}",
        "{{COMPLEXITY_LEVEL}}": config['COMPLEXITY_LEVEL'],
        "{{COMPLEXITY_DESCRIPTION}}": complexity_info.get('description', '')
    }

    if config['GENERATION_MODE'] == 'Actor-Emulation':
        replacements["{{MODE_DESCRIPTION}}"] = "Goal: Emulate a specific threat actor..."
        campaign_info = next((c for c in data['campaigns_by_name'] if c['name'] == config['CAMPAIGN_NAME']), {})
        replacements.update({
            "{{CAMPAIGN_NAME}}": config['CAMPAIGN_NAME'],
            "{{GROUP_ID_OR_CAMPAIGN_ID}}": campaign_info.get('external_references', [{}])[0].get('external_id', 'N/A'),
            "{{CAMPAIGN_DESCRIPTION}}": campaign_info.get('description', 'N/A'),
            "{{CAMPAIGN_GOAL}}": campaign_info.get('primary_motivation', 'Not specified')
        })
        prompt = re.sub(r"", '', prompt, flags=re.DOTALL)
    else: # Capability-Centric
        replacements["{{MODE_DESCRIPTION}}"] = "Goal: Analyze a general cyber capability..."
        tech_info = data['cti_data']['techniques'].get(config['FOCUS_TECHNIQUE_ID'], {})
        replacements.update({
            "{{TACTIC_ID}}": next((t['tactic_id'] for t in attack_chain if t['technique_id'] == config['FOCUS_TECHNIQUE_ID']), 'N/A'),
            "{{TACTIC_NAME}}": tech_info.get('tactics', ['N/A'])[0],
            "{{TECHNIQUE_ID}}": config['FOCUS_TECHNIQUE_ID'],
            "{{TECHNIQUE_NAME}}": tech_info.get('name', 'N/A')
        })
        prompt = re.sub(r"", '', prompt, flags=re.DOTALL)

    attack_chain_str = ""
    for stage in attack_chain:
        query = f"attack pattern for ATT&CK Technique {stage['technique_id']} {stage['technique_name']}"
        # [핵심 수정] 'capec_data' 키를 사용하여 RAG 데이터 전달
        capec_context = search_capec(query, 1, data['rag_model'], data['rag_index'], data['capec_data'])
        attack_chain_str += f"""
        <tactic id="{stage['tactic_id']}" name="{stage['tactic_name']}">
            <technique id="{stage['technique_id']}">{stage['technique_name']}</technique>
            <capec_details>{capec_context}</capec_details>
            <description>TODO: Write a 140-220 word narrative description for this step.</description>
        </tactic>"""
    replacements["{{ATTACK_CHAIN_PLACEHOLDER}}"] = attack_chain_str

    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, str(value))
        
    print("  - ✔ 최종 프롬프트 조합 완료.")
    return prompt

# --- 5. LLM 호출 및 결과 저장 ---
def generate_scenario(prompt, model_name):
    """[최종 수정됨] LLM을 호출하여 시나리오를 생성하는 함수"""
    print("\n[Step 3] LLM을 로드하고 시나리오 생성을 시작합니다...")
    print(f"  - 모델: {model_name}")
    print("  - (LLM 로딩은 VRAM 크기에 따라 몇 분 정도 소요될 수 있습니다)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "user", "content": prompt}]
    # [핵심 수정] tokenizer의 결과가 딕셔너리인지 텐서인지에 따라 유연하게 처리
    inputs_data = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    
    if isinstance(inputs_data, dict):
        input_ids = inputs_data.get('input_ids')
        attention_mask = inputs_data.get('attention_mask')
    else: # Tensor
        input_ids = inputs_data
        attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4096,
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    scenario_match = re.search(r'<scenario>.*</scenario>', full_output, re.DOTALL)
    if scenario_match:
        final_scenario = scenario_match.group(0)
    else:
        print("  - [경고] 모델 응답에서 <scenario> 블록을 찾지 못했습니다. 전체 응답을 저장합니다.")
        final_scenario = re.split(r'\[/INST\]|<s>|</s>', full_output)[-1].strip()

    del model, tokenizer, inputs_data, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return final_scenario

# --- 6. 메인 실행 함수 ---
def main():
    all_data = load_all_data(FILE_PATHS)
    if not all_data: return

    try:
        attack_chain = build_attack_chain(SCENARIO_CONFIG, all_data['cti_data'])
        final_prompt = assemble_final_prompt(SCENARIO_CONFIG, all_data, attack_chain)
        
        with open("debug_prompt.txt", "w", encoding="utf-8") as f: f.write(final_prompt)
        print("  - 디버깅용 프롬프트가 'debug_prompt.txt'에 저장되었습니다.")

        final_scenario = generate_scenario(final_prompt, MODEL_NAME)
        
        print("\n--- 생성된 시나리오 (요약) ---")
        print(final_scenario[:500] + "...")
        print("---------------------------\n")

        filename = f"scenario_{SCENARIO_CONFIG.get('CAMPAIGN_NAME', 'Capability')}_{SCENARIO_CONFIG['INDUSTRY']}_{SCENARIO_CONFIG['COMPLEXITY_LEVEL']}.xml"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f: f.write(final_scenario)
        print(f"✔ 시나리오가 '{filepath}'에 성공적으로 저장되었습니다.")

    except Exception as e:
        import traceback
        print(f"\n❌ 파이프라인 실행 중 심각한 오류가 발생했습니다: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()