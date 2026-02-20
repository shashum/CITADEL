import os
import json
import re
import random
import copy
import pickle
from datetime import datetime
from collections import defaultdict

# RAG 및 데이터 처리를 위한 라이브러리
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Claude API 라이브러리
import anthropic

# Get API key from environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
client = anthropic.Anthropic(api_key=api_key)

def read_file(file_path: str) -> str:
    """텍스트 파일 읽기 헬퍼."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_path} not found."


# --- 파일 경로 설정 ---
FILE_PATHS = {
    "prompt_template": "prompt_file.txt",          # 지금 올려둔 프롬프트 파일
    "cti_data": "preprocessed_cti_data.json",
    "faiss_index": "capec_index.faiss",
    "capec_data": "capec_data.pkl",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "attack_levels": "attack_levels.json",
    "attack_surface_en": "attack_surface_en.json",
    "attack_vector_en": "attack_vector_en.json",
    "campaign-merge": "campaign-merge.json",
    "kev_data": "known_exploited_vulnerabilities.csv",
}

OUTPUT_DIR = "Output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 데이터 로드 함수 ---
def load_all_data(paths):
    data = {}
    print("Checking data files...")
    try:
        # 필수 데이터 파일 로드
        with open(paths["cti_data"], "r", encoding="utf-8") as f:
            data["cti_data"] = json.load(f)
        with open(paths["attack_levels"], "r", encoding="utf-8") as f:
            data["attack_levels"] = json.load(f)
        with open(paths["attack_surface_en"], "r", encoding="utf-8") as f:
            data["attack_surface_en"] = json.load(f)
        with open(paths["attack_vector_en"], "r", encoding="utf-8") as f:
            data["attack_vector_en"] = json.load(f)
        with open(paths["campaign-merge"], "r", encoding="utf-8") as f:
            data["campaign-merge"] = json.load(f)

        # KEV 데이터
        if os.path.exists(paths["kev_data"]):
            try:
                kev_df = pd.read_csv(paths["kev_data"], encoding="utf-8")
            except Exception:
                kev_df = pd.read_csv(paths["kev_data"], encoding="latin-1")
            if "cveID" in kev_df.columns:
                data["kev_details_dict"] = kev_df.set_index("cveID").to_dict(
                    orient="index"
                )
                data["kev_set"] = set(data["kev_details_dict"].keys())
            else:
                data["kev_details_dict"] = {}
                data["kev_set"] = set()
        else:
            data["kev_details_dict"] = {}
            data["kev_set"] = set()

        # RAG 관련 (Faiss, Pickle, SentenceTransformer)
        print("  - Loading RAG Engine...")
        data["rag_index"] = faiss.read_index(paths["faiss_index"])
        with open(paths["capec_data"], "rb") as f:
            data["capec_data"] = pickle.load(f)
        data["rag_model"] = SentenceTransformer(
            paths["embedding_model_name"], device=DEVICE
        )

        # Helper 매핑
        data["groups_by_name"] = data["cti_data"].get("groups", {})
        data["campaigns_by_name"] = {
            c.get("name"): c
            for c in data["campaign-merge"].get("objects", [])
            if c.get("type") == "campaign"
        }

        return data
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None


# --- 2. RAG 검색 함수 ---
def search_capec(query, top_k=1, rag_model=None, rag_index=None, capec_data=None):
    if not all([rag_model, rag_index, capec_data]):
        return "No CAPEC context."
    query_vector = rag_model.encode([query])
    _, indices = rag_index.search(np.array(query_vector).astype("float32"), top_k)
    full_text = capec_data["documents"][indices[0][0]]
    # Description만 추출 시도
    match = re.search(r"Description:(.*?)(?=\n\w+:)", full_text, re.DOTALL)
    return match.group(1).strip() if match else full_text[:500]


# --- 3. 공격 체인 구성 함수 (기존 로직) ---
def build_attack_chain(config, data):
    complexity = config["COMPLEXITY_LEVEL"]
    # 복잡도에 따른 Tactic 목록 정의
    if complexity == "Simple":
        tactics = ["Initial Access", "Execution", "Impact"]
    elif complexity == "Standard":
        tactics = [
            "Initial Access",
            "Execution",
            "Persistence",
            "Privilege Escalation",
            "Defense Evasion",
            "Impact",
        ]
    else:  # Complex
        tactics = [
            "Reconnaissance",
            "Resource Development",
            "Initial Access",
            "Execution",
            "Persistence",
            "Privilege Escalation",
            "Defense Evasion",
            "Credential Access",
            "Discovery",
            "Lateral Movement",
            "Collection",
            "Exfiltration",
            "Impact",
        ]

    source_techs = set()
    campaigns = config.get("CAMPAIGN_NAMES", [])

    # 캠페인 기반 후보 기술 선정
    if campaigns:
        for camp in campaigns:
            grp = data["groups_by_name"].get(camp)
            if grp:
                source_techs.update(grp.get("used_techniques", []))

    # 없으면 전체 기술에서 랜덤
    if not source_techs:
        source_techs = set(data["cti_data"]["techniques"].keys())

    # Tactic 별 기술 매핑
    tech_map = defaultdict(list)
    for tid in source_techs:
        tinfo = data["cti_data"]["techniques"].get(tid)
        if tinfo:
            for tac in tinfo.get("tactics", []):
                tech_map[tac].append(tid)

    chain = []
    kev_set = data.get("kev_set", set())

    for tac in tactics:
        candidates = tech_map.get(tac)
        if not candidates:
            # 해당 Tactic에 매핑된 기술이 없으면 전체 데이터셋에서 랜덤 추출
            all_techs = data["cti_data"]["techniques"]
            candidates = [
                tid for tid, info in all_techs.items() if tac in info.get("tactics", [])
            ]

        if candidates:
            # KEV 우선 선택
            kev_candidates = [
                tid
                for tid in candidates
                if kev_set.intersection(
                    set(
                        data["cti_data"]["techniques"][tid].get(
                            "associated_cves", []
                        )
                    )
                )
            ]

            chosen_id = (
                random.choice(kev_candidates) if kev_candidates else random.choice(candidates)
            )
            tech_info = data["cti_data"]["techniques"][chosen_id]

            # KEV 정보 추출
            related_cves = list(
                kev_set.intersection(set(tech_info.get("associated_cves", [])))
            )

            # RAG로 CAPEC 검색
            query = f"attack pattern for {chosen_id} {tech_info['name']}"
            capec_desc = search_capec(
                query, 1, data["rag_model"], data["rag_index"], data["capec_data"]
            )

            chain.append(
                {
                    "tactic": tac,
                    "technique_id": chosen_id,
                    "technique_name": tech_info["name"],
                    "capec_context": capec_desc,
                    "kev_cves": related_cves,
                }
            )

    return chain


# --- 4. <threat_intelligence_document> XML 조립 ---
def assemble_threat_document(config, data, attack_chain):
    """LLM에게 넘겨줄 <threat_intelligence_document> 문자열을 생성."""
    # Available lists
    avail_surfaces = [s["NAME"] for s in data["attack_surface_en"]]
    avail_vectors = [v["NAME"] for v in data["attack_vector_en"]]

    # Campaign Details
    campaign_info_xml = ""
    for name in config.get("CAMPAIGN_NAMES", []):
        c_obj = data["campaigns_by_name"].get(name, {})
        campaign_info_xml += f"""
        <campaign name="{name}">
            <description>{c_obj.get('description', 'N/A')}</description>
            <motivation>{c_obj.get('primary_motivation', 'Unknown')}</motivation>
        </campaign>"""

    # Attack Chain XML construction
    chain_xml = ""
    for stage in attack_chain:
        # KEV XML
        kev_xml = ""
        for cve in stage["kev_cves"]:
            k_info = data["kev_details_dict"].get(cve, {})
            kev_xml += (
                f"<kev id='{cve}' "
                f"name='{k_info.get('product', 'Unknown')}' "
                f"type='{k_info.get('vulnerabilityName', 'Unknown')}'>"
                f"{k_info.get('shortDescription', '')}</kev>"
            )

        chain_xml += f"""
        <stage tactic="{stage['tactic']}">
            <technique_id>{stage['technique_id']}</technique_id>
            <technique_name>{stage['technique_name']}</technique_name>
            <capec_context>{stage['capec_context']}</capec_context>
            {kev_xml}
        </stage>"""

    # Final Document Assembly
    document = f"""
<threat_intelligence_document>
    <metadata>
        <target_industry>{config['INDUSTRY']}</target_industry>
        <complexity_level>{config['COMPLEXITY_LEVEL']}</complexity_level>
        <attacker_skill>{config['ATTACK_LEVEL']}</attacker_skill>
    </metadata>
    <threat_actor_profile>
        {campaign_info_xml if campaign_info_xml else "<generic_profile>APT Group targeting " + config['INDUSTRY'] + "</generic_profile>"}
    </threat_actor_profile>
    <available_options>
        <surfaces>{json.dumps(avail_surfaces)}</surfaces>
        <vectors>{json.dumps(avail_vectors)}</vectors>
    </available_options>
    <attack_chain>
        {chain_xml}
    </attack_chain>
</threat_intelligence_document>
""".strip()

    return document


# --- 5. 메인 실행 함수 ---
def run_generator():
    # 출력 폴더 준비
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 데이터 로드
    all_data = load_all_data(FILE_PATHS)
    if not all_data:
        return

    # 예시 설정 (실제로는 반복문 등으로 설정 가능)
    config = {
        "INDUSTRY": "Water and wastewater",
        "COMPLEXITY_LEVEL": "Complex",  # Simple / Standard / Complex
        "ATTACK_LEVEL": "Medium",
        # "CAMPAIGN_NAMES": ["Lazarus Group"],  # 필요시 사용
    }

    print(f"\nGenerating Scenario for {config['INDUSTRY']}...")

    # 1. 공격 체인 구성 (파이썬에서)
    attack_chain = build_attack_chain(config, all_data)

    # 2. threat_intelligence_document XML 생성
    threat_doc = assemble_threat_document(config, all_data, attack_chain)

    # 3. 프롬프트 템플릿 로드 (지금 올려둔 prompt-gemini.txt)
    system_prompt = read_file(FILE_PATHS["prompt_template"])

    # 4. Claude 호출
    try:
        response = client.messages.create(
            # 실제 사용 중인 Claude 모델명으로 교체해서 사용
            model="claude-opus-4-1-20250805",
            # claude-opus-4-1-20250805
            max_tokens=5000,
            temperature=0.1,
            # prompt-gemini.txt 내용을 system에 넣어서
            # "너는 시나리오를 만드는 분석가" 규칙을 주입
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    # user 메시지에는 **순수하게 threat_intelligence_document만** 넣기
                    "content": threat_doc,
                }
            ],
        )

        scenario_content = response.content[0].text

        # 결과 저장
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"Scenario_{ts}.xml")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(scenario_content)

        print(f"\n✅ Scenario saved to {filename}")
        print(scenario_content[:500] + "...\n")  # 앞부분 미리보기

    except Exception as e:
        print(f"❌ API Error: {e}")


if __name__ == "__main__":
    run_generator()