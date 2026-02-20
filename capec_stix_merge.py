# capec_merge.py
import os, re, json, glob
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

CAPEC_ID_RE = re.compile(r"CAPEC-\d+")

# ========== 1) CSV 로더 (사이트 제공 CSV 묶음) ==========
def load_capec_csv_folder(folder_path: str) -> Dict[str, dict]:
    """
    폴더 내 모든 CSV를 읽어 'CAPEC-####' 키로 통합.
    - 'Comprehensive CAPEC Dictionary.csv'가 있으면 베이스로 삼고
      다른 CSV의 열을 누락 컬럼에 한해 보강.
    반환: { "CAPEC-98": {...}, ... }
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"[CSV] No CSV files in: {folder_path}")
        return {}

    base: Dict[str, dict] = {}
    primary = os.path.join(folder_path, "Comprehensive CAPEC Dictionary.csv")
    # 1) Primary
    if os.path.exists(primary):
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(primary, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise RuntimeError(f"[CSV] Failed to read: {primary}")

        for _, row in df.iterrows():
            cid = row.get("ID") or row.get("'ID")
            if pd.notna(cid):
                capec_id = f"CAPEC-{int(cid)}"
                base[capec_id] = {k: v for k, v in row.dropna().to_dict().items()}
                base[capec_id]["capec_id"] = capec_id

    # 2) Others
    for p in csv_files:
        if os.path.basename(p) == "Comprehensive CAPEC Dictionary.csv":
            continue
        for enc in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            print(f"[CSV] Skip unreadable: {p}")
            continue

        id_col = "ID" if "ID" in df.columns else ("'ID" if "'ID" in df.columns else None)
        if not id_col:
            print(f"[CSV] Skip(no ID col): {p}")
            continue

        for _, row in df.iterrows():
            cid = row.get(id_col)
            if pd.notna(cid):
                capec_id = f"CAPEC-{int(cid)}"
                rec = base.setdefault(capec_id, {"capec_id": capec_id})
                for k, v in row.dropna().to_dict().items():
                    if k not in rec:
                        rec[k] = v

    print(f"[CSV] Loaded CAPEC records: {len(base)}")
    return base


# ========== 2) STIX 로더 (capec-STIX 2.0/2.1) ==========
def _extract_capec_id_from_external_refs(external_refs: list) -> str:
    """
    STIX external_references에서 CAPEC-#### 식별자 추출
    """
    if not isinstance(external_refs, list):
        return None
    for ref in external_refs:
        # external_id 우선
        ext_id = (ref or {}).get("external_id")
        if isinstance(ext_id, str) and CAPEC_ID_RE.fullmatch(ext_id.strip()):
            return ext_id.strip()
        # url/description에 CAPEC-####가 있을 수도 있음
        for k in ("url", "description", "source_name"):
            val = (ref or {}).get(k)
            if isinstance(val, str):
                m = CAPEC_ID_RE.search(val)
                if m:
                    return m.group(0)
    return None

def _ts(s: str) -> datetime:
    # STIX modified/created ISO8601 -> datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.min

def load_capec_stix(paths: List[str]) -> Dict[str, dict]:
    """
    STIX bundle 파일들에서 CAPEC attack-pattern을 수집하여
    { 'CAPEC-####': {stix 기반 필드...} }로 반환
    - STIX 2.0/2.1 모두 지원 (bundle.objects[].type == "attack-pattern")
    - x_capec_* 커스텀 필드 포함(존재 시)
    """
    out: Dict[str, dict] = {}
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.json")))
        else:
            files.append(p)
    files = [f for f in files if os.path.exists(f)]

    count_objs = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[STIX] Skip {fp}: {e}")
            continue

        objs = data.get("objects") if isinstance(data, dict) else None
        if not isinstance(objs, list):
            continue

        for o in objs:
            if not isinstance(o, dict):
                continue
            if o.get("type") != "attack-pattern":
                continue

            capec_id = _extract_capec_id_from_external_refs(o.get("external_references", []))
            if not capec_id:
                continue

            count_objs += 1
            rec = out.get(capec_id, {"capec_id": capec_id})
            # 일반 필드
            rec["stix_id"] = o.get("id")
            rec["name"] = o.get("name") or rec.get("name")
            rec["description_stix"] = o.get("description") or rec.get("description_stix")
            rec["external_references"] = o.get("external_references") or rec.get("external_references")
            rec["modified"] = o.get("modified") or rec.get("modified") or o.get("x_mitre_modified")
            rec["created"] = o.get("created") or rec.get("created")

            # CAPEC 커스텀 확장 필드(있을 경우)
            # STIX capec 스키마에서 자주 보이는 키들: x_capec_abstraction, x_capec_likelihood_of_attack, x_capec_typical_severity, ...
            for k in list(o.keys()):
                if k.startswith("x_capec_"):
                    rec[k] = o.get(k)

            out[capec_id] = rec

    print(f"[STIX] Loaded CAPEC attack-pattern objects: {count_objs} (unique IDs: {len(out)})")
    return out


# ========== 3) 병합 로직 ==========
# 필드 매핑(CSV -> 통일명) : CSV 컬럼명이 다양할 수 있어 안전 매핑
CSV_MAP = {
    "Name": "name_csv",
    "Description": "description_csv",
    "Abstraction": "abstraction_csv",
    "Likelihood Of Attack": "likelihood_csv",
    "Typical Severity": "severity_csv",
    "Execution Flow": "execution_flow_csv",
    "Prerequisites": "prerequisites_csv",
    "Consequences": "consequences_csv",
    "Mitigations": "mitigations_csv",
    "Skills Required": "skills_required_csv",
    "Taxonomy Mappings": "taxonomy_mappings_csv"
}

def merge_capec_csv_stix(csv_dict: Dict[str, dict], stix_dict: Dict[str, dict]) -> Dict[str, dict]:
    """
    우선순위 규칙:
      - name/description: STIX 우선, 비어있을 때 CSV로 보완 (둘 다 있으면 둘 다 보존: *_stix, *_csv)
      - abstraction/likelihood/severity 등: STIX x_capec_* 우선, 없으면 CSV
      - 실행/전제/결과/완화/스킬: CSV가 더 풍부한 경우가 많아 CSV 우선(하지만 STIX쪽 x_capec_* 있으면 병합)
      - modified/created: STIX 타임스탬프 유지
    """
    all_ids = set(csv_dict.keys()) | set(stix_dict.keys())
    merged: Dict[str, dict] = {}

    for capec_id in sorted(all_ids, key=lambda x: int(x.split("-")[1])):
        c = csv_dict.get(capec_id, {})
        s = stix_dict.get(capec_id, {})

        rec = {"capec_id": capec_id}

        # 이름
        rec["name"] = s.get("name") or c.get("Name")
        rec["name_stix"] = s.get("name")
        rec["name_csv"] = c.get("Name")

        # 설명(둘 다 보존 + 대표 설명 선택)
        rec["description_stix"] = s.get("description_stix")
        rec["description_csv"] = c.get("Description")
        rec["description"] = rec["description_stix"] or rec["description_csv"]

        # STIX 메타
        rec["stix_id"] = s.get("stix_id")
        rec["external_references"] = s.get("external_references")
        rec["modified"] = s.get("modified")
        rec["created"] = s.get("created")

        # 커스텀 x_capec_* + CSV 보강
        # STIX 쪽
        for k in list(s.keys()):
            if k.startswith("x_capec_"):
                rec[k] = s[k]

        # CSV 매핑
        for src, dst in CSV_MAP.items():
            if c.get(src) and dst not in rec:
                rec[dst] = c.get(src)

        # 대표 필드(일반화)
        rec["abstraction"] = rec.get("x_capec_abstraction") or c.get("Abstraction")
        rec["likelihood_of_attack"] = rec.get("x_capec_likelihood_of_attack") or c.get("Likelihood Of Attack")
        rec["typical_severity"] = rec.get("x_capec_typical_severity") or c.get("Typical Severity")

        # 실행/전제/결과/완화/스킬: CSV 우선 + STIX 보강
        rec["execution_flow"] = c.get("Execution Flow") or rec.get("x_capec_execution_flow")
        rec["prerequisites"] = c.get("Prerequisites") or rec.get("x_capec_prerequisites")
        rec["consequences"] = c.get("Consequences") or rec.get("x_capec_consequences")
        rec["mitigations"] = c.get("Mitigations") or rec.get("x_capec_mitigations")
        rec["skills_required"] = c.get("Skills Required") or rec.get("x_capec_skills_required")

        # CWE/Taxonomy 힌트(문서 머릿부분에 키워드로 유용)
        # CSV Taxonomy에서 CWE-#### 추출
        cwe_csv = sorted(set(re.findall(r"CWE-\d+", str(c.get("Taxonomy Mappings", "")))))
        # STIX external_references/description 등에서도 CWE가 있을 수 있음
        cwe_stix = set()
        for ref in (s.get("external_references") or []):
            for k in ("external_id", "url", "description", "source_name"):
                val = (ref or {}).get(k)
                if isinstance(val, str):
                    cwe_stix |= set(re.findall(r"CWE-\d+", val))
        rec["cwes"] = sorted(set(cwe_csv) | cwe_stix)

        merged[capec_id] = rec

    print(f"[MERGE] merged records: {len(merged)}")
    return merged


# ========== 4) 임베딩용 문서 빌더 ==========
def build_capec_docs_from_merged(merged: Dict[str, dict]) -> Tuple[List[str], List[dict]]:
    """
    각 CAPEC 항목을 한 덩어리 텍스트로 구성
    텍스트 = 헤더(식별자/이름/추정치) + 요약(설명) + 세부(전제/흐름/결과/완화/스킬) + 참조
    메타 = {'kind':'capec', 'id':capec_id, 'name':..., 'cwes':[...] , 'modified':...}
    """
    docs, metas = [], []
    for capec_id, r in merged.items():
        header = f"{capec_id} | {r.get('name','')}"
        header += f" | Likelihood:{r.get('likelihood_of_attack','?')} | Severity:{r.get('typical_severity','?')}"
        if r.get("cwes"):
            header += " | CWE:" + ",".join(r["cwes"])

        parts = [header, ""]
        if r.get("description"):
            parts.append("Description: " + str(r["description"]).strip())
        # 보조 설명 둘 다 보관
        if r.get("description_stix") and r["description_stix"] != r.get("description"):
            parts.append("Description(STIX): " + str(r["description_stix"]).strip())
        if r.get("description_csv") and r["description_csv"] != r.get("description"):
            parts.append("Description(CSV): " + str(r["description_csv"]).strip())

        # 상세
        for k,label in [
            ("prerequisites","Prerequisites"),
            ("execution_flow","Execution Flow"),
            ("consequences","Consequences"),
            ("mitigations","Mitigations"),
            ("skills_required","Skills Required"),
            ("abstraction","Abstraction"),
        ]:
            v = r.get(k)
            if v:
                parts.append(f"{label}: {v}")

        # 외부참조 요약
        if r.get("external_references"):
            refs_lines = []
            for ref in r["external_references"]:
                if not isinstance(ref, dict): 
                    continue
                line = []
                if ref.get("source_name"): line.append(ref["source_name"])
                if ref.get("external_id"): line.append(ref["external_id"])
                if ref.get("url"): line.append(ref["url"])
                if ref.get("description"): line.append(ref["description"])
                if line:
                    refs_lines.append(" - " + " | ".join(line))
            if refs_lines:
                parts.append("External References:\n" + "\n".join(refs_lines))

        # 타임스탬프
        if r.get("modified") or r.get("created"):
            parts.append(f"Timestamps: modified={r.get('modified')} created={r.get('created')}")

        text = "\n".join(parts).strip()
        meta = {
            "kind": "capec",
            "id": capec_id,
            "name": r.get("name"),
            "cwes": r.get("cwes", []),
            "modified": r.get("modified"),
        }
        docs.append(text)
        metas.append(meta)

    return docs, metas


# ========== 5) 사용 예시 ==========
if __name__ == "__main__":
    # 예시 경로
    CSV_DIR = os.getenv("CAPEC_CSV_DIR", "../CAPEC_CSV")
    STIX_PATHS = os.getenv("CAPEC_STIX_PATHS", "MITRE-CTI\capec\2.1").split(os.pathsep)
    # 위 두 변수는 환경에 맞게 바꾸세요 (파일/폴더 혼용 가능)

    csv_dict = load_capec_csv_folder(CSV_DIR)
    stix_dict = load_capec_stix(STIX_PATHS)
    merged = merge_capec_csv_stix(csv_dict, stix_dict)
    docs, metas = build_capec_docs_from_merged(merged)

    print(f"[DOCS] {len(docs)} ready for embedding. Example:\n")
    print(docs[0][:1000] + ("..." if len(docs[0]) > 1000 else ""))
