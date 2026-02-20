# merge_mitre_campaigns.py
from __future__ import annotations
from pathlib import Path
import json
from uuid import uuid4
from typing import Iterable, Dict, Any, List

# ---- 설정 ----
# 루트 디렉토리(예: "MITRE-CTI"). 필요하면 여러 개로 확장 가능.
ROOTS = [Path("..\MITRE-CTI")]
# 출력 파일
OUT_FILE = Path("campaigns_bundle.json")
# STIX spec version (예시 파일 기준)
SPEC_VERSION = "2.0"

def find_campaign_json_files(roots: Iterable[Path]) -> Iterable[Path]:
    """루트들에서 */campaign/*.json 파일을 재귀적으로 찾는다."""
    for root in roots:
        # enterprise-attack/campaign, mobile-attack/campaign 등 모두 포괄
        yield from root.rglob("campaign/*.json")

def load_json(fp: Path) -> Any:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_objects_from_any_json(data: Any) -> Iterable[Dict[str, Any]]:
    """
    입력 JSON이
      - STIX bundle: {"type":"bundle","objects":[...]}
      - 배열: [ {...}, {...} ]
      - 단일 객체: { ... }
    인 경우를 모두 처리하고, 객체들을 평탄화하여 yield.
    """
    if isinstance(data, dict):
        if data.get("type") == "bundle" and isinstance(data.get("objects"), list):
            for obj in data["objects"]:
                if isinstance(obj, dict):
                    yield obj
        else:
            # 일반 단일 객체
            yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item

def collect_campaigns(files: Iterable[Path]) -> List[Dict[str, Any]]:
    """
    모든 파일에서 type=="campaign" 인 객체를 모으고,
    동일 id 중복은 제거한다.
    """
    seen_ids = set()
    campaigns: List[Dict[str, Any]] = []

    for fp in files:
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"[WARN] Skip invalid JSON: {fp} ({e})")
            continue

        for obj in iter_objects_from_any_json(data):
            if not isinstance(obj, dict):
                continue
            if obj.get("type") != "campaign":
                continue

            oid = obj.get("id")
            if oid:
                if oid in seen_ids:
                    # 이미 같은 캠페인 id가 있으면 스킵
                    print(f"[INFO] Skip duplicate campaign id: {oid} in file {fp}")
                seen_ids.add(oid)

            campaigns.append(obj)

    return campaigns

def make_bundle(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    STIX 2.0 bundle로 감싸기.
    """
    # 보기 좋게 name 기준 정렬(없으면 id 정렬)
    objects_sorted = sorted(objects, key=lambda o: (o.get("name") or "", o.get("id") or ""))
    return {
        "type": "bundle",
        "id": f"bundle--{uuid4()}",
        "spec_version": SPEC_VERSION,
        "objects": objects_sorted,
    }

def main():
    files = list(find_campaign_json_files(ROOTS))
    print(f"[INFO] Found {len(files)} JSON file(s).")

    campaigns = collect_campaigns(files)
    print(f"[INFO] Collected {len(campaigns)} campaign object(s) after de-dup.")

    bundle = make_bundle(campaigns)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote bundle to: {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
