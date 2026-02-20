import json

def parse_mitre_stix(stix_path, output_path):
    print(f"📂 Loading STIX data from {stix_path}...")
    with open(stix_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)
    
    parsed_techniques = {}
    
    for obj in bundle.get('objects', []):
        # 기술(Technique) 객체만 필터링 (deprecated 제외)
        if obj.get('type') == 'attack-pattern' and not obj.get('x_mitre_deprecated', False):
            
            # T-ID 추출 (external_references 안에서 찾기)
            t_id = None
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    t_id = ref.get('external_id')
                    break
            
            if not t_id: continue
            
            # Data Sources 추출 (가장 중요!)
            data_sources = obj.get('x_mitre_data_sources', [])
            
            # 데이터 저장 구조
            parsed_techniques[t_id] = {
                "name": obj.get('name'),
                "description": obj.get('description', ''),
                "data_sources": data_sources, # 리스트 형태 그대로 저장
                "tactics": [t['phase_name'] for t in obj.get('kill_chain_phases', []) if t['kill_chain_name'] == 'mitre-attack']
            }
            
    # 결과를 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_techniques, f, indent=4)
        
    print(f"✅ Parsed {len(parsed_techniques)} techniques. Saved to {output_path}")

# 사용법
parse_mitre_stix("CASG_2509\data_att&ck\mobile-attack.json", "mobile_mitre_techniques_db.json")