import os

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

print(os.listdir())

for key, path in FILE_PATHS.items():
    if os.path.exists(path):
        print(f"[OK] {key}: {path} 파일 존재")
    else:
        print(f"[ERROR] {key}: {path} 파일 없음!")
