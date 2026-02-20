from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import difflib

class ContextExtractor:
    def __init__(self, techniques_db):
        self.descriptions = []
        self.t_ids = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        
        # 1. 데이터 준비 및 불용어 설정
        # (보안 도메인에서 변별력이 없는 단어들 제거)
        self.custom_stop_words = [
            'adversaries', 'adversary', 'attack', 'may', 'use', 'using', 'used', 
            'target', 'system', 'systems', 'network', 'information', 'data', 
            'access', 'security', 'example', 'methods', 'method', 'technique',
            'techniques', 'perform', 'order', 'windows', 'code', 'function'
        ]
        
        self._prepare_data(techniques_db)
        self._compute_tfidf()

    def _prepare_data(self, db):
        for t_id, info in db.items():
            desc = info.get('description', '')
            if desc:
                self.t_ids.append(t_id)
                self.descriptions.append(desc)

    def _compute_tfidf(self):
        print("📊 [Init] TF-IDF 분석 모델 학습 중...")
        # 2. 벡터화 (stop_words='english' + 커스텀 불용어)
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            ngram_range=(1, 2) # "access token" 같은 2단어 조합도 허용
        )

        # scikit-learn 기본 english에 의존하고, 결과에서 필터링하는 방식 사용.
        self.tfidf_matrix = self.vectorizer.fit_transform(self.descriptions)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def extract_keywords(self, t_id, top_n=5):
        """특정 T-ID의 설명에서 TF-IDF 점수가 가장 높은 키워드 추출"""
        try:
            idx = self.t_ids.index(t_id)
        except ValueError:
            return []

        # 해당 문서의 TF-IDF 점수 가져오기
        feature_index = self.tfidf_matrix[idx, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [self.tfidf_matrix[idx, x] for x in feature_index])
        
        # 점수 내림차순 정렬
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        
        # 상위 N개 단어 추출 (커스텀 불용어 제외)
        top_keywords = []
        for idx, score in sorted_scores:
            word = self.feature_names[idx]
            if word not in self.custom_stop_words:
                top_keywords.append(word)
            if len(top_keywords) >= top_n:
                break
                
        return top_keywords

    def infer_surface_vector(self, t_id, technique_name):
        """
        TF-IDF 키워드를 기반으로 Surface/Vector 결정
        """
        keywords = self.extract_keywords(t_id)
        
        # 1. 키워드 기반 매핑 (동적 매핑)
        # 추출된 '통계적으로 중요한 단어'가 무엇인지에 따라 분류
        detected_surface = None
        
        # 매핑 룰 (키워드 -> 표준 Surface)
        # 키워드는 단수형/복수형 등이 섞여있을 수 있으므로 부분일치 확인
        rules = { # https://attack.mitre.org/datasources/ 참조
            "registry": "Windows Registry",
            "token": "OS Access Token",
            "process": "OS Process",
            "service": "Windows Service",
            "driver": "Kernel Driver",
            "firmware": "System Firmware",
            "bios": "System Firmware",
            "account": "User Account",
            "credential": "Credential Store",
            "password": "Credential Store",
            "file": "File System",
            "directory": "File System",
            "folder": "File System",
            "powershell": "PowerShell Interpreter",
            "cmd": "Command Prompt",
            "script": "Script File",
            "browser": "Web Browser",
            "cloud": "Cloud Instance",
            "container": "Container",
            "dll": "Dynamic Link Library",
            "image": "System Image"
        }
        
        # TF-IDF 상위 키워드 순회하며 매핑 시도
        found_keywords = []
        for word in keywords:
            for rule_key, surface_val in rules.items():
                if rule_key in word:
                    detected_surface = surface_val
                    found_keywords.append(word)
                    break
            if detected_surface: break
        
        # 2. 매핑 실패 시: TF-IDF 최상위 키워드를 그대로 Surface로 사용 (가장 동적인 부분!)
        if not detected_surface:
            if keywords:
                # 예: "pipe" -> "Pipe" (규칙에 없던 새로운 자산 발견)
                detected_surface = f"Target Artifact ({keywords[0].capitalize()})"
            else:
                detected_surface = "Target System Artifact"
        
        # 3. Vector 구성: 기술명 + (주요 키워드)
        # 예: "Modify Registry (via keys)"
        vector_suffix = f" (via {found_keywords[0]})" if found_keywords else ""
        detected_vector = technique_name + vector_suffix
        
        return detected_surface, detected_vector