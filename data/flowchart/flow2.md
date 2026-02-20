graph LR
    %% 노드 스타일 정의
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef llm fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef python fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% 1. 데이터 소스 그룹
CampData[("📂 Campaign Data<br>📋 Surface/Vector List<br>📚 CTI & KEV DB<br>🔍 CAPEC Vector DB<br>📜 Prompt File<br>태그 지시(<'description' 등>)")]:::input

    %% 2. 실행 파이프라인
    subgraph Pipeline [Process Flow]
        
        %% Step 1: 기획
        note1[/"⚙️ User Config<br>(산업군/난이도/복잡도)"/] -.-> Step1_Plan
        CampData --> Step1_Plan
        Step1_Plan{{"🧠 Step 1: 기획<br>(LLM)<br>TTPs + Surface/Vector 선택"}}:::llm

        %% Step 2: 보강
        Step1_Plan --"Draft Plan"--> Step2_Enrich[("🛠️ Step 2: 데이터 보강<br>(Python 코드 기반 필터링)<br>1. Map CVE/CWE (검색)<br>2. Retrieve CAPEC (RAG)<br>3. 사용된 Surface/Vector 필터링")]:::python
        
        CTI_DB --"ID Matching (Technique ID)"--> Step2_Enrich
        RAG_DB --"Vector Search (CAPEC 데이터)"--> Step2_Enrich

        %% Step 3: 생성
        Step2_Enrich --"전체 구조 전달"--> Step3_Gen{{"✍️ Step 3: 시나리오 생성<br>(LLM)"}}:::llm
        
        note3[/"프롬프트 + XML 태깅"/] -.-> Step3_Gen
    end

    %% 3. 결과
    Step3_Gen --> Final_XML("📄 Final Scenario (XML)")