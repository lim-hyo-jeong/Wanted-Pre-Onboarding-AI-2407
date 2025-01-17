# Wanted-Pre-Onboarding-AI-2407
## 원티드 프리온보딩 AI 챌린지 2407 <프롬프트 엔지니어로 온보딩하기>

![pe](https://github.com/user-attachments/assets/5402c25e-af20-41d0-acea-1e0b16cdce40)

<br><br>

## 강의 커리큘럼 (2024.07.06~2024.07.14)
![curriculum](https://github.com/user-attachments/assets/15e16f59-9f80-4361-b2fe-09817f845bb0)

<br><br>

### 강의 자료 (.pdf) 
1. [Week 1-1. 프롬프트 엔지니어의 업무 및 노하우.pdf](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/blob/main/w1-1/%5B240706%5D%20w1-1%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EC%9D%98%20%EC%97%85%EB%AC%B4%20%EB%B0%8F%20%EB%85%B8%ED%95%98%EC%9A%B0.pdf)
2. [Week 1-2. 프롬프트 엔지니어링 실습: 작업 생산성 높이기.pdf](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/blob/main/w1-2/%5B240707%5D%20w1-2%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81%20%EC%8B%A4%EC%8A%B5%20-%20%EC%9E%91%EC%97%85%20%EC%83%9D%EC%82%B0%EC%84%B1%20%EB%86%92%EC%9D%B4%EA%B8%B0.pdf)
3. [Week 2-1. 프롬프트 엔지니어링 심화 실습: LLM Agent 만들기.pdf](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/blob/main/w2-1/%5B240713%5D%20w2-1%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81%20%EC%8B%AC%ED%99%94%20%EC%8B%A4%EC%8A%B5%20-%20LLM%20Agent%20%EB%A7%8C%EB%93%A4%EA%B8%B0.pdf)
4. [Week 2-2. 프롬프트 버전 관리와 테스트 방법.pdf](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/blob/main/w2-2/%5B240714%5D%20w2-2%20%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8%20%EB%B2%84%EC%A0%84%20%EA%B4%80%EB%A6%AC%EC%99%80%20%ED%85%8C%EC%8A%A4%ED%8A%B8%20%EB%B0%A9%EB%B2%95.pdf)

<br><br>

## 강의 자료 (실습코드)
1. [Week 1-2. 프롬프트 엔지니어링 실습: 작업 생산성 높이기 실습 코드](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/tree/main/w1-2/practice)
2. [Week 2-1. 프롬프트 엔지니어링 심화 실습: LLM Agent 만들기 실습 코드](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/tree/main/w2-1/practice)
3. [Week 2-2. 프롬프트 버전 관리와 테스트 방법 실습 코드](https://github.com/lim-hyo-jeong/Wanted-Pre-Onboarding-AI-2407/tree/main/w2-2/practice)

<br><br>


## 실습 코드 실행 방법
### Week 1-2. 실습 코드 실행 방법
1. OpenAI API Key 발급
    - OpenAI API Key 발급 url: https://platform.openai.com/api-keys 
    - OpenAI API Key 세팅: 업로드한 practice 실습 디렉토리의 각 3가지 예제 디렉토리 안에 .env 파일을 만들고 키 세팅
    ```
    OPENAI_API_KEY = "키 입력"
    ```
2. Anaconda 가상환경 생성 
    - 업로드한 practice 실습 디렉토리에서 다음 명령어 실행
    ```
    conda env create -f environment.yml
    ```
3. Anaconda 가상환경 실행 
    - 다음 명령어 실행
    ```
    conda activate wanted
    ```
3. p1 보고서 키워드 추출 및 요약 실습: 주피터 노트북 실행
4. p2 페르소나 챗봇 만들기 실습: p2_persona_chatbot 디렉토리로 이동하여 다음 명령어 실행
    ```
    streamlit run app.py
    ```
5. p3 도서 베스트셀러 예측을 통한 판매량 증진 전략 실습: 주피터 노트북 실행 

<br>

### Week 2-1. 실습 코드 실행 방법
1. OpenAI API Key 세팅
    - 업로드한 practice 실습 디렉토리의 각 p1, p2 예제 디렉토리 안에 .env 파일을 만들고 키 세팅
    ```
    OPENAI_API_KEY = "키 입력"
    ```
2. LangChain API Key 발급
    - LangChain API Key 발급 url: https://www.langchain.com/langsmith
    - LangChain API Key 세팅: 업로드한 practice 실습 디렉토리의 p3 예제 디렉토리 안에 .env 파일을 만들고 다음과 같이 세팅 (OPENAI_API_KEY와 함께 세팅하면 됩니다)
    ```
    OPENAI_API_KEY = "OpenAI API 키 입력"
    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY = "랭체인 API 키 입력"
    LANGCHAIN_PROJECT = "프로젝트 이름 입력"
    ```
3. Anaconda 가상환경 생성 
    - 업로드한 practice 실습 디렉토리에서 다음 명령어 실행
    ```
    conda env create -f environment.yml
    ```
4. Anaconda 가상환경 실행 
    - 다음 명령어 실행
    ```
    conda activate wanted2
    ```
5. p1 OpenAI Assistants API 활용 실습 (금융보고서 QA): 주피터 노트북 실행
6. p2 LangChain Tools 실습 (90년대 인기 애니메이션의 한국어 OST 유튜브 링크 가져오기): 주피터 노트북 실행
7. p3 LangChain Agent 실습: p3_langchain_agent 디렉토리로 이동하여 다음 명령어 실행
    ```
    streamlit run app.py
    ```

<br>

### Week 2-2. 실습 코드 실행 방법
1. OpenAI API Key, LangChain API Key 세팅 
    - 업로드한 practice 실습 디렉토리의 p1 예제 디렉토리 안에 .env 파일을 만들고 다음과 같이 세팅 
    ```
    OPENAI_API_KEY = "OpenAI API 키 입력"
    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY = "랭체인 API 키 입력"
    LANGCHAIN_PROJECT = "프로젝트 이름 입력"
    ```
3. Anaconda 가상환경 실행 
    - 다음 명령어 실행
    ```
    conda activate wanted2
    ```
4. p1 LangSmith Prompt Versioning 실습: 주피터 노트북 실행

<br><br>

## 사전과제 

1. 다양한 프롬프팅 기법 중 하나를 선택하여 그 기법의 개념과 예시를 설명하세요. (예: Few-Shot, Chain-of-Thought, ReAct 등)
2. 프롬프트 엔지니어링 기법 중 궁금하거나 좀 더 구체적으로 알고 싶은 부분이 있다면 작성해주세요.
3. 어떤 작업을 할 때 LLM을 자주 사용하는지 작성해주세요. (예: 코드 개선, 아이디에이션, 문서 요약, 보고서 초안 생성 등) 
4. 평소에 LLM을 사용하면서 어떤 점이 가장 어려웠나요? (예: 원하는 결과가 도출되지 않음, 출력 포맷팅, API 사용 방법 등)
5. 이 강의에서 기대하는 것과 얻고 싶은 것이 무엇인지 자유롭게 작성해주세요. 
6. [준비 사항] 실습을 위하여 OpenAI API Key가 필요합니다. 강의 전 OpenAI API Key를 미리 발급받아 준비해주세요.

**과제 제출 방법**

1. 레포지토리의 `Issues` 탭에서 `New issue` 를 클릭합니다.
2. 사전과제 템플릿에 대하여 `Get Started` 버튼을 클릭합니다.
3. 제목에서 제출일과 이름 부분을 수정합니다.
4. 사전 과제의 답변을 작성합니다.
5. `Submit new issue` 를 클릭해서 제출합니다.

<br><br>

## For Feedback
* Email: lim.gadi@gmail.com
* GitHub: [@lim-hyo-jeong](https://github.com/lim-hyo-jeong)
* Linkedin: https://www.linkedin.com/in/lim-hyo-jeong/