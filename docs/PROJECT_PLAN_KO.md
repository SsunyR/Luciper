# 프로젝트 계획서: Luciper

## 1. 프로젝트 개요

* **목표:** 사용자가 특정 도메인(주제, 환경, 발음 등)에 특화된 OpenAI Whisper 모델을 손쉽게 생성하고 활용할 수 있는 오픈소스 서비스 개발
* **핵심 기능:**
    * 다양한 Whisper 기반 모델(`tiny`, `base`, `small`, `medium`, `large` 등 최신 버전 포함) 선택 및 Fine-tuning
    * 텍스트 데이터를 활용한 TTS(Text-to-Speech) 기반 초기 Fine-tuning 및 지속 학습
    * Fine-tuned 모델을 이용한 음성 파일 자막 생성
    * 생성된 자막 수정 및 이를 활용한 지속적인 모델 성능 개선 파이프라인 구축
    * Fine-tuned 모델 및 관련 메타데이터(별칭, 도메인 설명 등) 저장, 관리, 수정 기능
    * 생성된 자막과 수정된 자막 비교를 통한 정확도(예: WER - Word Error Rate) 확인 기능
* **개발 언어:** `Python`
* **개발 방식:** AI 에이전트(Vibe coding)를 활용한 페어 프로그래밍
* **패키지 관리:** `uv` 사용
* **아키텍처:** 기능별 모듈화 구조 (명확한 Input/Output 정의)
* **테스트:** 각 모듈 개발 시 `Jupyter Notebook`을 활용한 기능 검증

## 2. 목표 사용자

* 특정 도메인의 음성 인식이 필요한 개발자 또는 연구자
* 자신만의 고품질 음성 인식 모델을 구축하고 싶은 개인 또는 팀
* 반복적인 자막 생성 및 수정 작업을 통해 모델 성능을 점진적으로 개선하고자 하는 사용자
* 코딩 경험이 적더라도 GUI를 통해 Whisper Fine-tuning을 경험하고 싶은 사용자

## 3. 기술 스택 (최소 최적 구성 제안)

* **Core Language:** `Python 3.10+`
* **ASR Model:** OpenAI Whisper (Hugging Face `transformers` 라이브러리 내 Whisper 구현체 사용 - `transformers`가 Fine-tuning 및 파이프라인 구축에 더 용이할 수 있음)
* **Fine-tuning Framework:** Hugging Face `transformers` + `accelerate` + `datasets` (Fine-tuning 파이프라인, 데이터셋 처리, 분산 학습 지원에 효율적)
* **TTS Engine:**
    * `Piper TTS`: Mozilla에서 개발한 빠르고 로컬에서 동작하는 고품질 TTS 엔진. 다양한 언어 및 목소리 지원. (오프라인 Fine-tuning 데이터 생성에 적합)
    * `gTTS`: Google Translate API 기반, 간단하지만 온라인 연결 필요 및 제약 존재. (초기 테스트용으로 사용 가능)
* **Web Framework & UI:** `Gradio`
    * Python 네이티브, 머신러닝 모델 데모 및 인터페이스 구축에 매우 용이.
    * 오디오 입력, 텍스트 입력/출력, 버튼 등 UI 컴포넌트를 쉽게 구현 가능.
    * "최소한의 기술 스택" 요구사항 충족 및 빠른 프로토타이핑 가능.
* **Package Manager:** `uv`
* **Data Handling:** `Pandas` (메타데이터 관리), Hugging Face `datasets` (오디오/텍스트 데이터셋 구성 및 처리)
* **Audio Processing:** `librosa` 또는 `soundfile` (오디오 파일 로드, 전처리)
* **Notebook Environment:** `Jupyter Notebook` / `JupyterLab` (모듈 테스트 및 실험)
* **Storage:**
    * 모델 파일: 로컬 파일 시스템 또는 클라우드 스토리지(선택 사항)
    * 메타데이터: `JSON` 파일 또는 `SQLite` (초기에는 JSON으로 단순하게 시작, 추후 확장성 고려 시 SQLite)

### 선정 이유:

* `transformers` 라이브러리는 Whisper Fine-tuning을 위한 `Trainer` API, 데이터셋 처리 등 편리한 기능을 다수 제공하여 개발 생산성을 높입니다.
* `Piper TTS`는 로컬 환경에서 고품질의 TTS 생성이 가능하여 외부 API 의존성 없이 Fine-tuning용 데이터셋 구축이 가능합니다.
* `Gradio`는 복잡한 프론트엔드 개발 없이 Python 코드만으로 인터랙티브한 웹 UI를 빠르게 구축할 수 있어 최소 기술 스택 및 AI 페어 프로그래밍 환경에 적합합니다.
* `uv`는 빠른 속도의 차세대 Python 패키지 관리자로 최신 트렌드를 반영합니다.

## 4. 시스템 아키텍처 (모듈식 구조)
    
```
+-------------------------+      +-------------------------+      +----------------------------+
|   Web Interface (Gradio)|----->| Model Management Module|----->| Fine-tuning Module         |
| - 모델 생성/선택        |<-----| - Base 모델 선택        | <----- | - 데이터 로드/전처리       |
| - 메타데이터 관리       |       | - Fine-tuned 모델 저장/로드|    | - Whisper Fine-tuning      |
| - 작업 선택 (학습/추론) |       | - 메타데이터 CRUD       |       | (transformers.Trainer)     |
| - 자막 생성/수정/비교   |      +-------------------------+       | - 모델 저장                |
| - TTS 학습 데이터 생성  |                |                       +------------+---------------+
+-----------+-------------+               |                                     |
|                                         | (Model Path/Object)                 | (Fine-tuned Model)
|                                         V                                     V
+-----------+-------------+      +-------------------------+      +----------------------------+
| Data Preparation Module |<-----| Inference Module        |----->| Evaluation Module         |
| - Text -> TTS 오디오    |       | - 오디오 로드           |      | - 생성 자막 vs 수정 자막   |
| - 오디오 + 텍스트 정렬  |        | - 선택 모델 로드        |      | - WER 등 정확도 계산       |
| - 자막 수정 데이터 처리 |----->  | - 자막 생성 (Whisper)   |----->| - 결과 시각화/표시         |
+-------------------------+      +-------------------------+      +----------------------------+
       (Training Data)          (Generated Subtitles)              (Accuracy Metrics)

```

### 모듈 설명:

* **Web Interface (Gradio):** 사용자 인터페이스 제공. 사용자의 입력을 받아 다른 모듈 호출 및 결과 표시.
    * `Input:` 사용자 액션 (파일 업로드, 텍스트 입력, 버튼 클릭 등)
    * `Output:` UI 업데이트 (모델 목록, 자막 텍스트, 정확도 등)
* **Model Management Module:** Whisper Base 모델 선택, Fine-tuned 모델 및 메타데이터 저장/로드/관리.
    * `Input:` 모델 이름, 경로, 메타데이터 정보
    * `Output:` 모델 객체, 모델 파일 경로, 저장/수정된 메타데이터
* **Data Preparation Module:** Fine-tuning을 위한 데이터셋 생성 및 준비.
    * `Input:` 텍스트 (TTS용), 오디오 파일 + 수정된 자막 텍스트
    * `Output:` Whisper Fine-tuning에 적합한 형식의 데이터셋 (오디오 + 텍스트 쌍)
* **Fine-tuning Module:** 실제 모델 Fine-tuning 수행. (`transformers`의 `Trainer` 활용)
    * `Input:` 준비된 데이터셋, 모델 경로(Base 또는 이전에 Fine-tuned된 모델), 학습 설정값(Hyperparameters)
    * `Output:` Fine-tuned된 모델 파일 경로
* **Inference Module:** 선택된 모델을 사용하여 오디오 파일로부터 자막 생성.
    * `Input:` 오디오 파일 경로, 사용할 모델 경로
    * `Output:` 생성된 자막 텍스트 (또는 SRT 등 파일 형식)
* **Evaluation Module:** 생성된 자막과 사용자가 수정한 자막을 비교하여 정확도(`WER` 등) 계산.
    * `Input:` 생성된 자막 텍스트, 수정된 자막 텍스트
    * `Output:` 정확도 지표 (숫자 또는 시각화 데이터)

## 5. 웹 인터페이스 상세 기능

* **모델 관리:**
    * `[생성]` 새 Fine-tuning 모델 세션 시작 (Base 모델 선택, 별칭/도메인 등 메타데이터 입력)
    * `[선택]` 기존에 저장된 Fine-tuned 모델 목록 표시 및 선택 기능
    * `[수정]` 선택된 모델의 메타데이터(별칭, 도메인 등) 수정 기능
    * `[삭제]` 선택된 Fine-tuned 모델 삭제 기능 (주의 필요)
* **작업 선택:**
    * 모델 선택 후, 해당 모델로 수행할 작업 선택:
        * `[TTS 기반 학습]`: 입력된 텍스트를 TTS로 변환하여 Fine-tuning 수행
        * `[자막 생성]`: 오디오/비디오 파일 업로드하여 자막 생성 후 결과물 수정
* **TTS 기반 학습:**
    * 텍스트 입력 영역 (대량 텍스트 붙여넣기 또는 텍스트 파일 업로드)
    * 사용할 TTS 엔진 선택 (`Piper`, `Coqui` 등) 및 목소리 선택 옵션 (가능하다면)
    * Fine-tuning 시작 버튼
    * 학습 진행 상태 표시 (로그 또는 Progress bar)
* **자막 생성:**
    * 오디오/비디오 파일 업로드 인터페이스
    * 자막 생성 시작 버튼
    * 생성된 자막 표시 영역
* **자막 수정 및 평가:**
    * 생성된 자막을 편집할 수 있는 텍스트 영역 제공
    * 원본(생성된 자막)과 수정본(편집된 자막) 비교 표시 기능
    * `[정확도 확인]` 버튼: `WER` 등 계산 결과 표시
    * `[수정 내용으로 학습]` 버튼: 수정된 자막 데이터를 활용하여 현재 선택된 모델을 추가 Fine-tuning하는 파이프라인 실행

## 6. 개발 계획 (단계별 접근 - Vibe Coding 활용)

* **Phase 1: 환경 설정 및 기본 기능 구현**
    * 프로젝트 구조 설정 (`src`, `tests`, `notebooks`, `data` 등)
    * `uv` 를 이용한 가상환경 설정 및 기본 라이브러리 설치 (`python`, `openai-whisper` 또는 `transformers`, `torch`, `jupyterlab`, `uv`)
    * `Jupyter Notebook`에서 기본 Whisper 모델 로드 및 자막 생성 기능 테스트
    * 간단한 오디오 파일로 자막 생성 모듈 (`Inference Module` 기초) 구현 및 테스트
* **Phase 2: Fine-tuning 파이프라인 구축**
    * Hugging Face `transformers` 와 `datasets` 를 이용한 Fine-tuning 기본 파이프라인 구축 (샘플 데이터셋 활용)
    * `Fine-tuning Module` 구현 및 Notebook 기반 테스트
    * GPU 환경 설정 및 테스트 (Fine-tuning은 GPU 필수)
* **Phase 3: 데이터 준비 기능 구현**
    * 텍스트 입력 -> TTS 변환 -> 오디오 파일 생성 기능 구현 (`Data Preparation Module` - TTS 파트)
    * `Piper TTS` 연동 및 Notebook 테스트
    * 생성된 TTS 데이터셋을 Fine-tuning 파이프라인에 연결하는 로직 구현
* **Phase 4: 모델 관리 기능 구현**
    * Fine-tuned 모델 저장 및 로드 기능 구현 (`Model Management Module` 핵심 기능)
    * 모델 메타데이터 (`JSON` 또는 `SQLite`) 저장/로드/수정 기능 구현
    * Notebook 기반 테스트
* **Phase 5: Gradio 웹 인터페이스 구축 (기본)**
    * `Gradio` 설치 및 기본 앱 구조 생성 (`Web Interface Module`)
    * 모델 선택 (Base, Fine-tuned 목록) 기능 구현
    * 오디오 파일 업로드 및 자막 생성 요청 -> `Inference Module` 연동 -> 결과 표시 기능 구현
    * 텍스트 입력 -> TTS 기반 학습 요청 -> `Data Preparation` 및 `Fine-tuning Module` 연동 기능 구현 (백그라운드 실행 필요)
* **Phase 6: 자막 수정 및 지속 학습 파이프라인 완성**
    * `Gradio` 인터페이스에 자막 편집 기능 추가
    * 생성 자막 vs 수정 자막 비교 및 정확도 계산 기능 (`Evaluation Module`) 연동 및 결과 표시
    * 수정된 자막 데이터를 `Data Preparation Module` 로 전달하고, 이를 이용해 `Fine-tuning Module` 을 다시 실행하는 지속 학습 파이프라인 완성
* **Phase 7: 테스트, 리팩토링 및 문서화**
    * 각 모듈 및 전체 워크플로우에 대한 통합 테스트
    * 코드 리팩토링 및 Vibe coding 과정에서의 피드백 반영
    * `README` 작성 (프로젝트 소개, 설치 방법, 사용법, 기여 방법 등)
    * 주석 및 코드 문서화 개선

## 7. 테스트 전략

* **모듈 단위 테스트:** 각 모듈의 핵심 기능은 `Jupyter Notebook`을 이용하여 다양한 입력값에 대해 예상대로 동작하는지 검증합니다. (AI 페어 프로그래밍 파트너와 함께 테스트 케이스 작성 및 실행)
* **통합 테스트:** `Gradio` 웹 인터페이스를 통해 전체 워크플로우(데이터 입력 -> 학습/추론 -> 결과 확인 -> 수정 -> 재학습)가 정상적으로 동작하는지 확인합니다.
* **데이터 테스트:** 다양한 종류의 오디오 파일(깨끗한 음성, 소음 환경, 다른 억양 등)과 텍스트 데이터를 사용하여 시스템의 강건성을 테스트합니다.

## 8. 추가 고려 사항

* **하드웨어 요구사항:** Whisper Fine-tuning은 상당한 컴퓨팅 자원(특히 GPU 메모리)을 요구합니다. 사용자에게 최소/권장 GPU 사양을 안내해야 합니다. (예: NVIDIA GPU VRAM 8GB 이상 권장, `Large` 모델은 더 필요)
* **오류 처리:** 파일 입출력 오류, 모델 로딩 실패, 학습 중단 등 발생 가능한 예외 상황에 대한 처리가 필요합니다.
* **비동기 처리:** 웹 인터페이스에서 Fine-tuning과 같이 시간이 오래 걸리는 작업은 비동기적으로 처리하여 사용자 경험을 개선해야 합니다. (`Gradio`는 자체적으로 처리하거나 `asyncio` 등을 활용)

## 9. 라이선스

* MIT