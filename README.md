# Luciper 프로젝트

이 프로젝트는 Whisper 음성 인식 모델을 도메인에 맞게 간단하게 fine tuning 할 수 있도록 설계된 오픈 소스 프로젝트입니다. 초기에는 TTS를 통한 학습 데이터 생성, 이후 영상 자막 자동 생성 및 사용자 피드백을 통한 지속적 모델 개선 방식을 채택하였습니다.

## 주요 특징

- **모듈화된 구조:**  
  데이터 전처리, TTS 데이터 생성, Whisper 모델 학습, 자막 자동 생성 및 피드백 수집 모듈이 독립적으로 구성되어 있어 필요한 부분을 손쉽게 수정/확장할 수 있습니다.
  
- **간편한 환경 설정:**  
  Docker와 requirements.txt를 활용하여 복잡한 초기 환경 구성이 필요 없으며, GitHub에서 바로 포크하여 사용할 수 있습니다.
  
- **경량 데이터베이스:**  
  SQLite를 이용해 영상 업로드 내역, 자막 수정 데이터, 모델 학습 로그 등 필요한 메타데이터를 간단하게 관리합니다.
  
- **피드백 기반 재학습:**  
  사용자가 웹 UI를 통해 자동 생성된 자막을 수정하면 그 피드백 데이터가 모델 재학습에 활용되어 지속적인 성능 개선이 가능합니다.

## 전체 아키텍처

```text
[TTS 입력 텍스트] → [TTS 음성 생성 및 오디오 전처리]
                                    ↓
                [초기 Whisper fine-tuning 데이터셋 구성]
                                    ↓
                [사용자 영상 업로드 및 오디오 추출]
                                    ↓
                   [Whisper를 통한 자막 자동 생성]
                                    ↓
               [웹 UI를 통한 자막 편집 및 피드백 수집]
                                    ↓
                    [수정 데이터로 모델 재학습]
                                    ↓
         [평가 (WER, BLEU, 정렬 정확도) 및 성능 개선 피드백]
```

## 프로젝트 구조

```
Luciper/
├── data/
│   ├── raw/                  # 원본 텍스트, 영상, 오디오 파일 저장
│   ├── tts/                  # TTS로 생성한 음성 파일
│   ├── tts_text/             # TTS 생성 시 사용된 텍스트 파일 (.txt)
│   ├── videos/               # 사용자 업로드 영상
│   ├── audios/               # 영상에서 추출한 오디오
│   ├── transcripts/          # Whisper 전체 텍스트 결과 (.txt)
│   ├── subtitles/            # Whisper 타임스탬프 자막 (.srt) 및 사용자 수정 자막
│   └── dataset/              # fine-tuning용 정제된 데이터셋 (오디오, 메타데이터)
│
├── preprocessing/
│   ├── clean_audio.py        # 오디오 노이즈 제거, 정규화 등 전처리
│   ├── format_dataset.py     # Whisper 학습 포맷으로 데이터 변환
│   └── evaluate_alignment.py # 자막-오디오 싱크 평가 도구
│
├── tts/
│   ├── generate_tts.py       # 텍스트를 TTS 음성으로 변환
│   └── tts_config.yaml       # TTS 설정 파일 (언어, 속도 등)
│
├── whisper/
│   ├── transcribe.py         # Whisper를 통한 자막 자동 생성
│   ├── train.py              # fine-tuning 스크립트
│   ├── config.yaml           # fine-tuning 하이퍼파라미터 설정
│   └── evaluate.py           # 자막 품질 평가 (WER, BLEU 등)
│
├── feedback/
│   ├── subtitle_editor_ui/   # 자막 편집 웹 UI (React 또는 Streamlit)
│   ├── collect_feedback.py   # 사용자 수정 자막 저장
│   └── update_dataset.py     # 피드백 반영 재학습 데이터셋 업데이트
│
├── api/
│   ├── main.py               # FastAPI 기반 API 서버
│   └── routes/
│       ├── upload.py         # 영상 업로드 및 오디오 추출 처리
│       └── subtitle.py       # 자막 생성 및 수정 결과 제공
│
├── evaluation/
│   ├── metrics.py            # WER, CER, BLEU 등의 평가 지표 계산
│   └── visualization.py      # 평가 결과 시각화 도구
│
├── utils/
│   ├── audio_utils.py        # 오디오 추출 및 변환 도구 (FFmpeg 활용)
│   ├── file_utils.py         # 파일 입출력 및 경로 관리 보조
│   └── logging_utils.py      # 통합 로깅 관리
│
├── scripts/
│   ├── run_pipeline.sh       # 전체 파이프라인 실행 스크립트
│   └── docker_setup.sh       # Docker 환경 자동 구성 스크립트
│
├── requirements.txt          # Python 라이브러리 의존성 목록
├── Dockerfile                # Docker 이미지 빌드 파일
└── README.md                 # 이 파일
```

## 시작하기

### 1. 환경 설정

#### Docker를 이용한 설정 (권장)
```bash
# Docker 이미지 빌드
docker build -t whisper-finetune .
# 컨테이너 실행
docker run -p 8000:8000 -v $(pwd):/app whisper-finetune
```

#### 로컬 환경 설정
```bash
# Python 3.8 이상 필요
pip install -r requirements.txt
```

### 2. 초기 TTS 데이터 생성
```bash
# 기본 경로(config: tts/tts_config.yaml, input: data/raw/input.txt, audio output: data/tts/, text output: data/tts_text/) 사용 시
python tts/generate_tts.py

# 또는 특정 경로 지정 시
# python tts/generate_tts.py --config <path_to_config> --input_text <path_to_input> --output_audio_dir <path_to_audio> --output_text_dir <path_to_text>
```

### 3. 오디오 전처리 (선택 사항이지만 권장)
```bash
# data/tts/ 에 있는 오디오를 정제하여 data/dataset/audio_cleaned/ 에 저장
python preprocessing/clean_audio.py
```

### 4. 데이터셋 포맷팅 (메타데이터 생성)
```bash
# data/dataset/audio_cleaned/ 의 오디오와 data/tts_text/ 의 텍스트를 매핑하여 data/dataset/metadata.csv 생성
python preprocessing/format_dataset.py
```

### 5. Whisper 모델 Fine Tuning
```bash
# data/dataset/metadata.csv 를 사용하여 모델 학습
python whisper/train.py --config whisper/config.yaml
```

### 6. API 서버 실행 및 영상 업로드
- FastAPI 서버 실행 (프로젝트 루트 디렉토리에서):
  ```bash
  uvicorn api.main:app --reload --port 8000
  ```
  * `--reload` 옵션은 개발 중 코드 변경 시 서버 자동 재시작을 위함입니다.
- API 서버가 실행되면 (기본적으로 `http://127.0.0.1:8000`), 다음 엔드포인트를 사용할 수 있습니다:
  * **`POST /api/v1/upload/video`**: 영상 파일을 업로드합니다. (예: `curl -X POST -F 'file=@/path/to/your/video.mp4' http://127.0.0.1:8000/api/v1/upload/video`)
  * **`GET /api/v1/subtitle/{job_id}`**: (구현 예정) 특정 작업 ID의 자막 결과를 가져옵니다.
  * **`PUT /api/v1/subtitle/{job_id}`**: (구현 예정) 수정된 자막을 제출합니다.
  * **`GET /`**: API 루트 엔드포인트.

### 7. 자막 생성 및 피드백 (추후 구현)
- 업로드된 영상에 대해 오디오 추출 및 fine-tuning된 Whisper 모델을 사용한 자막 생성이 자동으로 트리거됩니다. (현재는 파일 저장만 구현됨)
- 웹 UI (`feedback/subtitle_editor_ui/`)를 통해 생성된 자막을 확인하고 수정합니다.
- 수정된 자막은 `PUT /api/v1/subtitle/{job_id}` 엔드포인트를 통해 제출되어 `feedback/` 모듈에서 처리됩니다.
- 수집된 피드백 데이터는 `feedback/update_dataset.py`를 통해 재학습용 데이터셋에 반영됩니다.
- 주기적으로 업데이트된 데이터셋으로 Whisper 모델 fine-tuning을 재실행합니다 (`python whisper/train.py ...`).

## 기여 방법

1. 본 저장소를 fork하여 로컬 환경에서 개발을 진행합니다.
2. 기능 추가 또는 버그 수정 후 pull request를 생성합니다.
3. 추가적인 문서나 테스트 코드도 함께 업데이트 부탁드립니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.