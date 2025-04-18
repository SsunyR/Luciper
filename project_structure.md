---

# ✅ Luciper 프로젝트 전체 구조

```
📦 Luciper/
├── data/
│   ├── raw/                  # 원본 텍스트, 영상, 오디오 파일 저장
│   ├── tts/                  # TTS로 생성된 audio 파일
│   ├── tts_text/             # TTS 생성 시 사용된 text 파일 (.txt)
│   ├── videos/               # 사용자 업로드 영상
│   ├── audios/               # 영상에서 추출된 오디오
│   ├── transcripts/          # Whisper 전체 텍스트 결과 (.txt)
│   ├── subtitles/            # Whisper 자막 + 사용자 수정 자막
│   └── dataset/              # fine-tuning용 정제된 데이터셋
│
├── preprocessing/
│   ├── clean_audio.py        # 노이즈 제거, 볼륨 정규화 등 전처리
│   ├── format_dataset.py     # Whisper 학습 포맷으로 변환
│   └── evaluate_alignment.py # 자막-음성 싱크 정렬 평가
│
├── tts/
│   ├── generate_tts.py       # 텍스트 → TTS 음성 생성
│   └── tts_config.yaml       # TTS 설정 (언어, 속도 등)
│
├── whisper/
│   ├── transcribe.py         # Whisper inference (영상 → 자막)
│   ├── train.py              # fine-tuning 스크립트
│   ├── config.yaml           # fine-tuning 하이퍼파라미터 설정
│   └── evaluate.py           # 생성 자막 평가 (WER, BLEU 등)
│
├── feedback/
│   ├── subtitle_editor_ui/   # 웹 UI (React or Streamlit 등)
│   ├── collect_feedback.py   # 수정된 자막 저장
│   └── update_dataset.py     # 피드백 반영해 학습용 데이터 갱신
│
├── api/
│   ├── main.py               # FastAPI 서버 (자막 생성, 수정 요청 등)
│   └── routes/
│       ├── upload.py         # 영상 업로드 및 처리 요청
│       └── subtitle.py       # 자막 생성/수정 결과 제공
│
├── evaluation/
│   ├── metrics.py            # WER, CER, BLEU 등의 평가 지표
│   └── visualization.py      # 평가 결과 시각화
│
├── utils/
│   ├── audio_utils.py        # 오디오 추출, 변환
│   ├── file_utils.py         # 파일 입출력 보조 함수
│   └── logging_utils.py      # 로그 관리
│
├── scripts/
│   ├── run_pipeline.sh       # 전체 파이프라인 실행 스크립트
│   └── docker_setup.sh       # Docker 기반 환경 구성 자동화
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔧 주요 개선 요소 반영 설명

| 범주             | 추가 요소 | 설명 |
|------------------|-----------|------|
| **전처리**        | `clean_audio.py` | TTS 및 영상 오디오에서 노이즈 제거, 볼륨 정규화 등 음질 향상 |
| **평가 지표**      | `metrics.py` / `evaluate.py` | WER, BLEU 등 다양한 평가 지표를 통해 자막 품질 수치화 |
| **자막 정렬 평가** | `evaluate_alignment.py` | 자막과 음성의 싱크 정확도 점검 |
| **데이터 시각화**   | `visualization.py` | 자막 수정 내역, 모델 성능 변화 등을 시각적으로 분석 |
| **경량 API 서버** | FastAPI | 영상 업로드, 자막 추론, 수정 내역 제출을 위한 REST API 제공 |
| **웹 UI**        | subtitle_editor_ui | 사용자 친화적인 자막 수정 인터페이스 제공 (React, Streamlit 등) |
| **전체 실행 스크립트** | `run_pipeline.sh` | TTS → 모델 학습 → 영상 처리 → 피드백 수집까지 일괄 처리 가능 |

---

## 🧪 전체 파이프라인 흐름 요약

```text
[TTS 입력 텍스트] → [TTS 생성 (gTTS 등)] → [오디오 정제]
                                   ↓
                [Whisper fine-tune 초기 학습 데이터]
                                   ↓
                [사용자 영상 업로드 및 자막 자동 생성]
                                   ↓
               [웹 UI로 자막 수정 → 피드백 데이터 생성]
                                   ↓
              [피드백 데이터 반영 → Whisper 재학습]
                                   ↓
         [WER/BLEU/정렬 정확도 평가 → 성능 개선 확인]
```

---

## 핵심 모듈 개발 순서 (우선순위)

    1. 유틸리티 함수 구현: 파일 처리, 오디오 추출 등의 기본 유틸리티
    2. TTS 모듈 구현: fine-tuning 데이터셋 생성을 위한 첫 단계
    3. 전처리 모듈: 오디오 클리닝 및 데이터셋 포맷 변환
    4. Whisper fine-tuning 스크립트: 모델 학습 파이프라인
    5. API 서버: 사용자 인터페이스 및 데이터 입출력 관리

