# Vehicle Signal Processing — EKF 기반 배터리 내부 단락(ISC) 검출 시스템

> 기말 프로젝트 | 전기차 BMS 신호처리 및 딥러닝 기반 ISC 이상 탐지

---

## 프로젝트 개요

실제 전기차(EV) BMS 환경을 반영한 **리튬이온 배터리(NCM811) 내부 단락(ISC) 조기 검출 시스템**입니다.

3인 팀이 각자 개발한 알고리즘을 하나의 완성된 아키텍처로 통합하였으며, 두 가지 핵심 문제를 해결하였습니다.

1. **적정 노이즈 선정** — 실제 BMS 센서 오차(10 mV)를 반영한 AWGN 합성으로 강건성 검증
2. **Offline / Online 이원화 파이프라인** — 실시간 스트리밍(Online)과 사후 정밀 분석(Offline) 환경을 모두 지원

---

## 팀 역할 분담

| 역할 | 담당 모듈 |
|------|-----------|
| EKF 설계 및 SOC 추정 | `utils/ekf.py`, `utils/pipeline.py`, `utils/ocv.py`, `utils/ekf_eval.py` |
| 전처리 / 디지털 필터 | `utils/denoise.py`, `utils/data.py` |
| 주파수 변환 및 ISC 판별 | `utils/fft_processor.py`, `utils/autoencoder.py` |

---

## 전체 파이프라인 구조

```
[Raw 배터리 데이터 (전압 / 전류)]
          │
          ▼
  ┌───────────────┐
  │   OCV-SOC     │  CC 방전 데이터 + CV 앵커로 룩업테이블 구축
  │  테이블 구축  │  (IR 보정 포함, 7차 다항식 피팅)
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  EKF (1RC)   │  상태: [SOC, Vrc]  /  측정: 전압
  │  SOC 추정    │  Q/R 파라미터: NIS 기반 그리드서치로 최적화
  └───────┬───────┘
          │  Innovation (Residual)
          ▼
    ┌─────┴──────┐
    │            │
 [Online]    [Offline]
    │            │
    │     ┌──────────────┐
    │     │ Zero-phase   │  Butterworth LPF (sosfiltfilt)
    │     │ Butterworth  │  위상 지연 없음 / 미래 데이터 필요
    │     └──────┬───────┘
    │            │
    ▼            ▼
  ┌──────────────────────┐
  │   Welch PSD 변환     │  nperseg=256 → 129차원 PSD 벡터
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  PSD Autoencoder     │  정상 PSD로 학습 → 재구성 오차(MSE)로 이상 탐지
  │  (latent_dim = 4)    │  임계값: train_mean + 3 × train_std
  └──────────┬───────────┘
             │
             ▼
       [ISC 탐지 결과]
        10Ω / 100Ω / 1000Ω
```

---

## 핵심 기여 사항

### 1. 적정 노이즈(AWGN) 선정 및 강건성 검증

| 노이즈 수준 | Normal 오탐지 | ISC 100Ω 탐지 | ISC 1000Ω 탐지 | 결론 |
|---|---|---|---|---|
| No Noise (0 V) | 없음 | ✓ | ✓ | 현실 미반영 |
| **std = 0.01 V (10 mV)** | **없음** | **✓** | **✓** | **최적 (BMS 기준)** |
| std = 0.05 V (50 mV) | 오탐지 발생 | ✓ | ✓ | 특이도 저하 |
| std = 0.10 V (100 mV) | 없음 | ✗ | ✗ | 미세 단락 소멸 |

- 100 mV 노이즈는 임계값을 4배 이상 치솟게 하여 1000 Ω 미세 단락 신호를 소멸시킴
- **실제 상용 BMS 센서 오차(10 mV)를 기준으로 AWGN std = 0.01 V 선정**

### 2. Offline / Online 이원화 파이프라인

| 구분 | 전처리 | 특징 | 1000Ω 탐지 |
|---|---|---|---|
| **Online** | Raw Residual → Welch PSD | 실시간 스트리밍 가능 (Causal) | ✗ (MSE 비율 0.76) |
| **Offline** | Butterworth LPF → Welch PSD | 위상 지연 없음, 사후 분석 전용 | ✓ (MSE 비율 1.96) |

- Online: 10 Ω, 100 Ω ISC 실시간 탐지 가능
- Offline: 1000 Ω 미세 단락 포함 전 케이스 탐지, 정밀 사후 진단용

---

## 파일 구조

```
Vehicle-Signal-Processing/
│
├── utils/                        # 공통 알고리즘 모듈
│   ├── data.py                   # 충/방전 블록 로딩, 쿨롱 카운팅 SOC
│   ├── ekf.py                    # Extended Kalman Filter (1RC 등가회로)
│   ├── ekf_eval.py               # NIS / MAE / RMSE 평가 지표
│   ├── ocv.py                    # OCV-SOC 룩업테이블 구축 (IR 보정)
│   ├── pipeline.py               # EKF 전체 실행 파이프라인
│   ├── denoise.py                # Zero-phase Butterworth LPF (Offline)
│   ├── fft_processor.py          # Welch PSD 변환 (Online/Offline 공통)
│   ├── autoencoder.py            # PSD Autoencoder 모델 (PyTorch)
│   └── plotting.py               # 시각화 유틸
│
├── scripts/                      # 실행 스크립트
│   ├── cc_ekf_design.py          # [Step 1] EKF Q/R 그리드서치 및 최적화
│   ├── main_usage_example.py     # [Step 2] Denoise 단독 동작 확인
│   ├── visualize_pipeline.py     # [Step 3] 전체 파이프라인 시각화
│   ├── prepare_dataset.py        # [Step 4] PSD 데이터셋 사전 저장
│   ├── eval_1.2cc.py             # [Step 5] Autoencoder 최종 평가
│   └── final_report.py           # [보고서] 노이즈 비교 + 파이프라인 비교 시각화
│
├── results/                      # 실행 결과 자동 저장
│   ├── cc_best_ekf_params.csv
│   ├── final_fig1~4_*.png        # 그래프 이미지
│   └── final_table1~2_*.png      # 표 이미지
│
├── voltage_prediction_and_ISC_detection-V1.0/   # 데이터셋 (별도 다운로드)
├── requirements.txt
└── .gitignore
```

---

## 데이터셋

**다운로드:** [Zenodo — voltage_prediction_and_ISC_detection V1.0](https://zenodo.org/records/7703318)

다운받은 폴더를 프로젝트 루트에 위치시키세요.

```
Vehicle-Signal-Processing/
└── voltage_prediction_and_ISC_detection-V1.0/
    └── swhlqu-voltage_prediction_and_ISC_detection-dd56682/
        ├── NCM811_NORMAL_TEST/
        │   ├── CC/
        │   └── DST/
        └── NCM811_ISC_TEST/
            ├── CC/
            └── DST/
```

### 데이터셋 용어

| 용어 | 의미 |
|------|------|
| NCM811 | NCM811 계열 리튬이온 배터리 |
| Normal / ISC | 정상 조건 / 내부 단락(Internal Short Circuit) 조건 |
| CC | Constant Current — 정전류 충방전 |
| DST | Dynamic Stress Test — 실제 차량 모사 동적 전류 프로파일 |
| BD | Normal 데이터셋의 정상 배터리 구분명 |
| CS | ISC 데이터셋의 단락 배터리 구분명 |

### CSV 컬럼 구조

| 인덱스 | 컬럼 | 구분 |
|--------|------|------|
| 0 | 시간 [s] | 충전 |
| 1 | 전류 [A] | 충전 |
| 2 | 용량 [Ah] | 충전 |
| 3 | SOC\|DOD [%] | 충전 |
| 4 | 전압 [V] | 충전 |
| 5 | (구분 빈 컬럼) | — |
| 6 | 시간 [s] | 방전 |
| 7 | 전류 [A] | 방전 |
| 8 | 용량 [Ah] | 방전 |
| 9 | SOC\|DOD [%] | 방전 |
| 10 | 전압 [V] | 방전 |

---

## 설치 및 실행

### 환경 설정

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 실행 순서

#### Step 1 — EKF Q/R 파라미터 최적화
```powershell
python scripts\cc_ekf_design.py
```
- CC 방전 데이터로 OCV-SOC 테이블 생성
- Q/R 그리드서치 → NIS 기반 최적 파라미터 선정
- `results/cc_best_ekf_params.csv` 저장

#### Step 2 — Denoise 동작 확인 (선택)
```powershell
python scripts\main_usage_example.py
```
- Raw residual vs Filtered residual 비교 그래프 생성

#### Step 3 — 전체 파이프라인 시각화 (선택)
```powershell
python scripts\visualize_pipeline.py
```
- `EKF → Innovation → Denoise → Welch PSD` 흐름 4-panel 그래프

#### Step 4 — PSD 데이터셋 사전 저장 (선택)
```powershell
python scripts\prepare_dataset.py
```
- 전체 DST 데이터를 PSD 벡터로 변환 후 `.npy` 저장

#### Step 5 — Autoencoder ISC 탐지 최종 평가
```powershell
python scripts\eval_1.2cc.py
```
- 1.2CC DST 데이터 기준 학습/평가
- ISC 10 Ω / 100 Ω / 1000 Ω 탐지 결과 및 그래프 저장

#### 보고서 시각화 (기말 프로젝트 결과물)
```powershell
python scripts\final_report.py
```
- 노이즈 수준별 강건성 비교 (Figure 1, 2 + Table 1)
- Offline vs Online 파이프라인 비교 (Figure 3, 4 + Table 2)
- `results/final_*.png` 8개 파일 생성

---

## 주요 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| R0 | 0.015 Ω | 직렬 내부 저항 |
| R1 | 0.01 Ω | RC 병렬 저항 |
| C1 | 2400 F | RC 병렬 커패시터 |
| Butterworth cutoff | 0.05 Hz | Zero-phase LPF 차단 주파수 |
| Welch nperseg | 256 | PSD 세그먼트 길이 (→ 129차원) |
| Autoencoder latent_dim | 4 | 잠재 공간 차원 |
| ISC 탐지 임계값 | mean + 3σ | 학습 MSE 기준 |
| 최적 AWGN std | 0.01 V (10 mV) | BMS 센서 오차 반영 |

---

## 참조 데이터셋 논문

> *"Internal short circuit early detection and power prediction of lithium-ion batteries using deep learning"*
> Zenodo Dataset V1.0 — [https://zenodo.org/records/7703318](https://zenodo.org/records/7703318)
