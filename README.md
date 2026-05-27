# 전체 파이프 라인 
전체 파이프라인

## 파일 구조
```text
Vehicle-Signal-Processing
|
|
|
|--- voltage_prediction_and_ISC_detection-V1.0\swhlqu-voltage_prediction_and_ISC_detection-dd56682\
|    NCM523_dataset\
|    NCM811_full_life_cycle_dataset\
|    NCM811_ISC_TEST\
|    NCM811_NORMAL_TEST\
|    NCM811_Random_test\
|    ...
|
|
|
|--- utils\
|    data.py
|    ekf.py
|    ocv.py
|    pipeline.py
|    plotting.py
|
|
|
|--- scripts\
|    main_usage_example.py
|
|
|
|--- images\
|    시각화 이미지 넣을 폴더
|
|
|--- README.md
|
|
|
|--- requirements.txt
```

## 데이터셋 구조
### 데이터셋 다운로드 링크
[[https://zenodo.org/records/7703318]]

다운받아서 현재 폴더로 옮겨주세요.

### CSV 컬럼 위치

| 컬럼 인덱스 | 원본 컬럼 | 영어 매칭 | 구분 |
|---:|---|---|---|
| 0 | `测试时间/Sec` | `charge_time_sec` | 충전 |
| 1 | `电流/A` | `charge_current_a` | 충전 |
| 2 | `容量/Ah` | `charge_capacity_ah` | 충전 |
| 3 | `SOC\|DOD/%` | `charge_soc_dod_percent` | 충전 |
| 4 | `电压/V` | `charge_voltage_v` | 충전 |
| 5 | 빈 컬럼 | - | 구분용 |
| 6 | `测试时间/Sec.1` | `discharge_time_sec` | 방전 |
| 7 | `电流/A.1` | `discharge_current_a` | 방전 |
| 8 | `容量/Ah.1` | `discharge_capacity_ah` | 방전 |
| 9 | `SOC\|DOD/%.1` | `discharge_soc_dod_percent` | 방전 |
| 10 | `电压/V.1` | `discharge_voltage_v` | 방전 |

### 데이터셋 용어
|용어|의미|
|---|---|
|NCM523|NCM523 계열 리튬이온 배터리 데이터셋|
|NCM811|NCM811 계열 리튬이온 배터리 데이터셋|
|full life cycle|배터리 전체 수명 주기 데이터를 포함한 데이터셋|
|Normal|내부 단락이 없는 정상 조건 데이터|
|ISC|Internal Short Circuit, 내부 단락 조건 데이터|
|Random|랜덤 전류 프로파일 조건 데이터|
|CC|Constant Current, 정전류 충방전 조건| 일정 전류 실험 
|DST|Dynamic Stress Test, 동적 전류 프로파일 조건| 실제 차량처럼 전류가 계속 변하는 실험
|BD|Normal dataset에서 사용되는 정상 배터리 데이터 구분명|
|CS|ISC test dataset에서 사용되는 내부 단락 배터리 데이터 구분명|

#### 실행 순서

1. EKF Q/R 설계

```powershell
python scripts\cc_ekf_design.py
```

역할:
- CC 데이터들을 불러옵니다.
- OCV-SOC 테이블을 만듭니다.
- 여러 Q/R 조합으로 EKF를 반복 실행합니다.
- NIS, MAE, RMSE 기준으로 최적 Q/R을 찾습니다.
- `results/cc_qr_grid_search.csv`와 `results/cc_best_ekf_params.csv`를 저장합니다.

이 단계는 뒤의 `visualize_pipeline.py`, `prepare_dataset.py`, `eval_1.2cc.py`가 최적 EKF 파라미터를 읽기 위한 기준 단계입니다.

2. denoise 단독 확인

```powershell
python scripts\main_usage_example.py
```

역할:
- 정상 CC 데이터 하나를 불러옵니다.
- EKF를 실행해서 `residual`을 만듭니다.
- `denoise_innovation()`을 적용해서 `filtered_residual`을 만듭니다.
- `images/denoise_result.png`에 raw residual과 filtered residual 비교 그래프를 저장합니다.

이 파일은 Autoencoder 학습용이 아니라 전처리 함수가 제대로 동작하는지 보는 예제입니다.

3. 전체 파이프라인 시각화

```powershell
python scripts\visualize_pipeline.py
```

역할:
- `cc_ekf_design.py`가 저장한 최적 Q/R을 읽습니다.
- ISC 1.2CC 10ohm DST 데이터 하나에 EKF를 적용합니다.
- raw innovation, denoised innovation, filtered PSD를 한 그림으로 확인합니다.
- `results/pipeline_test_1.2cc_denoised.png`를 저장합니다.

보고서에는 이 그림을 넣으면 `EKF → innovation → denoise → PSD` 흐름 설명에 좋습니다.

4. PSD 데이터셋 저장

```powershell
python scripts\prepare_dataset.py
```

역할:
- Normal DST 전체와 ISC DST 전체를 돌립니다.
- 각 파일마다 `EKF → denoise → filtered_residual PSD` 변환을 수행합니다.
- `processed_data/normal_psd_filtered.npy`, `processed_data/isc_psd_filtered.npy`를 저장합니다.

이 파일은 PSD 입력 데이터를 미리 저장하고 싶을 때 사용합니다.
단, 현재 최종 평가 파일인 `eval_1.2cc.py`는 내부에서 PSD를 직접 만들기 때문에 이 단계는 필수는 아닙니다.

### 5. Autoencoder 최종 평가

```powershell
python scripts\eval_1.2cc.py
```

역할:
- `cc_best_ekf_params.csv` 또는 `cc_qr_grid_search.csv`에서 최적 Q/R을 읽습니다.
- Normal 1.2CC DST 데이터는 학습/정상 테스트로 나눕니다.
- ISC 10ohm, 100ohm, 1000ohm 데이터를 테스트합니다.
- 각 데이터는 `EKF → denoise → filtered_residual PSD` 순서로 Autoencoder 입력이 됩니다.
- 정상 PSD로 Autoencoder를 학습하고, 재구성 오차 MSE로 ISC 이상 여부를 판단합니다.
- `results/inference_1.2cc_denoised.png`와 `results/autoencoder_1.2cc_denoised_metrics.csv`를 저장합니다.

핵심 수정점

기존 문제는 Autoencoder가 raw `residual`의 PSD를 바로 쓰거나, `cc_ekf_design.py`에서 찾은 최적 Q/R과 연결되지 않을 수 있다는 점이었습니다.
정리본에서는 `visualize_pipeline.py`, `prepare_dataset.py`, `eval_1.2cc.py`가 모두 최적 Q/R을 읽고, `filtered_residual` 기준 PSD를 사용하도록 맞췄습니다.
