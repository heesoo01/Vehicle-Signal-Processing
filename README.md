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

git clone ...
cd Vehicle-Signal-Processing

python -m venv venv
.\venv\Scripts\activate.ps1

python -m pip install -r requirements.txt

python scripts/main_usage_example.py

git pull   # 최신 업데이트 받기
git push   # 내 수정 업로드

