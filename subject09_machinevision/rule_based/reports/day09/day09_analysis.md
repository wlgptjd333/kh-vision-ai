# Day09 실습 결과 분석

## 1. Version / Lineage
- Git Commit: `reports/day09/version_manifest.json`의 `git_commit` 참고
- Day08 Common Query Fingerprint: `67ff120e5ad336a8…`
- 주요 Config / Artifact Hash: Manifest에 파일별로 기록

## 2. Day08 재현성 점검
- 동일 Common Query 확인: True (20장)
- One-shot Accuracy / FP / FN: 0.85 / 1 / 2
- Few-shot Accuracy / FP / FN: 0.90 / 0 / 2
- 저장된 Metric 일치 여부: 두 방법 모두 일치

## 3. Day05 Experiment
- Baseline 설정값 / Accuracy / FP / FN: inlier_ratio_min 0.8664 / 0.9167 / 1 / 0
- Candidate 설정값 / Accuracy / FP / FN: inlier_ratio_min 0.8108 / 0.8333 / 1 / 1
- 실제로 바뀐 변수 하나: inlier_ratio_min
- 결과 해석: 기준을 낮춰도 FP는 줄지 않았고 불량 1장을 놓쳤다. Calibration 비교이므로 이 값으로 Rule을 바꾸지 않는다.

## 4. Day07 Drift
- 고정 조건: Normal Reference, Threshold 0.067458, 정상 Test 20장
- 변경한 Brightness: 밝기 계수 0.65 (평균 137.69 → 89.26)
- Baseline과 Lighting의 Mean Score / FP Rate: 0.0516 / 0.10 → 0.1129 / 1.00
- 변화 원인: 제품은 같은데 조명만 바뀌어도 Feature가 정상 기준에서 멀어진다. 프로그램 오류가 아니라 입력 데이터의 변화(Drift)다.

## 5. Hybrid 성능 비교
- 동일 Query 20장 확인: 확인함
- Policy Version: day09-hybrid-v2.0
- One-shot Accuracy / FP / FN: 0.85 / 1 / 2
- Few-shot Accuracy / FP / FN: 0.90 / 0 / 2
- Hybrid OR Accuracy / FP / FN: 0.90 / 1 / 1
- Hybrid AND Accuracy / FP / FN: 0.85 / 0 / 3
- 결합이 개선한 지표: OR은 Few-shot보다 FN이 1 줄었다. AND는 One-shot보다 FP가 1 줄었다.
- 결합이 악화한 지표: OR은 Few-shot보다 FP가 1 늘었다. AND는 FN이 3으로 늘었다.
- 실제 현장 적용 전 추가로 검증할 내용: 이 20장은 비교용이므로 정책을 고르면 별도 최종 Test가 필요하다. 놓치면 안 되는 검사면 OR 쪽을 우선 검토한다.

## 6. Mini Challenge
- 진행하지 않음 (`src/day09_mini_drift.py`는 교재 틀만 있음)

## 7. 오늘의 결론
- 재현 가능한 검사 실험에 필요한 기록: 데이터 Fingerprint, Config, Artifact Hash, 결과 파일, Git Commit
- Hybrid를 사용하면 좋은 상황: 두 방법이 서로 다른 샘플을 틀리고, FP와 FN 중 무엇을 줄일지 정해져 있을 때
- 사용하지 않아야 할 상황: 기술을 많이 붙이는 것 자체가 목적일 때, 별도 Test 없이 결과만 보고 정책을 고를 때
