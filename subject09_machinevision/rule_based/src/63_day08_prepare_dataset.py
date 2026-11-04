# ==================================================
# ① 라이브러리 불러오기 및 기본 경로 설정
#    → random : 이미지 선택 순서를 무작위로 섞기
#    → shutil : 원본 이미지를 실습 폴더에 복사
#    → Path   : 파일과 폴더 경로 관리
#
#    SEED = 42
#    → 다시 실행해도 동일한 이미지가 선택되도록 설정
# ==================================================
import random
import shutil
from pathlib import Path


# 7일차에서 보존한 MVTec bottle 원본
RAW_ROOT = Path(
    "samples/day07/original/bottle"
)

# 8일차 실습용 데이터 저장 폴더
DAY08_ROOT = Path(
    "samples/day08"
)

# 데이터 구성 결과를 기록할 보고서
REPORT_PATH = Path(
    "reports/day08_dataset_summary.md"
)

# 이미지 선택 결과를 동일하게 유지하는 값
SEED = 42


# ==================================================
# ② Day08 실습 폴더 초기화
#    → 이전에 생성한 samples/day08 폴더를 삭제
#    → 새 실습 데이터를 만들 준비
#
#    주의!
#    DAY08_ROOT 안의 기존 파일은 모두 삭제됩니다.
#    원본 RAW_ROOT는 삭제하지 않습니다.
# ==================================================
if DAY08_ROOT.exists():

    shutil.rmtree(
        DAY08_ROOT
    )


# ==================================================
# ③ 데이터를 복사하는 공통 함수 작성
#    → 여러 종류의 데이터 복사 작업에서 재사용
#
#    copy_files()
#    → 일반 이미지 복사
#
#    copy_defect_sources()
#    → 불량 이미지와 해당 Ground Truth Mask 복사
# ==================================================


# --------------------------------------------------
# ③-1. 일반 이미지를 복사하는 함수
#
#      files       : 복사할 이미지 목록
#      destination : 저장할 폴더
# --------------------------------------------------
def copy_files(files, destination):

    # 저장할 폴더가 없으면 생성
    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    # 이미지 목록을 한 장씩 복사
    for path in files:

        shutil.copy2(
            path,
            destination / path.name
        )


# --------------------------------------------------
# ③-2. 불량 이미지와 Mask를 함께 복사하는 함수
#
#      defect_type : broken_small / contamination
#
#      images 폴더 : 실제 결함 이미지
#      masks 폴더  : 실제 결함 위치 정답 이미지
# --------------------------------------------------
def copy_defect_sources(
    files,
    defect_type,
    destination_root
):

    # 불량 이미지 저장 폴더
    image_dir = (
        destination_root
        / defect_type
        / "images"
    )

    # Ground Truth Mask 저장 폴더
    mask_dir = (
        destination_root
        / defect_type
        / "masks"
    )

    # 이미지와 Mask 폴더 생성
    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 불량 이미지를 한 장씩 처리
    for image_path in files:

        # 해당 불량 이미지와 짝이 되는 Mask 경로
        #
        # 예:
        # 이미지 000.png
        # Mask   000_mask.png
        mask_path = (
            RAW_ROOT
            / "ground_truth"
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        # Mask 파일이 없으면 오류 발생
        if not mask_path.exists():

            raise FileNotFoundError(
                mask_path
            )

        # 불량 원본 이미지 복사
        shutil.copy2(
            image_path,
            image_dir / image_path.name
        )

        # 실제 결함 위치 Mask 복사
        shutil.copy2(
            mask_path,
            mask_dir / mask_path.name
        )


# ==================================================
# ④ MVTec bottle 원본 이미지 목록 준비
#    → 정상 Train / 정상 Test 이미지 가져오기
#    → 작은 파손 / 오염 이미지 가져오기
#    → 무작위로 섞어 선택 순서를 결정
#
#    같은 SEED를 사용하면 동일한 분할 재현 가능
# ==================================================


# --------------------------------------------------
# ④-1. 원본 이미지 목록 불러오기
# --------------------------------------------------

# 정상 Train 이미지
normal_train = sorted(
    (RAW_ROOT / "train/good").glob("*.png")
)

# 정상 Test 이미지
normal_test = sorted(
    (RAW_ROOT / "test/good").glob("*.png")
)

# 작은 파손 이미지
broken_small = sorted(
    (RAW_ROOT / "test/broken_small").glob("*.png")
)

# 오염 이미지
contamination = sorted(
    (RAW_ROOT / "test/contamination").glob("*.png")
)


# --------------------------------------------------
# ④-2. 종류별 무작위 선택 도구 준비
#
#       각각 다른 Seed를 사용
#       → 이미지 선택 순서는 매번 동일하게 재현
# --------------------------------------------------
rng_normal = random.Random(SEED)

rng_broken = random.Random(
    SEED + 1
)

rng_contamination = random.Random(
    SEED + 2
)


# --------------------------------------------------
# ④-3. 이미지 목록의 순서 섞기
#
#       원본 파일을 수정하지 않고
#       Python 리스트의 순서만 변경
# --------------------------------------------------
rng_normal.shuffle(
    normal_train
)

rng_broken.shuffle(
    broken_small
)

rng_contamination.shuffle(
    contamination
)


# ==================================================
# ⑤ 정상 이미지 역할 분리
#    → One-shot / Few-shot / 불량 합성 / Challenge
#      용도별로 서로 다른 정상 이미지를 선택
#
#    정상 Train 원본에서 총 41장 선택
#    정상 Test 원본에서 Query 10장 선택
# ==================================================


# --------------------------------------------------
# ⑤-1. One-shot 정상 기준 이미지 1장
#       → 새로운 이미지를 비교할 Reference
# --------------------------------------------------
one_reference = normal_train[0:1]


# --------------------------------------------------
# ⑤-2. One-shot Calibration 정상 이미지 20장
#       → Similarity Threshold를 정하는 데 사용
# --------------------------------------------------
one_calibration = normal_train[1:21]


# --------------------------------------------------
# ⑤-3. Few-shot 정상 Support 5장
#       → Normal Prototype 생성에 사용
# --------------------------------------------------
few_normal_support = normal_train[21:26]


# --------------------------------------------------
# ⑤-4. Cut-Paste 합성 대상 정상 이미지 10장
#       → 실제 결함 Patch를 붙일 정상 이미지
# --------------------------------------------------
cutpaste_normal = normal_train[26:36]


# --------------------------------------------------
# ⑤-5. Mini Challenge용 정상 이미지 5장
#       → 새로운 결함 합성 실험에 사용
# --------------------------------------------------
mini_normal = normal_train[36:41]


# --------------------------------------------------
# ⑤-6. Common Query 정상 이미지 10장
#       → One-shot과 Few-shot에서 동일하게 검사
#       → 정상 Train이 아닌 Test 원본에서 선택
# --------------------------------------------------
common_normal_query = normal_test[:10]


# ==================================================
# ⑥ 불량 이미지 역할 분리
#    → broken_small은 Few-shot과 공통 Query,
#      불량 합성에 각각 사용
#
#    → contamination은 Mini Challenge에서 사용
# ==================================================


# --------------------------------------------------
# ⑥-1. Few-shot 불량 Support 5장
#       → Anomaly Prototype 생성에 사용
# --------------------------------------------------
few_anomaly_support = broken_small[0:5]


# --------------------------------------------------
# ⑥-2. Common Query 불량 이미지 10장
#       → One-shot / Few-shot에서 동일하게 검사
# --------------------------------------------------
common_anomaly_query = broken_small[5:15]


# --------------------------------------------------
# ⑥-3. Cut-Paste 불량 원본 이미지 2장
#       → 실제 작은 파손 Patch 추출에 사용
# --------------------------------------------------
cutpaste_defect_source = broken_small[15:17]


# --------------------------------------------------
# ⑥-4. Mini Challenge 오염 이미지 2장
#       → 새로운 결함 유형 합성에 사용
# --------------------------------------------------
mini_defect_source = contamination[:2]


# ==================================================
# ⑦ 데이터 중복 검사
#    → 기준 생성용 이미지와 평가용 이미지 등이
#      서로 겹치지 않는지 확인
#
#    같은 이미지를 여러 역할에 사용하면
#    실험 결과가 실제보다 좋아 보일 수 있음
#
#    이 코드는 아래 Normal Train 역할들과
#    broken_small 역할들 사이의 중복을 검사
# ==================================================


# --------------------------------------------------
# ⑦-1. 정상 Train 데이터의 역할별 목록
# --------------------------------------------------
normal_groups = [
    set(one_reference),
    set(one_calibration),
    set(few_normal_support),
    set(cutpaste_normal),
    set(mini_normal),
]


# --------------------------------------------------
# ⑦-2. 정상 Train 역할 간 이미지 중복 확인
# --------------------------------------------------
for i in range(
    len(normal_groups)
):

    for j in range(
        i + 1,
        len(normal_groups)
    ):

        # 두 역할의 이미지가 겹치면 오류 발생
        if normal_groups[i] & normal_groups[j]:

            raise RuntimeError(
                "Normal Train 역할 간 데이터가 겹칩니다."
            )


# --------------------------------------------------
# ⑦-3. broken_small 역할별 목록
# --------------------------------------------------
broken_groups = [
    set(few_anomaly_support),
    set(common_anomaly_query),
    set(cutpaste_defect_source),
]


# --------------------------------------------------
# ⑦-4. broken_small 역할 간 이미지 중복 확인
# --------------------------------------------------
for i in range(
    len(broken_groups)
):

    for j in range(
        i + 1,
        len(broken_groups)
    ):

        # 같은 불량 이미지가 겹치면 오류 발생
        if broken_groups[i] & broken_groups[j]:

            raise RuntimeError(
                "broken_small 역할 간 데이터가 겹칩니다."
            )


# ==================================================
# ⑧ 역할별 실습 폴더로 이미지 복사
#    → ⑤, ⑥에서 선택한 이미지를 실제 폴더에 저장
#    → 원본 데이터는 변경하지 않음
#
#    각 역할을 나누어 두면
#    이후 64~69번 코드에서 바로 사용할 수 있음
# ==================================================


# --------------------------------------------------
# ⑧-1. One-shot Reference 1장 복사
# --------------------------------------------------
copy_files(
    one_reference,
    DAY08_ROOT / "one_shot/reference",
)


# --------------------------------------------------
# ⑧-2. One-shot Calibration 20장 복사
# --------------------------------------------------
copy_files(
    one_calibration,
    DAY08_ROOT / "one_shot/calibration/normal",
)


# --------------------------------------------------
# ⑧-3. Few-shot 정상 Support 5장 복사
# --------------------------------------------------
copy_files(
    few_normal_support,
    DAY08_ROOT / "few_shot/support/normal",
)


# --------------------------------------------------
# ⑧-4. Few-shot 불량 Support 5장 복사
# --------------------------------------------------
copy_files(
    few_anomaly_support,
    DAY08_ROOT / "few_shot/support/anomaly",
)


# --------------------------------------------------
# ⑧-5. 공통 평가용 정상 Query 10장 복사
# --------------------------------------------------
copy_files(
    common_normal_query,
    DAY08_ROOT / "common_query/normal",
)


# --------------------------------------------------
# ⑧-6. 공통 평가용 불량 Query 10장 복사
# --------------------------------------------------
copy_files(
    common_anomaly_query,
    DAY08_ROOT / "common_query/anomaly/broken_small",
)


# --------------------------------------------------
# ⑧-7. Cut-Paste 합성 대상 정상 이미지 10장
# --------------------------------------------------
copy_files(
    cutpaste_normal,
    DAY08_ROOT / "cutpaste/normal",
)


# --------------------------------------------------
# ⑧-8. 작은 파손 원본 2장과 Mask 함께 복사
#       → Cut-Paste에서 실제 결함 영역 추출
# --------------------------------------------------
copy_defect_sources(
    cutpaste_defect_source,
    "broken_small",
    DAY08_ROOT / "cutpaste/defect_source",
)


# --------------------------------------------------
# ⑧-9. Mini Challenge 정상 이미지 5장 복사
# --------------------------------------------------
copy_files(
    mini_normal,
    DAY08_ROOT / "mini_challenge/normal",
)


# --------------------------------------------------
# ⑧-10. 오염 원본 2장과 Mask 함께 복사
#        → Mini Challenge에서 오염 결함 합성
# --------------------------------------------------
copy_defect_sources(
    mini_defect_source,
    "contamination",
    DAY08_ROOT / "mini_challenge/defect_source",
)


# ==================================================
# ⑨ 데이터 구성 요약 보고서 작성
#    → 각 실습에서 사용할 이미지 개수 기록
#    → 중복 검사 통과 여부 기록
#    → Markdown 파일로 저장
#
#    보고서 위치:
#    reports/day08_dataset_summary.md
# ==================================================


# --------------------------------------------------
# ⑨-1. 데이터 구성 내용을 Markdown으로 정리
# --------------------------------------------------
summary = f"""# Day08 Dataset Summary

Seed: {SEED}

## One-shot

- Reference: {len(one_reference)}
- Normal Calibration: {len(one_calibration)}

## Few-shot Support

- Normal: {len(few_normal_support)}
- Anomaly(broken_small): {len(few_anomaly_support)}

## Common Query

- Normal: {len(common_normal_query)}
- Anomaly(broken_small): {len(common_anomaly_query)}

## Cut-Paste

- Normal Target: {len(cutpaste_normal)}
- Defect Source(broken_small): {len(cutpaste_defect_source)}

## Mini Challenge

- Normal Target: {len(mini_normal)}
- Defect Source(contamination): {len(mini_defect_source)}

## Data Leakage Check

PASS
"""


# --------------------------------------------------
# ⑨-2. reports 폴더가 없으면 생성
# --------------------------------------------------
REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# ⑨-3. Markdown 보고서 저장
# --------------------------------------------------
REPORT_PATH.write_text(
    summary,
    encoding="utf-8"
)


# ==================================================
# ⑩ 실행 결과 출력
#    → 각 실습별 데이터 개수 출력
#    → 중복 검사 결과 출력
#    → Summary 보고서 저장 위치 확인
# ==================================================
print("=" * 70)
print("DAY08 DATASET PREPARE")
print("=" * 70)

print(
    "One-shot Reference        :",
    len(one_reference)
)

print(
    "One-shot Calibration      :",
    len(one_calibration)
)

print(
    "Few-shot Normal Support   :",
    len(few_normal_support)
)

print(
    "Few-shot Anomaly Support  :",
    len(few_anomaly_support)
)

print(
    "Common Query Normal       :",
    len(common_normal_query)
)

print(
    "Common Query Anomaly      :",
    len(common_anomaly_query)
)

print(
    "Cut-Paste Normal          :",
    len(cutpaste_normal)
)

print(
    "Cut-Paste Defect Source   :",
    len(cutpaste_defect_source)
)

print(
    "Mini Defect Source        :",
    len(mini_defect_source)
)

print(
    "Data Leakage Check        : PASS"
)

print(
    "Saved                     :",
    REPORT_PATH
)
