
# ==================================================
# ① 필요한 라이브러리와 공통 함수 불러오기
#    → shutil : 이전 Heatmap 결과 폴더 초기화
#    → Path   : 이미지 및 파일 경로 관리
#    → OpenCV : Heatmap / Overlay / Mask 저장
#    → 공통 함수 : ResNet18 특징 추출 및 시각화
# ==================================================
import shutil
from pathlib import Path

import cv2

from day07_anomaly_core import (
    anomaly_map,
    create_backbone,
    extract_features,
    load_reference,
    make_heatmap_overlay,
)


# ==================================================
# ② 입력 및 출력 경로 설정
#    → 52번에서 만든 Normal Reference 사용
#    → 정상 / 이상 Test 이미지 사용
#    → 실제 결함 위치인 Ground Truth Mask 사용
#    → 결과는 outputs/day07/heatmap에 저장
# ==================================================

# 정상 기준 Vector / Map이 저장된 파일
REFERENCE_PATH = Path(
    "artifacts/day07/normal_reference.npz"
)

# 정상 Test 이미지 폴더
TEST_NORMAL_DIR = Path(
    "samples/day07/test/normal"
)

# 이상 Test 이미지 폴더
TEST_ANOMALY_DIR = Path(
    "samples/day07/test/anomaly"
)

# 실제 결함 위치가 표시된 Mask 폴더
GROUND_TRUTH_DIR = Path(
    "samples/day07/ground_truth"
)

# Heatmap과 Overlay를 저장할 폴더
OUTPUT_DIR = Path(
    "outputs/day07/heatmap"
)


# ==================================================
# ③ 필요한 파일과 폴더 확인
#    → Normal Reference가 있는지 확인
#    → 정상 / 이상 Test 폴더가 있는지 확인
#    → Ground Truth Mask 폴더가 있는지 확인
#    → 필요한 경로가 없으면 오류 발생
# ==================================================
required_paths = [
    REFERENCE_PATH,
    TEST_NORMAL_DIR,
    TEST_ANOMALY_DIR,
    GROUND_TRUTH_DIR,
]

for path in required_paths:

    if not path.exists():
        raise FileNotFoundError(
            f"필요한 파일 또는 폴더를 확인하세요: {path}"
        )


# ==================================================
# ④ 이전 Heatmap 결과 폴더 초기화
#    → 이전 실행 결과를 삭제
#    → 새로운 Heatmap 저장 폴더 생성
#
#    주의!
#    outputs/day07/heatmap 폴더만 초기화합니다.
#    원본 Test 데이터는 삭제하지 않습니다.
# ==================================================

# 기존 Heatmap 결과가 있으면 삭제
if OUTPUT_DIR.exists():
    shutil.rmtree(
        OUTPUT_DIR
    )

# 결과 저장 폴더 다시 생성
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# ⑤ Normal Reference Map 불러오기
#    → 52번 코드에서 저장한 정상 기준 사용
#    → 이번에는 위치별 비교용 reference_map 사용
#    → reference_vector는 사용하지 않음
# ==================================================
_, reference_map = load_reference(
    REFERENCE_PATH
)


# ==================================================
# ⑥ Feature 추출용 ResNet18 준비
#    → 공통 함수 create_backbone() 사용
#    → 사전학습된 ResNet18의 layer3까지 활용
#    → 모델을 새로 학습하지 않고 특징만 추출
# ==================================================
model = create_backbone()


# ==================================================
# ⑦ Heatmap을 만들 대표 Test 이미지 선택
#    → 정상 이미지 2장
#    → 큰 파손 이미지 2장
#    → 작은 파손 이미지 2장
#    → 오염 이미지 2장
#
#    전체 8장의 대표 이미지를 선택합니다.
# ==================================================

# 선택한 이미지 정보를 저장할 리스트
samples = []


# ----------------------------------------------
# ⑦-1. 정상 Test 이미지 목록 가져오기
# ----------------------------------------------
normal_files = sorted(
    TEST_NORMAL_DIR.glob(
        "*.png"
    )
)


# ----------------------------------------------
# ⑦-2. 정상 이미지 앞에서 2장 선택
#       → defect_type = good
# ----------------------------------------------
for path in normal_files[:2]:

    samples.append(
        (
            "good",
            path,
        )
    )


# ----------------------------------------------
# ⑦-3. 이상 이미지 종류별로 2장씩 선택
#       → broken_large
#       → broken_small
#       → contamination
# ----------------------------------------------
for defect_name in [
    "broken_large",
    "broken_small",
    "contamination",
]:

    # 현재 결함 종류의 이미지 목록
    defect_files = sorted(
        (
            TEST_ANOMALY_DIR
            / defect_name
        ).glob("*.png")
    )

    # 앞에서 2장씩 선택
    for path in defect_files[:2]:

        samples.append(
            (
                defect_name,
                path,
            )
        )


# ----------------------------------------------
# ⑦-4. 선택한 이미지가 없으면 오류 발생
# ----------------------------------------------
if not samples:
    raise RuntimeError(
        "Heatmap을 만들 Test 이미지가 없습니다."
    )


# ==================================================
# ⑧ 이미지별 Heatmap 및 Overlay 생성
#    → Test 이미지에서 Feature Map 추출
#    → 정상 기준 Map과 위치별 차이 계산
#    → 차이를 Heatmap으로 표현
#    → 원본 이미지와 Heatmap을 합성
# ==================================================
for defect_type, image_path in samples:

    # ----------------------------------------------
    # ⑧-1. 검사 이미지의 Feature Map 추출
    #       → extract_features() 사용
    #       → 이미지 전체 Vector는 이번에 사용 안 함
    # ----------------------------------------------
    _, spatial_map = extract_features(
        model,
        image_path,
    )

    # ----------------------------------------------
    # ⑧-2. 정상 기준과 위치별 차이 계산
    #       → anomaly_map() 사용
    #
    #       차이가 작은 위치 : Score가 낮음
    #       차이가 큰 위치   : Score가 높음
    # ----------------------------------------------
    score_map = anomaly_map(
        spatial_map,
        reference_map,
    )

    # ----------------------------------------------
    # ⑧-3. Heatmap과 Overlay 생성
    #       → make_heatmap_overlay() 사용
    #
    #       heatmap_color : 위치별 차이를 색으로 표시
    #       overlay       : 원본 + Heatmap 합성
    # ----------------------------------------------
    (
        _,
        heatmap_color,
        overlay,
    ) = make_heatmap_overlay(
        image_path,
        score_map,
    )

    # ----------------------------------------------
    # ⑧-4. 저장할 파일 이름 만들기
    #
    #       예: broken_small_000
    # ----------------------------------------------
    stem = (
        f"{defect_type}_"
        f"{image_path.stem}"
    )

    # ----------------------------------------------
    # ⑧-5. Heatmap 이미지 저장
    #
    #       예: broken_small_000_heatmap.jpg
    # ----------------------------------------------
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{stem}_heatmap.jpg"
        ),
        heatmap_color,
    )

    # ----------------------------------------------
    # ⑧-6. 원본 + Heatmap Overlay 저장
    #
    #       예: broken_small_000_overlay.jpg
    # ----------------------------------------------
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{stem}_overlay.jpg"
        ),
        overlay,
    )


    # ==================================================
    # ⑨ 이상 이미지의 Ground Truth Mask 복사
    #    → 실제 결함 위치가 표시된 정답 이미지
    #    → Heatmap의 강조 위치와 비교하기 위해 저장
    #
    #    정상 이미지(good)는 Mask를 저장하지 않음
    # ==================================================
    if defect_type != "good":

        # ------------------------------------------
        # ⑨-1. 해당 이미지의 Ground Truth 경로
        #
        #       예: broken_small/000_mask.png
        # ------------------------------------------
        mask_path = (
            GROUND_TRUTH_DIR
            / defect_type
            / f"{image_path.stem}_mask.png"
        )

        # ------------------------------------------
        # ⑨-2. Mask 파일이 있는지 확인
        # ------------------------------------------
        if mask_path.exists():

            # --------------------------------------
            # ⑨-3. Ground Truth Mask 읽기
            #       → 흑백 이미지로 읽음
            # --------------------------------------
            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            # --------------------------------------
            # ⑨-4. Ground Truth Mask 저장
            #
            #       예: broken_small_000_gt_mask.png
            # --------------------------------------
            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / f"{stem}_gt_mask.png"
                ),
                mask,
            )

    # ----------------------------------------------
    # ⑨-5. 현재 이미지 저장 완료 메시지 출력
    # ----------------------------------------------
    print(
        "Saved:",
        stem,
    )


# ==================================================
# ⑩ Heatmap 생성 결과 확인
#    → 처리한 대표 이미지 개수 출력
#    → 저장된 결과 폴더 확인
# ==================================================
print("=" * 70)
print("DAY07 HEATMAP COMPLETE")
print("=" * 70)

# ----------------------------------------------
# ⑩-1. 시각화한 이미지 개수 출력
# ----------------------------------------------
print(
    "Samples:",
    len(samples),
)

# ----------------------------------------------
# ⑩-2. 결과 저장 경로 출력
# ----------------------------------------------
print(
    "Saved  :",
    OUTPUT_DIR,
)
