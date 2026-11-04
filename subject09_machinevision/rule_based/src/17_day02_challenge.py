from pathlib import Path

import cv2
import numpy as np


image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# TODO
#
# 1. 이미지의 문제를 판단합니다.
# 2. 필요한 전처리를 선택합니다.
# 3. Parameter를 직접 정합니다.
# 4. result에 최종 결과를 저장합니다.
# --------------------------------------------------

# 1. 문제 판단
#    전체 밝기 평균 124(0~255 중간), 병뚜껑과 배경이 모두 밝은 회색 계열이다.
#    배경(위쪽 모서리 134~147)이 병뚜껑 몸통(131)보다 오히려 밝아서
#    밝기만으로는 병뚜껑과 배경을 나누기 어렵다.
#
# 2~3. 전처리 후보와 Parameter 비교
#    Brightness(beta), Gamma, CLAHE를 2~3개 값으로 바꿔 Otsu 결과를 비교한다.
#    결과는 outputs/day02/challenge/trials/ 에 저장한다.

def apply_gamma(source, gamma):
    table = [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    return cv2.LUT(source, np.array(table).astype("uint8"))


def apply_clahe(source, clip_limit):
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


trials = {
    "beta_40": cv2.convertScaleAbs(image, alpha=1.0, beta=40),
    "gamma_1.5": apply_gamma(image, 1.5),
    "gamma_2.0": apply_gamma(image, 2.0),
    "clahe_1.0": apply_clahe(image, 1.0),
    "clahe_2.0": apply_clahe(image, 2.0),
    "clahe_3.0": apply_clahe(image, 3.0),
}

trial_dir = output_dir / "trials"
trial_dir.mkdir(parents=True, exist_ok=True)

for name, trial in trials.items():
    trial_gray = trial if len(trial.shape) == 2 else cv2.cvtColor(trial, cv2.COLOR_BGR2GRAY)
    otsu_value, trial_otsu = cv2.threshold(
        trial_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imwrite(str(trial_dir / f"{name}.jpg"), trial)
    cv2.imwrite(str(trial_dir / f"{name}_otsu.jpg"), trial_otsu)
    print(f"시도 {name:10s} Otsu={otsu_value:5.1f} 흰색 비율={(trial_otsu > 0).mean():.1%}")

# 4. 최종 선택: 전처리 미사용 (3일차 추가 확인)
#    - beta: 전체를 같은 양만큼 밝히므로 Otsu 값만 111→151로 바뀌고 결과 Binary는 원본과 동일
#    - gamma: 조각 수는 줄지만 배경도 함께 흰색으로 남아 분리가 좋아지지 않음
#    - CLAHE 2.0 이상: 병뚜껑 표면 Texture가 강조되어 병뚜껑 내부가 흑백 점으로 부서짐
#    → 어떤 전처리도 병뚜껑-배경 분리를 개선하지 못함. 원인은 화질이 아니라 촬영조건(밝은 배경)
#    → 원본을 그대로 쓰고, 항상 분리되는 어두운 경계선(뚜껑-링 틈, 병목)은 3일차 Contour에서 확인
result = image.copy()


# --------------------------------------------------
# 전처리 결과 저장
# --------------------------------------------------

result_path = (
    output_dir
    / "challenge_result.jpg"
)

cv2.imwrite(
    str(result_path),
    result,
)


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

def to_gray(source):

    if len(source.shape) == 2:
        return source

    return cv2.cvtColor(
        source,
        cv2.COLOR_BGR2GRAY,
    )


original_gray = to_gray(
    image
)

result_gray = to_gray(
    result
)


# --------------------------------------------------
# 전처리 전 Otsu
#
# 이번 Challenge에서는
# 밝은 병뚜껑을 흰색 객체로 확인합니다.
# --------------------------------------------------

_, original_otsu = cv2.threshold(
    original_gray,
    0,
    255,
    cv2.THRESH_BINARY
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 전처리 후 Otsu
# --------------------------------------------------

_, result_otsu = cv2.threshold(
    result_gray,
    0,
    255,
    cv2.THRESH_BINARY
    + cv2.THRESH_OTSU,
)


cv2.imwrite(
    str(
        output_dir
        / "challenge_original_otsu.jpg"
    ),
    original_otsu,
)

cv2.imwrite(
    str(
        output_dir
        / "challenge_result_otsu.jpg"
    ),
    result_otsu,
)


print("입력 :", image_path)
print("결과 :", result_path)
print("Challenge 완료")
