from pathlib import Path

import cv2
import numpy as np


reference_path = Path(
    "samples/day04/reference/reference_label.jpg"
)

challenge_dir = Path(
    "samples/day04/challenge"
)

output_dir = Path(
    "outputs/day04/challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


challenge_files = [
    challenge_dir / "shift.jpg",
    challenge_dir / "rotate.jpg",
    challenge_dir / "distance.jpg",
    challenge_dir / "tilt_glare.jpg",
]


# --------------------------------------------------
# 1. 사용할 방법을 선택합니다.
#
# TEMPLATE
# ORB
# SIFT
# SIFT_HOMOGRAPHY
# --------------------------------------------------

# 실제 재촬영 이미지는 위치·회전·거리·기울기가 함께 달라질 수 있다.
# 4일차 비교에서 rot15 / scale90까지 위치를 찾은 방법은
# SIFT + Homography였으므로 이 방법을 선택한다.
selected_method = "SIFT_HOMOGRAPHY"


reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        reference_path
    )


# --------------------------------------------------
# 30_homography.py에서 가져온 준비 코드
#
# Reference의 SIFT 특징점과
# 로고 네 모서리 좌표를 한 번만 만든다.
# --------------------------------------------------

reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY,
)

sift = cv2.SIFT_create()

reference_keypoints, reference_descriptors = (
    sift.detectAndCompute(
        reference_gray,
        None,
    )
)

matcher = cv2.BFMatcher(
    cv2.NORM_L2,
    crossCheck=False,
)

reference_h, reference_w = (
    reference_gray.shape[:2]
)

reference_corners = np.float32([
    [0, 0],
    [reference_w - 1, 0],
    [reference_w - 1, reference_h - 1],
    [0, reference_h - 1],
]).reshape(
    -1,
    1,
    2,
)


# --------------------------------------------------
# 2. Challenge 이미지를 하나씩 읽습니다.
# --------------------------------------------------

for target_path in challenge_files:

    target = cv2.imread(
        str(target_path)
    )

    if target is None:
        print(
            "읽기 실패:",
            target_path
        )
        continue


    # --------------------------------------------------
    # 3. SIFT Good Match (Lowe ratio 0.75)
    # --------------------------------------------------

    target_gray = cv2.cvtColor(
        target,
        cv2.COLOR_BGR2GRAY,
    )

    target_keypoints, target_descriptors = (
        sift.detectAndCompute(
            target_gray,
            None,
        )
    )

    good_matches = []

    if target_descriptors is not None:
        for candidate in matcher.knnMatch(
            reference_descriptors,
            target_descriptors,
            k=2,
        ):
            if len(candidate) < 2:
                continue

            m, n = candidate

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)


    # --------------------------------------------------
    # 4. RANSAC + Homography로 로고 위치를 찾는다.
    # --------------------------------------------------

    inlier_count = 0
    inlier_ratio = 0.0
    status = "NOT_FOUND"
    preview = target.copy()

    if len(good_matches) >= 4:
        reference_points = np.float32([
            reference_keypoints[match.queryIdx].pt
            for match in good_matches
        ]).reshape(-1, 1, 2)

        target_points = np.float32([
            target_keypoints[match.trainIdx].pt
            for match in good_matches
        ]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            reference_points,
            target_points,
            cv2.RANSAC,
            5.0,
        )

        if (
            homography is not None
            and inlier_mask is not None
        ):
            inlier_count = int(
                inlier_mask.ravel().sum()
            )
            inlier_ratio = (
                inlier_count
                / len(good_matches)
            )

            projected_corners = cv2.perspectiveTransform(
                reference_corners,
                homography,
            )

            cv2.polylines(
                preview,
                [np.int32(projected_corners)],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            status = "FOUND"

    cv2.putText(
        preview,
        (
            f"good={len(good_matches)} "
            f"inlier={inlier_count} "
            f"ratio={inlier_ratio:.2f}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(
            output_dir
            / f"{target_path.stem}_{selected_method.lower()}.jpg"
        ),
        preview,
    )


    print(
        target_path.name,
        "method=",
        selected_method,
        f"good={len(good_matches)}",
        f"inlier={inlier_count}",
        f"ratio={inlier_ratio:.3f}",
        f"status={status}",
    )
