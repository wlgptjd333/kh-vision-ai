from __future__ import annotations


# ==================================================
# [01] 필요한 라이브러리 가져오기 및 프로젝트 경로 설정
# ==================================================
# argparse: 실행할 때 입력 옵션을 받습니다.
# csv     : Feature 데이터를 CSV로 저장합니다.
# sys     : 프로젝트 내부 모듈을 불러옵니다.
# Path    : 파일 및 폴더 경로를 관리합니다.
# OpenCV  : MP4에서 Frame을 읽습니다.
# NumPy   : 관절좌표 및 Confidence를 계산합니다.
# YOLO    : 영상에서 사람의 관절을 추출합니다.

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# 현재 파일이 scripts 폴더에 있다고 가정합니다.
# 프로젝트 최상위 폴더를 ROOT로 지정합니다.

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ==================================================
# [02] 이전 수업에서 만든 공통 함수 가져오기
# ==================================================
# common.py:
# project.json 설정을 읽고 실행 장치를 결정합니다.
#
# feature_utils.py:
# 관절좌표 정규화 및 관절각도를 계산합니다.
#
# pose_utils.py:
# COCO 17개 관절의 이름을 가져옵니다.

from src.common import (
    load_config,
    resolve_device,
)

from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)

from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# [03] 실행 옵션 설정
# ==================================================
# --source:
# 분석할 MP4 파일의 경로입니다.
#
# --kp-conf:
# 관절 Confidence가 낮은지 판단하는 기준입니다.
# 기본값은 0.50입니다.
#
# --max-frames:
# 처리할 최대 Frame 수입니다.
# 기본값은 150개이며 0이면 전체 Frame을 처리합니다.
#
# 주의:
# kp-conf는 낮은 Confidence 관절 개수를 세는 기준이며,
# 해당 Frame을 자동으로 제외하는 조건은 아닙니다.

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "videos"
            / "feature_demo.mp4"
        ),
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="처리할 최대 Frame 수, 0이면 전체 처리",
    )

    return parser.parse_args()


# ==================================================
# [04] 38-D Frame Feature의 컬럼 이름 생성
# ==================================================
# COCO 17개 관절마다 x, y 좌표를 사용합니다.
#
# 17개 관절 × 2개 좌표 = 34개
#
# 여기에 관절각도 4개를 추가합니다.
#
# 왼쪽 팔꿈치 각도
# 오른쪽 팔꿈치 각도
# 왼쪽 무릎 각도
# 오른쪽 무릎 각도
#
# 최종적으로 38개 Feature 이름을 반환합니다.

def feature_names():
    names = []

    for name in (
        COCO_KEYPOINT_NAMES
    ):
        names.extend(
            [
                f"{name}_x",
                f"{name}_y",
            ]
        )

    names.extend(
        [
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
        ]
    )

    return names


# ==================================================
# [05] 입력 영상 확인 및 project.json 설정 읽기
# ==================================================
# main():
# 실제 Feature 추출 작업을 시작합니다.
#
# parse_args():
# 실행 옵션을 가져옵니다.
#
# source.exists():
# 분석할 MP4 파일이 실제 존재하는지 확인합니다.
#
# load_config():
# configs/project.json 설정을 읽습니다.
#
# resolve_device():
# CPU 또는 GPU 실행 장치를 결정합니다.

def main() -> None:
    args = parse_args()

    source = Path(
        args.source
    )

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    config = load_config()

    device = resolve_device(
        config["device"]
    )


    # ==============================================
    # [06] YOLO Pose 모델 및 MP4 영상 준비
    # ==============================================
    # project.json의 pose_model 값을 사용합니다.
    #
    # OpenCV로 MP4 영상을 엽니다.
    #
    # 영상이 정상적으로 열리지 않으면 오류를 발생시킵니다.
    #
    # FPS:
    # 1초 동안 재생되는 Frame 수입니다.
    #
    # FPS를 읽을 수 없으면 30으로 설정합니다.

    model = YOLO(
        config["pose_model"]
    )

    capture = cv2.VideoCapture(
        str(source)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다: "
            f"{source}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0


    # ==============================================
    # [07] 출력 CSV 경로와 컬럼 구성 준비
    # ==============================================
    # feature_names():
    # 38개 Feature 이름을 가져옵니다.
    #
    # CSV 저장 경로:
    # reports/day03/tables/
    #
    # CSV 기본정보 컬럼 5개:
    # frame_id
    # time_sec
    # valid_pose
    # mean_keypoint_conf
    # low_keypoint_count
    #
    # 여기에 38개 Feature 컬럼을 추가합니다.
    #
    # 따라서 전체 CSV 컬럼 수는 43개입니다.

    names = feature_names()

    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / (
            f"{source.stem}"
            "_frame_features.csv"
        )
    )

    fieldnames = [
        "frame_id",
        "time_sec",
        "valid_pose",
        "mean_keypoint_conf",
        "low_keypoint_count",
    ] + names

    rows = []

    frame_id = 0


    # ==============================================
    # [08] MP4에서 Frame을 순서대로 읽기
    # ==============================================
    # while True:
    # 영상을 Frame 단위로 반복 처리합니다.
    #
    # max_frames를 초과하면 반복을 종료합니다.
    #
    # capture.read():
    # 현재 Frame을 읽습니다.
    #
    # 더 이상 읽을 Frame이 없으면 종료합니다.
    #
    # 정상 Frame이면 frame_id를 1 증가시킵니다.

    while True:
        if (
            args.max_frames > 0
            and frame_id >= args.max_frames
        ):
            break

        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_id += 1


        # ==========================================
        # [09] 현재 Frame에서 YOLO Pose 추론
        # ==========================================
        # model.predict():
        # 현재 Frame에서 사람과 관절을 검출합니다.
        #
        # source:
        # 현재 영상 Frame입니다.
        #
        # imgsz:
        # project.json의 image_size를 사용합니다.
        #
        # conf:
        # project.json의 person_confidence를 사용합니다.
        #
        # device:
        # CPU 또는 GPU 실행 장치입니다.
        #
        # 결과에서 첫 번째 추론 결과를 가져옵니다.

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config[
                "person_confidence"
            ],
            device=device,
            verbose=False,
        )[0]


        # ==========================================
        # [10] 현재 Frame의 기본정보 생성
        # ==========================================
        # frame_id:
        # 현재 Frame 번호입니다.
        #
        # time_sec:
        # 영상 시작부터 현재 Frame까지의 시간입니다.
        #
        # valid_pose:
        # 정상 Pose Feature 생성 여부입니다.
        #
        # mean_keypoint_conf:
        # 17개 관절 Confidence 평균입니다.
        #
        # low_keypoint_count:
        # Confidence가 기준보다 낮은 관절 개수입니다.
        #
        # 처음에는 기본값을 저장합니다.
        # 정상 Pose를 계산하면 값을 갱신합니다.
        #
        # 38개 Feature 값은 먼저 빈 문자열로 초기화합니다.

        base_row = {
            "frame_id": frame_id,
            "time_sec": round(
                (frame_id - 1) / fps,
                4,
            ),
            "valid_pose": 0,
            "mean_keypoint_conf": (
                0.0
            ),
            "low_keypoint_count": 17,
        }

        for name in names:
            base_row[name] = ""


        # ==========================================
        # [11] YOLO Pose 검출 결과 확인
        # ==========================================
        # result.keypoints:
        # YOLO가 검출한 관절 정보입니다.
        #
        # result.keypoints.xy:
        # 관절의 Pixel 좌표입니다.
        #
        # 사람이 검출되었다면
        # 첫 번째 사람의 관절좌표를 가져옵니다.
        #
        # 결과 형태:
        # 17개 관절 × (x, y)

        if (
            result.keypoints
            is not None
            and result.keypoints.xy
            is not None
            and len(
                result.keypoints.xy
            ) > 0
        ):
            keypoints = (
                result.keypoints.xy[0]
                .cpu()
                .numpy()
            )


            # ======================================
            # [12] 관절별 Confidence 가져오기
            # ======================================
            # result.keypoints.conf:
            # 각 관절의 검출 신뢰도입니다.
            #
            # Confidence 정보가 있으면 가져옵니다.
            #
            # 없다면 17개 관절의 Confidence를
            # 모두 1.0으로 설정합니다.

            if (
                result.keypoints.conf
                is not None
            ):
                confidence = (
                    result.keypoints.conf[0]
                    .cpu()
                    .numpy()
                )
            else:
                confidence = (
                    np.ones(
                        17,
                        dtype=np.float32,
                    )
                )


            # ======================================
            # [13] 평균 Confidence와 저신뢰 관절 수 계산
            # ======================================
            # mean_conf:
            # 17개 관절 Confidence의 평균입니다.
            #
            # low_count:
            # --kp-conf 기준보다 낮은 관절 개수입니다.
            #
            # 예:
            # kp-conf = 0.50
            #
            # Confidence 0.50 미만인 관절이
            # 3개라면 low_count = 3입니다.
            #
            # 이 값은 QA 참고 정보로 사용합니다.

            mean_conf = float(
                np.mean(
                    confidence
                )
            )

            low_count = int(
                np.sum(
                    confidence
                    < args.kp_conf
                )
            )


            # ======================================
            # [14] 관절좌표 정규화 및 관절각도 계산
            # ======================================
            # normalize_by_shoulder_width():
            # 관절좌표를 신체 기준으로 정규화합니다.
            #
            # selected_joint_angles():
            # 팔꿈치와 무릎의 관절각도를 계산합니다.
            #
            # 정규화에 실패하면 ValueError가 발생합니다.
            #
            # 이 경우 현재 Frame의 Feature는
            # 기본값 상태로 유지됩니다.

            try:
                normalized, _, _ = (
                    normalize_by_shoulder_width(
                        keypoints
                    )
                )

                angles = (
                    selected_joint_angles(
                        keypoints
                    )
                )


                # ==================================
                # [15] 정규화 좌표와 관절각도를 38-D로 결합
                # ==================================
                # 정규화 관절좌표:
                # 17개 × 2 = 34개
                #
                # 관절각도:
                # 팔꿈치 2개 + 무릎 2개 = 4개
                #
                # 총 38개 Feature를 만듭니다.

                values = []

                for point in normalized:
                    values.extend(
                        [
                            float(
                                point[0]
                            ),
                            float(
                                point[1]
                            ),
                        ]
                    )

                values.extend(
                    [
                        angles[
                            "left_elbow_angle"
                        ],
                        angles[
                            "right_elbow_angle"
                        ],
                        angles[
                            "left_knee_angle"
                        ],
                        angles[
                            "right_knee_angle"
                        ],
                    ]
                )


                # ==================================
                # [16] 정상 Pose 정보와 38-D Feature 기록
                # ==================================
                # 정규화와 각도 계산에 성공하면
                # valid_pose를 1로 변경합니다.
                #
                # 평균 Confidence를 기록합니다.
                #
                # 낮은 Confidence 관절 개수를 기록합니다.
                #
                # 38개 Feature 값을 CSV 행에 저장합니다.
                #
                # 숫자는 소수점 6자리로 반올림합니다.

                base_row[
                    "valid_pose"
                ] = 1

                base_row[
                    "mean_keypoint_conf"
                ] = round(
                    mean_conf,
                    4,
                )

                base_row[
                    "low_keypoint_count"
                ] = low_count

                for name, value in zip(
                    names,
                    values,
                ):
                    base_row[name] = (
                        round(
                            float(value),
                            6,
                        )
                    )

            except ValueError:
                pass


        # ==========================================
        # [17] Frame 결과 누적 및 진행 상황 출력
        # ==========================================
        # 현재 Frame의 결과를 rows에 추가합니다.
        #
        # 30 Frame마다 진행 상황을 출력합니다.
        #
        # 예:
        # Processed: 30
        # Processed: 60
        # Processed: 90

        rows.append(
            base_row
        )

        if frame_id % 30 == 0:
            print(
                f"Processed: "
                f"{frame_id}"
            )


    # ==================================================
    # [18] CSV 저장 및 최종 처리 결과 출력
    # ==================================================
    # capture.release():
    # 영상 파일 사용을 종료합니다.
    #
    # csv.DictWriter:
    # 각 Frame의 결과를 CSV로 저장합니다.
    #
    # CSV에는 기본정보 5개와
    # Feature 38개가 포함됩니다.
    #
    # 마지막에 다음 정보를 출력합니다.
    #
    # 전체 처리 Frame 수
    # 정상 Pose Frame 수
    # Frame당 Feature 개수
    # 저장된 CSV 경로

    capture.release()

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    valid_count = sum(
        int(
            row["valid_pose"]
        )
        for row in rows
    )

    print()
    print(
        f"Total frames: "
        f"{frame_id}"
    )

    print(
        f"Valid pose frames: "
        f"{valid_count}"
    )

    print(
        f"Feature count per frame: "
        f"{len(names)}"
    )

    print(
        f"Saved: {output_path}"
    )


# ==================================================
# Python 파일 직접 실행 시 main() 호출
# ==================================================
# 이 파일을 직접 실행할 때만 main()을 시작합니다.

if __name__ == "__main__":
    main()
