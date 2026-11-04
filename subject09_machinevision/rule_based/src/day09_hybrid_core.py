VALID_LABELS = {"Normal", "Anomaly"}


def hybrid_judge(one_predicted, few_predicted, policy):
    if one_predicted not in VALID_LABELS or few_predicted not in VALID_LABELS:
        raise ValueError("예측값은 Normal 또는 Anomaly여야 합니다.")
    if policy not in {"OR", "AND"}:
        raise ValueError(f"지원하지 않는 정책: {policy}")

    one_ng = one_predicted == "Anomaly"
    few_ng = few_predicted == "Anomaly"

    is_anomaly = (
        (one_ng or few_ng) if policy == "OR"
        else (one_ng and few_ng)
    )
    votes = [
        method
        for method, flagged in [
            ("ONE_SHOT", one_ng),
            ("FEW_SHOT", few_ng),
        ]
        if flagged
    ]
    return {
        "predicted": "Anomaly" if is_anomaly else "Normal",
        "reason": "|".join(votes) if is_anomaly else "NORMAL",
    }


def calculate_metrics(actual_values, predicted_values):
    actual = list(actual_values)
    predicted = list(predicted_values)
    if len(actual) != len(predicted) or not actual:
        raise ValueError("정답과 예측 수가 같고 1개 이상이어야 합니다.")
    if not set(actual + predicted).issubset(VALID_LABELS):
        raise ValueError("정답과 예측에는 Normal/Anomaly만 사용합니다.")

    tp = sum(a == "Anomaly" and p == "Anomaly" for a, p in zip(actual, predicted))
    tn = sum(a == "Normal" and p == "Normal" for a, p in zip(actual, predicted))
    fp = sum(a == "Normal" and p == "Anomaly" for a, p in zip(actual, predicted))
    fn = sum(a == "Anomaly" and p == "Normal" for a, p in zip(actual, predicted))
    total = len(actual)
    return {
        "samples": total,
        "accuracy": round((tp + tn) / total, 4),
        "anomaly_precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "anomaly_recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


if __name__ == "__main__":
    assert hybrid_judge("Anomaly", "Normal", "OR")["predicted"] == "Anomaly"
    assert hybrid_judge("Anomaly", "Normal", "AND")["predicted"] == "Normal"
    assert hybrid_judge("Normal", "Normal", "OR")["predicted"] == "Normal"
    print("Hybrid Core Self Check: PASS")
