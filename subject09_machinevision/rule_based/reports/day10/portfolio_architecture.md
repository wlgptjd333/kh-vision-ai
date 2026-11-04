# Machine Vision Inspection Service Architecture

```text
Product Image
        ↓
Inspection Core (service_core.py)
        ↓
OCR / Rule (Day06 Frozen Rule)
        ↓
Measurement
        ↓
Final OK / NG
        ↓
Reason
        ↓
Result Dictionary
        ↓
FastAPI (api_main.py)
        ↓
JSON Response
        ↓
Web Dashboard
```

## 핵심 설계

- Day06 Frozen OCR Rule을 다시 수정하지 않고 재사용했다.
- 검사 기능과 HTTP API 코드를 분리했다.
- Final Decision뿐 아니라 Reason과 Version도 함께 반환했다.
- Web Dashboard에서 검사 결과와 History / Stats를 확인하도록 구성했다.
- 향후 Camera와 Edge Device 기반 검사로 Inspection Core를 교체할 수 있는 구조다.
