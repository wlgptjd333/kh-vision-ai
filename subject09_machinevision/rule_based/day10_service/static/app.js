// Day10 UI 개선: 기존 FastAPI API 구조를 바꾸지 않습니다.

// 사용 API: /api/health, /api/samples, /api/sample-image/{name},

//           /api/inspect, /api/stats, /api/history

  

const $ = (id) => document.getElementById(id);

const sampleSelect = $("sampleSelect");

const inspectButton = $("inspectButton");

const previewImage = $("previewImage");

const finalResult = $("finalResult");

let inspecting = false;

  

const REASONS = {

  NG_OCR_EMPTY: "OCR이 일련번호를 읽지 못했습니다.",

  NG_FORMAT: "인식한 문자가 정해진 13자리 숫자 형식과 다릅니다.",

  NG_EXPECTED_MISMATCH: "OCR 인식값과 제품의 기준 번호가 일치하지 않습니다.",

};

  

async function getJSON(url, options = {}) {

  const response = await fetch(url, { cache: "no-store", ...options });

  let data = {};

  try { data = await response.json(); } catch { /* 오류 메시지를 아래에서 구성 */ }

  if (!response.ok) {

    throw new Error(data.detail || `요청 실패 (HTTP ${response.status})`);

  }

  return data;

}

  

function showNotice(message) {

  const notice = $("notice");

  notice.textContent = message;

  notice.hidden = false;

}

function clearNotice() {

  $("notice").hidden = true;

  $("notice").textContent = "";

}

function setServerStatus(isOnline) {

  const element = $("serverStatus");

  element.textContent = isOnline ? "● 서버 연결 정상" : "● 서버 연결 확인 필요";

  element.className = `server-status ${isOnline ? "online" : "offline"}`;

}

function statusMessage(code) {

  return REASONS[code] || `추가 확인이 필요한 검사 조건 (${code})`;

}

function humanReason(reasons) {

  return (Array.isArray(reasons) && reasons.length)

    ? reasons.map(statusMessage).join(" ")

    : "모든 검사 조건을 통과했습니다.";

}

function setCheck(id, value) {

  const el = $(id);

  if (value === null || value === undefined) {

    el.textContent = "검사 전";

    el.className = "check-status pending";

  } else {

    el.textContent = value ? "통과" : "실패";

    el.className = `check-status ${value ? "pass" : "fail"}`;

  }

}

function renderWaiting(message = "검사할 제품을 선택하고 버튼을 누르세요.") {

  finalResult.className = "decision waiting";

  $("decisionSymbol").textContent = "…";

  $("decisionTitle").textContent = "검사 대기";

  $("decisionDescription").textContent = message;

  $("ocrText").textContent = "-";

  $("expectedText").textContent = "-";

  setCheck("patternStatus", null);

  setCheck("matchStatus", null);

  $("reasonText").textContent = "검사 완료 후 판정 이유가 표시됩니다.";

  $("reasonCode").textContent = "";

  ["ruleVersion", "serviceVersion", "requestId", "inspectedAt"].forEach((id) => $(id).textContent = "-");

}

function renderError(message) {

  finalResult.className = "decision waiting";

  $("decisionSymbol").textContent = "!";

  $("decisionTitle").textContent = "요청 오류";

  $("decisionDescription").textContent = "서버에서 검사를 완료하지 못했습니다.";

  $("reasonText").textContent = message;

  $("reasonCode").textContent = "검사 결과의 NG 판정과는 다릅니다.";

}

function displaySelectedSample() {

  const fileName = sampleSelect.value;

  renderWaiting(fileName ? "'선택 이미지 검사'를 누르면 OCR 검사를 실행합니다." : undefined);

  $("selectedFileName").textContent = fileName || "선택된 이미지 없음";

  $("previewPlaceholder").hidden = Boolean(fileName);

  previewImage.hidden = true;

  previewImage.removeAttribute("src");

  if (fileName) {

    previewImage.src = `/api/sample-image/${encodeURIComponent(fileName)}`;

    inspectButton.disabled = inspecting;

  } else {

    inspectButton.disabled = true;

  }

}

previewImage.addEventListener("load", () => { previewImage.hidden = false; });

previewImage.addEventListener("error", () => {

  previewImage.hidden = true;

  $("previewPlaceholder").textContent = "이미지 미리보기를 불러올 수 없습니다.";

  $("previewPlaceholder").hidden = false;

});

sampleSelect.addEventListener("change", displaySelectedSample);

  

async function loadSamples() {

  const data = await getJSON("/api/samples");

  const samples = Array.isArray(data.samples) ? data.samples : [];

  sampleSelect.replaceChildren();

  const placeholder = document.createElement("option");

  placeholder.value = "";

  placeholder.textContent = samples.length ? "검사 이미지를 선택하세요" : "사용 가능한 샘플이 없습니다";

  sampleSelect.appendChild(placeholder);

  for (const fileName of samples) {

    const option = document.createElement("option");

    option.value = fileName;

    option.textContent = fileName;

    sampleSelect.appendChild(option);

  }

  // 기존 화면처럼 첫 이미지가 미리 보이지만, 검사하기 전에는 판정하지 않습니다.

  if (samples.length) sampleSelect.value = samples[0];

  displaySelectedSample();

}

  

function renderResult(data) {

  const isOK = data.final_decision === "OK";

  finalResult.className = `decision ${isOK ? "ok" : "ng"}`;

  $("decisionSymbol").textContent = isOK ? "✓" : "✕";

  $("decisionTitle").textContent = isOK ? "OK · 정상" : "NG · 불량";

  $("decisionDescription").textContent = isOK

    ? "13자리 형식과 기준 번호가 모두 일치합니다."

    : "형식 또는 기준 번호 검사에서 불일치가 발견되었습니다.";

  const measurement = data.measurements || {};

  $("ocrText").textContent = measurement.ocr_text || "(인식 결과 없음)";

  $("expectedText").textContent = measurement.expected_text || "-";

  setCheck("patternStatus", measurement.pattern_ok);

  setCheck("matchStatus", measurement.expected_match);

  $("reasonText").textContent = humanReason(data.reasons);

  $("reasonCode").textContent = (data.reasons || []).join(", ") || "PASS";

  $("ruleVersion").textContent = data.rule_version || "-";

  $("serviceVersion").textContent = data.service_version || "-";

  $("requestId").textContent = data.request_id || "-";

  $("inspectedAt").textContent = data.inspected_at || "-";

}

  

async function runInspection() {

  if (inspecting || !sampleSelect.value) return;

  const fileName = sampleSelect.value;

  inspecting = true;

  inspectButton.disabled = true;

  sampleSelect.disabled = true;

  inspectButton.textContent = "검사 중…";

  clearNotice();

  finalResult.className = "decision waiting";

  $("decisionSymbol").textContent = "…";

  $("decisionTitle").textContent = "OCR 분석 중";

  $("decisionDescription").textContent = "선택한 이미지에서 일련번호를 읽고 비교하고 있습니다.";

  try {

    const data = await getJSON("/api/inspect", {

      method: "POST",

      headers: { "Content-Type": "application/json" },

      body: JSON.stringify({ file_name: fileName }),

    });

    renderResult(data);

    const updates = await Promise.allSettled([loadStats(), loadHistory()]);

    if (updates.some((result) => result.status === "rejected")) {

      showNotice("검사는 완료되었지만 통계 또는 이력을 불러오지 못했습니다. 새로고침 후 확인하세요.");

    }

  } catch (error) {

    const message = error instanceof Error ? error.message : String(error);

    renderError(message);

    showNotice(`검사 요청을 처리하지 못했습니다: ${message}`);

  } finally {

    inspecting = false;

    sampleSelect.disabled = false;

    inspectButton.disabled = !sampleSelect.value;

    inspectButton.textContent = "선택 이미지 검사";

  }

}

inspectButton.addEventListener("click", runInspection);

  

async function loadStats() {

  const data = await getJSON("/api/stats");

  const total = Number(data.total) || 0;

  const ok = Number(data.ok) || 0;

  const ng = Number(data.ng) || 0;

  $("totalCount").textContent = total.toLocaleString("ko-KR");

  $("okCount").textContent = ok.toLocaleString("ko-KR");

  $("ngCount").textContent = ng.toLocaleString("ko-KR");

  const okRate = total > 0 ? ok / total * 100 : 0;

  $("okRate").textContent = total ? `${okRate.toFixed(1)}%` : "-";

  $("okRateFill").style.width = `${Math.min(100, Math.max(0, okRate))}%`;

}

  

function addCell(row, text, className = "") {

  const cell = document.createElement("td");

  cell.textContent = text;

  if (className) cell.className = className;

  row.appendChild(cell);

  return cell;

}

async function loadHistory() {

  const data = await getJSON("/api/history?limit=10");

  const items = Array.isArray(data.history) ? [...data.history].reverse() : [];

  const body = $("historyBody");

  body.replaceChildren();

  if (!items.length) {

    const row = document.createElement("tr");

    const cell = addCell(row, "아직 저장된 검사 이력이 없습니다.", "empty-table");

    cell.colSpan = 4;

    body.appendChild(row);

    return;

  }

  for (const item of items) {

    const row = document.createElement("tr");

    addCell(row, String(item.inspected_at || "-").replace("T", " "));

    addCell(row, item.file_name || "-");

    const resultCell = document.createElement("td");

    const pill = document.createElement("span");

    const isOK = item.final_decision === "OK";

    pill.className = `result-pill ${isOK ? "ok" : "ng"}`;

    pill.textContent = item.final_decision || "-";

    resultCell.appendChild(pill);

    row.appendChild(resultCell);

    addCell(row, humanReason(item.reasons));

    body.appendChild(row);

  }

}

  

async function init() {

  renderWaiting();

  const results = await Promise.allSettled([getJSON("/api/health"), loadSamples(), loadStats(), loadHistory()]);

  setServerStatus(results[0].status === "fulfilled" && results[0].value.status === "ok");

  if (results[1].status === "rejected") {

    sampleSelect.replaceChildren();

    const errorOption = document.createElement("option");

    errorOption.textContent = "이미지 목록을 가져오지 못했습니다";

    sampleSelect.appendChild(errorOption);

    inspectButton.disabled = true;

    showNotice("이미지 목록을 불러오지 못했습니다. FastAPI 서버 실행 상태를 확인하세요.");

  }

  if (results[2].status === "rejected" || results[3].status === "rejected") {

    showNotice("통계 또는 검사 이력을 불러오지 못했습니다. 서버 연결을 확인하세요.");

  }

}

init();
