from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session
from .db import get_db
from .models import TextItem
from .services import analyze_text
from .tasks import analyze_textitem_task

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def render(request, name, **context):
    return templates.TemplateResponse(request=request, name="sentiment/"+name, context=context)

def checked(text):
    cleaned = (text or "").strip()
    return cleaned if 1 <= len(cleaned) <= 500 else None

def enqueue_or_mark_failed(item: TextItem, db: Session):
    try:
        analyze_textitem_task.delay(item.id, item.revision)
    except Exception:
        item.status = "FAILED"
        item.is_analyzing = False
        item.error_message = "BROKER_UNAVAILABLE"
        db.commit()

@router.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/items/", status_code=303)

@router.get("/predict/", response_class=HTMLResponse, name="predict_page")
def predict_page(request: Request):
    return render(request, "predict.html", text="", result=None, error=None)

@router.post("/predict/", response_class=HTMLResponse, name="predict")
def predict(request: Request, text: str = Form(default="")):
    value = checked(text)
    if value is None:
        return templates.TemplateResponse(request=request, name="sentiment/predict.html",
            context={"text": text, "result": None, "error": "문장은 1~500자로 입력하세요."}, status_code=400)
    # HTTP 요청 안에서 모델이 끝날 때까지 기다림: 동기 방식
    result = analyze_text(value, use_cache=False)
    return render(request, "predict.html", text=value, result={"input":value, **result}, error=None)

@router.get("/items/", response_class=HTMLResponse, name="item_list")
def item_list(request: Request, db: Session = Depends(get_db)):
    items = db.scalars(select(TextItem).order_by(TextItem.id.desc())).all()
    return render(request, "textitem_list.html", items=items)

@router.get("/new/", response_class=HTMLResponse, name="item_create_page")
def item_create_page(request: Request):
    return render(request, "textitem_form.html", item=None, text="", error=None)

@router.post("/new/", name="item_create")
def item_create(request: Request, text: str = Form(default=""), db: Session = Depends(get_db)):
    value = checked(text)
    if value is None:
        return templates.TemplateResponse(request=request, name="sentiment/textitem_form.html",
            context={"item":None, "text":text, "error":"문장은 1~500자로 입력하세요."}, status_code=400)
    # 먼저 저장 → Celery에는 ID만 전달 → 즉시 redirect
    item = TextItem(text=value, label="", score=None, status="PENDING", is_analyzing=True, revision=1)
    db.add(item)
    db.commit()
    db.refresh(item)
    enqueue_or_mark_failed(item, db)
    return RedirectResponse(request.url_for("item_detail", item_id=item.id), status_code=303)

@router.get("/items/{item_id}/", response_class=HTMLResponse, name="item_detail")
def item_detail(request: Request, item_id: int, db: Session = Depends(get_db)):
    item = db.get(TextItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="해당 글이 없습니다.")
    return render(request, "textitem_detail.html", item=item)

@router.get("/items/{item_id}/status", name="item_status")
def item_status(item_id: int, db: Session = Depends(get_db)):
    item = db.get(TextItem, item_id)
    if item is None:
        raise HTTPException(status_code=404)
    return {"id": item.id, "status": item.status, "is_analyzing":item.is_analyzing,
            "label": item.label, "score":item.score}

@router.get("/items/{item_id}/edit/", response_class=HTMLResponse, name="item_update_page")
def item_update_page(request: Request, item_id: int, db: Session = Depends(get_db)):
    item = db.get(TextItem, item_id)
    if item is None:
        raise HTTPException(status_code=404)
    return render(request, "textitem_form.html", item=item, text=item.text, error=None)

@router.post("/items/{item_id}/edit/", name="item_update")
def item_update(request: Request, item_id: int, text: str = Form(default=""), db: Session = Depends(get_db)):
    item = db.get(TextItem, item_id)
    if item is None:
        raise HTTPException(status_code=404)
    value = checked(text)
    if value is None:
        return templates.TemplateResponse(request=request, name="sentiment/textitem_form.html",
            context={"item":item,"text":text,"error":"문장은 1~500자로 입력하세요."}, status_code=400)
    item.text = value
    item.label = ""
    item.score = None
    item.status = "PENDING"
    item.is_analyzing = True
    item.error_message = ""
    item.revision += 1  # 이전 Celery 작업이 수정 후의 결과를 덮어쓰지 못하도록 구분
    db.commit()
    db.refresh(item)
    enqueue_or_mark_failed(item, db)
    return RedirectResponse(request.url_for("item_detail", item_id=item_id), status_code=303)

@router.get("/items/{item_id}/delete/", response_class=HTMLResponse, name="item_delete_page")
def item_delete_page(request: Request, item_id: int, db: Session = Depends(get_db)):
    item = db.get(TextItem, item_id)
    if item is None:
        raise HTTPException(status_code=404)
    return render(request,"textitem_confirm_delete.html",item=item)

@router.post("/items/{item_id}/delete/", name="item_delete")
def item_delete(item_id: int, db: Session = Depends(get_db)):
    item = db.get(TextItem, item_id)
    if item is None:
        raise HTTPException(status_code=404)
    db.delete(item)
    db.commit()
    return RedirectResponse("/items/", status_code=303)
