from fastapi.testclient import TestClient
from main import app
import sentiment.router as router_mod
import sentiment.tasks as tasks_mod

client = TestClient(app)

def test_sync(monkeypatch):
    monkeypatch.setattr(router_mod,"analyze_text",lambda text, use_cache=False: {"label":"Positive", "score":0.9})
    assert client.get("/health").status_code==200
    assert "Positive" in client.post("/predict/",data={"text":"기분좋음"}).text
    assert client.post("/predict/",data={"text":"  "}).status_code==400

def test_async_enqueued_and_task(monkeypatch):
    enqueued=[]
    monkeypatch.setattr(router_mod.analyze_textitem_task,"delay", lambda *args: enqueued.append(args))
    monkeypatch.setattr(tasks_mod, "analyze_text", lambda text, use_cache=False: {"label":"Positive","score":0.99})
    r=client.post("/new/",data={"text":"좋아요"},follow_redirects=False)
    assert r.status_code==303
    assert len(enqueued)==1
    item_id, rev=enqueued[-1]
    uri=r.headers["location"]
    assert client.get(f"/items/{item_id}/status").json()["status"]=="PENDING"
    assert tasks_mod.analyze_textitem_task.run(item_id,rev)["status"]=="DONE"
    assert client.get(f"/items/{item_id}/status").json()["status"]=="DONE"
    assert "Positive" in client.get(uri).text
    r2=client.post(uri+"edit/",data={"text":"싫어요"},follow_redirects=False)
    assert r2.status_code==303
    assert enqueued[-1]==(item_id,rev+1)
    assert tasks_mod.analyze_textitem_task.run(item_id,rev)["status"]=="ignored"
    assert tasks_mod.analyze_textitem_task.run(item_id,rev+1)["status"]=="DONE"
    assert client.post(uri+"delete/",follow_redirects=False).status_code==303
    assert client.get(uri).status_code==404

def test_redis_cache_hit_and_miss(monkeypatch):
    import sentiment.services as svc
    import json
    class InMemoryRedis:
        def __init__(self): self.store={}
        def get(self, key): return self.store.get(key)
        def setex(self, key, seconds, value):
            assert seconds==600
            self.store[key] = value
    fake=InMemoryRedis()
    count=[]
    def classifier(text, **kwargs):
        count.append(text)
        return [{"label":"Positive", "score":0.98}]
    monkeypatch.setattr(svc, "redis_cache", fake)
    monkeypatch.setattr(svc, "get_classifier", lambda: classifier)
    assert svc.analyze_text(" 좋은 하루 ",use_cache=True)["label"] == "Positive"
    assert svc.analyze_text("좋은 하루", use_cache=True)["label"] == "Positive"
    assert count == ["좋은 하루"]   # 두 번째 요청은 캐시 HIT

def test_broker_unavailable_status_failed(monkeypatch):
    def down(*args):
        raise ConnectionError("Redis is not connected")
    monkeypatch.setattr(router_mod.analyze_textitem_task,"delay",down)
    res=client.post('/new/',data={'text':'브로커 오류 테스트'},follow_redirects=False)
    assert res.status_code==303
    url=res.headers['location']
    assert 'FAILED' in client.get(url).text

def test_worker_failure_updates_db(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('fake hf error')
    calls=[]
    monkeypatch.setattr(router_mod.analyze_textitem_task,'delay',lambda *args: calls.append(args))
    monkeypatch.setattr(tasks_mod,'analyze_text',fail)
    res=client.post('/new/',data={'text':'워커 오류 테스트'},follow_redirects=False)
    item_id,rev=calls[-1]
    import pytest
    with pytest.raises(RuntimeError):
        tasks_mod.analyze_textitem_task.run(item_id,rev)
    assert client.get(f'/items/{item_id}/status').json()['status']=='FAILED'
