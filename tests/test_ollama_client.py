from reid_pipeline.reranking.ollama_client import OllamaVLMClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"response": '{"same_person": true, "confidence": 0.9}'}


def test_ollama_client_returns_model_response(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())
    client = OllamaVLMClient(model="qwen2.5vl:7b")
    assert client.generate([b"image"], "compare") == '{"same_person": true, "confidence": 0.9}'
