import importlib
import sys
from types import SimpleNamespace


def test_main_reuses_chat_for_follow_up_questions(monkeypatch):
    ask_module = importlib.import_module("app.ask")

    class FakeChat:
        def __init__(self):
            self.messages = []

        def send_message(self, message):
            self.messages.append(message)
            return SimpleNamespace(text=f"回答: {message}", usage_metadata=None)

    chat = FakeChat()
    created_models = []

    class FakeChats:
        def create(self, *, model):
            created_models.append(model)
            return chat

    class FakeClient:
        def __init__(self, *, api_key):
            self.chats = FakeChats()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    questions = iter(["追加質問", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask_module.main()

    assert created_models == [ask_module.MODEL]
    assert chat.messages == ["最初の質問", "追加質問"]