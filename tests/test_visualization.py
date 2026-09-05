from types import SimpleNamespace

from fish_speech.content_sequence import ContentSequence
from fish_speech.conversation import Conversation
from fish_speech.tokenizer import AUDIO_EMBED_TOKEN


class FakeTokenizer:
    semantic_begin_id = 100
    semantic_end_id = 199
    audio_token_id = 200

    def get_token_id(self, token):
        assert token == AUDIO_EMBED_TOKEN
        return self.audio_token_id

    def decode(self, tokens):
        return {
            1: "hello",
            self.audio_token_id: AUDIO_EMBED_TOKEN,
        }.get(tokens[0], f"<|semantic:{tokens[0] - self.semantic_begin_id}|>")


class FakeScalar(int):
    def item(self):
        return int(self)


def encoded(tokens, labels):
    return SimpleNamespace(
        tokens=[FakeScalar(token) for token in tokens],
        labels=[FakeScalar(label) for label in labels],
    )


def test_content_sequence_visualize_without_color(monkeypatch, capsys):
    sequence = ContentSequence()
    monkeypatch.setattr(
        sequence, "encode", lambda *args, **kwargs: encoded([1], [-100])
    )

    sequence.visualize(FakeTokenizer(), use_color=False)

    assert capsys.readouterr().out == "hello\n"


def test_content_sequence_visualize_merges_audio_and_semantic_tokens(
    monkeypatch, capsys
):
    sequence = ContentSequence()
    monkeypatch.setattr(
        sequence,
        "encode",
        lambda *args, **kwargs: encoded(
            [100, 101, 200, 200, 200, 1],
            [-100, -100, -100, -100, 1, 1],
        ),
    )

    sequence.visualize(
        FakeTokenizer(),
        merge_semantic_tokens=True,
        merge_audio_tokens=True,
        use_color=False,
    )

    assert (
        capsys.readouterr().out
        == "[<|semantic|>x2][<|audio_pad|>x2][<|audio_pad|>x1]hello\n"
    )


def test_conversation_visualize_forwards_options(monkeypatch):
    sequence = ContentSequence()
    monkeypatch.setattr(Conversation, "_build_content_sequence", lambda self: sequence)
    received = {}

    def visualize(tokenizer, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(sequence, "visualize", visualize)

    Conversation().visualize(
        FakeTokenizer(),
        merge_semantic_tokens=True,
        merge_audio_tokens=True,
        use_color=False,
    )

    assert received == {
        "ignore_loss_tokens": [],
        "merge_semantic_tokens": True,
        "merge_audio_tokens": True,
        "use_color": False,
    }
