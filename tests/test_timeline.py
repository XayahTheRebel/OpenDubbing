import pytest

from opendubbing.core.timeline import Phoneme, Sentence, Timeline, Word


class TestTimeline:
    def test_empty_timeline(self):
        timeline = Timeline()
        assert timeline.sentences == []
        assert timeline.version == "1.0"

    def test_append_and_len(self):
        timeline = Timeline()
        timeline.append(Sentence(id="s1", text="Hello"))
        assert len(timeline) == 1
        assert timeline.sentences[0].text == "Hello"

    def test_roundtrip_dict(self):
        timeline = Timeline(
            source_language="en",
            target_language="zh",
            sentences=[
                Sentence(
                    id="s1",
                    text="Hello",
                    words=[
                        Word(
                            text="Hello",
                            phonemes=[Phoneme(symbol="h", start=0.0, end=0.1)],
                        )
                    ],
                )
            ],
        )
        data = timeline.to_dict()
        restored = Timeline.from_dict(data)
        assert restored.source_language == "en"
        assert len(restored.sentences[0].words[0].phonemes) == 1

    def test_jsonl_save_and_load(self, tmp_path):
        timeline = Timeline()
        timeline.append(Sentence(id="s1", text="Hello world"))
        timeline.append(Sentence(id="s2", text="Goodbye"))
        path = tmp_path / "timeline.jsonl"
        timeline.save(path)
        loaded = Timeline.load(path)
        assert len(loaded) == 2
        assert loaded.sentences[0].text == "Hello world"

    def test_update_sentence(self):
        timeline = Timeline()
        timeline.append(Sentence(id="s1", text="Hello"))
        timeline.update_sentence("s1", translation="你好")
        assert timeline.sentences[0].translation == "你好"

    def test_update_missing_sentence_raises(self):
        timeline = Timeline()
        with pytest.raises(KeyError):
            timeline.update_sentence("missing", text="x")
