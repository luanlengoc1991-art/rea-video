import sys
import types

import pytest

from rea_video import overlay


@pytest.fixture
def fake_rapidocr(monkeypatch):
    calls = []

    class RapidOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", types.SimpleNamespace(RapidOCR=RapidOCR))
    monkeypatch.setattr(overlay, "_OCR", {})
    monkeypatch.setattr(overlay, "_GPU", {})
    monkeypatch.setenv("REA_OCR_GPU", "0")
    return calls


CPU = {"intra_op_num_threads": 1, "inter_op_num_threads": 1}  # a few threads per engine, not every core


def test_default_language_is_zh(monkeypatch):
    monkeypatch.delenv("REA_OCR_LANG", raising=False)
    assert overlay.ocr_lang() == "zh"


def test_unknown_language_is_rejected(monkeypatch):
    monkeypatch.setenv("REA_OCR_LANG", "fr")
    with pytest.raises(ValueError):
        overlay.ocr_lang()


def test_zh_uses_bundled_model(monkeypatch, fake_rapidocr):
    monkeypatch.setenv("REA_OCR_LANG", "zh")
    assert overlay.get_ocr() is overlay.get_ocr()
    assert fake_rapidocr == [CPU]


def test_ja_uses_japanese_recognizer(monkeypatch, tmp_path, fake_rapidocr):
    model = tmp_path / "japan.onnx"
    model.write_bytes(b"x")
    monkeypatch.setenv("REA_OCR_REC_MODEL", str(model))
    monkeypatch.setenv("REA_OCR_LANG", "ja")
    overlay.get_ocr()
    monkeypatch.setenv("REA_OCR_LANG", "zh")
    overlay.get_ocr()
    assert fake_rapidocr == [{**CPU, "rec_model_path": str(model)}, CPU]  # one instance per language


def test_cached_model_with_matching_checksum_is_not_downloaded(monkeypatch, tmp_path):
    monkeypatch.delenv("REA_OCR_REC_MODEL", raising=False)
    monkeypatch.setenv("REA_VIDEO_MODELS", str(tmp_path))
    name, _url, _sha = overlay._REC_MODELS["ja"]
    (tmp_path / name).write_bytes(b"model")
    monkeypatch.setitem(overlay._REC_MODELS, "ja", (name, "http://invalid", overlay._sha256(tmp_path / name)))
    monkeypatch.setattr(overlay, "fetch", lambda *a: pytest.fail("should not download"))
    assert overlay.rec_model("ja") == tmp_path / name


def test_ocr_lang_pref_is_validated(monkeypatch, tmp_path):
    from rea_video import memory

    monkeypatch.setenv("REA_VIDEO_HOME", str(tmp_path))
    with pytest.raises(ValueError):
        memory.set_pref("ocr_lang", "fr")


def test_rapidocr3_is_used_when_rapidocr_onnxruntime_is_missing(monkeypatch, tmp_path):
    import numpy as np

    made = []

    class Output:
        boxes = np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]], dtype=np.float32)
        txts = ("字幕",)
        scores = (0.9,)
        elapse = 0.1

    class RapidOCR:
        def __init__(self, params=None):
            made.append(params)

        def __call__(self, img):
            return Output() if img is not None else types.SimpleNamespace(boxes=None, txts=None, scores=None, elapse=0.0)

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)  # import raises ImportError
    monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=RapidOCR))
    monkeypatch.setattr(overlay, "_OCR", {})
    model = tmp_path / "japan.onnx"
    model.write_bytes(b"x")
    monkeypatch.setenv("REA_OCR_REC_MODEL", str(model))
    monkeypatch.setenv("REA_OCR_LANG", "ja")
    ocr = overlay.get_ocr()
    assert made[0]["Rec.model_path"] == str(model)
    res, _ = ocr(np.zeros((5, 10, 3), np.uint8))
    assert res == [[[[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]], "字幕", 0.9]]
    assert ocr(None)[0] is None  # nothing found: same as rapidocr-onnxruntime
    for box, text, conf in res:  # the shape overlay / packaging iterate over
        assert len(box) == 4 and text == "字幕" and conf == 0.9


def test_each_thread_gets_its_own_cpu_engine(monkeypatch, fake_rapidocr):
    import threading

    monkeypatch.setenv("REA_OCR_LANG", "zh")
    monkeypatch.delenv("REA_OCR_THREADS", raising=False)
    main = overlay.get_ocr()
    got = []
    t = threading.Thread(target=lambda: got.append(overlay.get_ocr()))
    t.start()
    t.join()
    assert got[0] is not main and len(fake_rapidocr) == 2


def test_graphics_card_engine_is_shared_one_call_at_a_time(monkeypatch, fake_rapidocr):
    monkeypatch.setenv("REA_OCR_LANG", "zh")
    monkeypatch.setenv("REA_OCR_GPU", "1")
    monkeypatch.setattr(overlay, "ocr_gpu", lambda: True)
    used = []
    monkeypatch.setattr(overlay, "_gpu_engine", lambda **kw: (lambda img: used.append("gpu") or ([], 0)))
    eng = overlay.get_ocr()
    eng(None)
    assert used == ["gpu"] and fake_rapidocr == []  # no CPU engine made while the card is free
    with overlay._GPU_USE:  # card busy: this thread falls back to its own CPU engine
        try:
            eng(None)
        except TypeError:  # the fake CPU engine is not callable; reaching it is what this checks
            pass
    assert used == ["gpu"] and fake_rapidocr == [CPU]
