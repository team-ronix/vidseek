"""
Quick smoke-test for Transformer and SentenceSegmenter model loading + embedding.
Run from the project root:  python test_embeddings.py
"""

import json
import tempfile
import os
import torch

# ── helpers ───────────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning enables computers to learn from data.",
    "Video search retrieves relevant segments from footage.",
]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a * b).sum().item()  # vectors are L2-normalised by the model


# ── Test 1: Transformer ───────────────────────────────────────────────────────

def test_transformer():
    print("\n" + "=" * 60)
    print("TEST 1 — Transformer (Transformer.py)")
    print("=" * 60)

    transcripts = [
        {"text": t, "video_path": "video1.mp4", "start": i * 10.0, "end": i * 10.0 + 9.0}
        for i, t in enumerate(SAMPLE_TEXTS)
    ]

    from Transformer import Transformer

    print("[1/3] Loading model …")
    t = Transformer(transcripts)
    print("      Model loaded OK")

    print("[2/3] Running transform() …")
    t.transform()
    embeddings = t.get_embeddings()
    metadata   = t.get_metadata()
    assert len(embeddings) == len(metadata), "embedding / metadata length mismatch"
    expected = len(transcripts)
    assert len(embeddings) == expected, f"expected {expected} embeddings, got {len(embeddings)}"
    print(f"      transform() OK — {len(embeddings)} embeddings, shape {embeddings[0].shape}")

    print("[3/3] transform_single_text() + similarity …")
    emb_a = t.transform_single_text("machine learning algorithms")
    emb_b = t.transform_single_text("deep neural networks")
    emb_c = t.transform_single_text("cooking pasta recipe")
    sim_ab = cosine_similarity(emb_a, emb_b)
    sim_ac = cosine_similarity(emb_a, emb_c)
    print(f"      'machine learning' vs 'deep neural networks' -> sim={sim_ab:.4f}")
    print(f"      'machine learning' vs 'cooking pasta recipe'  -> sim={sim_ac:.4f}")
    assert sim_ab > sim_ac, "expected tech texts to be more similar than unrelated text"
    print("      Similarity ordering OK")

    print("\nTEST 1 PASSED")


# ── Test 2: SentenceSegmenter ─────────────────────────────────────────────────

def test_sentence_segmenter():
    print("\n" + "=" * 60)
    print("TEST 2 — SentenceSegmentation (audio/SentenceSegmenter.py)")
    print("=" * 60)

    chunks = [
        {"text": "Machine learning is a subset of artificial intelligence.",
         "timestamp": [0.0, 4.0]},
        {"text": "Neural networks are inspired by the human brain.",
         "timestamp": [4.0, 8.0]},
        {"text": "Deep learning uses many layers of neurons.",
         "timestamp": [8.0, 12.0]},
        {"text": "Now let us talk about cooking delicious pasta.",
         "timestamp": [12.0, 16.0]},
        {"text": "Boil water and add salt before putting in the pasta.",
         "timestamp": [16.0, 20.0]},
    ]
    transcript_data = {"chunks": chunks}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(transcript_data, f)
        tmp_path = f.name

    try:
        from audio.SentenceSegmenter import SentenceSegmentation

        print("[1/3] Loading model …")
        seg = SentenceSegmentation(
            video_path="dummy_video.mp4",
            transcript_json=tmp_path,
            similarity_threshold=0.3
        )
        print("      Model loaded OK")

        print("[2/3] Running segment() …")
        segments = seg.segment()
        print(f"      segment() OK — produced {len(segments)} segment(s)")
        for i, s in enumerate(segments):
            print(f"        [{i}] {s['start']:.2f}s–{s['end']:.2f}s  \"{s['text'][:60]}\"")

        print("[3/3] Validating segment structure …")
        for s in segments:
            assert "text"       in s, "missing 'text'"
            assert "start"      in s, "missing 'start'"
            assert "end"        in s, "missing 'end'"
            assert "video_path" in s, "missing 'video_path'"
            assert s["video_path"] == "dummy_video.mp4"
        print("      Segment structure OK")

    finally:
        os.unlink(tmp_path)

    print("\nTEST 2 PASSED")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_transformer()
    test_sentence_segmenter()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
