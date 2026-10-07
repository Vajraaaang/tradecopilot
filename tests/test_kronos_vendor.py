"""The optional upstream model stays pinned, attributed, and packaged without credentials."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def test_pinned_upstream_files_and_only_declared_import_patch():
    root = Path(__file__).resolve().parents[1]
    vendor = root / 'src/tradecopilot/_vendor/kronos'
    assert (vendor / 'UPSTREAM.json').is_file(), 'Kronos source must be pinned before use'
    manifest = json.loads((vendor / 'UPSTREAM.json').read_text())
    assert manifest['commit'] == '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
    for name in ('kronos.py', 'module.py', 'LICENSE'):
        data = (vendor / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['packaged_sha256']
    source = (vendor / 'kronos.py').read_text()
    assert 'sys.path.append' not in source
    assert 'from tradecopilot._vendor.kronos.module import *' in source
    assert 'MIT License' in (vendor / 'LICENSE').read_text()


def test_tiny_native_cpu_tokenizer_forward_and_autoregression_without_pretrained_weights():
    torch = pytest.importorskip("torch")
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("einops")
    from tradecopilot._vendor.kronos.kronos import (
        Kronos,
        KronosPredictor,
        KronosTokenizer,
        auto_regressive_inference,
    )

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.random.default_generator.manual_seed(123)
            tokenizer = KronosTokenizer(
                d_in=6, d_model=8, n_heads=1, ff_dim=8, n_enc_layers=1, n_dec_layers=1,
                ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=1, s2_bits=1,
                beta=0.1, gamma0=1, gamma=1, zeta=1, group_size=2,
            ).eval()
            model = Kronos(
                s1_bits=1, s2_bits=1, n_layers=1, d_model=8, n_heads=1, ff_dim=8,
                ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=False,
            ).eval()
            values = torch.linspace(-0.5, 0.5, 24).reshape(1, 4, 6)
            x_stamp = torch.tensor([[[31, 9, 3, 1, 10], [32, 9, 3, 1, 10],
                                     [33, 9, 3, 1, 10], [34, 9, 3, 1, 10]]], dtype=torch.float32)
            y_stamp = torch.tensor([[[35, 9, 3, 1, 10], [36, 9, 3, 1, 10]]], dtype=torch.float32)
            s1, s2 = tokenizer.encode(values, half=True)
            full = tokenizer.encode(values)
            assert s1.dtype == s2.dtype == torch.int64 and s1.shape == s2.shape == (1, 4)
            assert ((s1 >= 0) & (s1 <= 1)).all() and ((s2 >= 0) & (s2 <= 1)).all()
            assert torch.equal(full, s1 + (s2 << 1))
            decoded = tokenizer.decode(full)
            assert decoded.shape == values.shape and torch.isfinite(decoded).all()
            torch.testing.assert_close(decoded, tokenizer.decode((s1, s2), half=True))
            logits = model(s1, s2, x_stamp, use_teacher_forcing=True, s1_targets=s1)
            assert len(logits) == 2 and all(logit.shape == (1, 4, 2) for logit in logits)
            assert all(torch.isfinite(logit).all() for logit in logits)
            generated = []
            for _ in range(2):
                torch.random.default_generator.manual_seed(456)
                generated.append(auto_regressive_inference(
                    tokenizer, model, values, x_stamp, y_stamp, max_context=4, pred_len=2,
                    sample_count=2, verbose=False,
                ))
            assert generated[0].shape == (1, 4, 6) and np.isfinite(generated[0]).all()
            np.testing.assert_array_equal(generated[0], generated[1])
            assert generated[0][:, -2:, :].shape == (1, 2, 6)
            predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=4)
            columns = ["open", "high", "low", "close", "volume", "amount"]
            frame = pandas.DataFrame(values[0].numpy(), columns=columns)
            times = pandas.Series(pandas.date_range("2026-10-01 09:31", periods=6, freq="min"))
            results = predictor.predict_batch(
                [frame, frame.copy()], [times.iloc[:4], times.iloc[:4]], [times.iloc[4:], times.iloc[4:]],
                pred_len=2, sample_count=1, verbose=False,
            )
            assert len(results) == 2
            for result in results:
                assert result.shape == (2, 6) and list(result.columns) == columns
                assert result.index.to_list() == times.iloc[4:].to_list()
                assert np.isfinite(result.to_numpy()).all()
    finally:
        torch.set_num_threads(threads)
