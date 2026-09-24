"""Unit tests for laformat: quantization, container layout, aliasing."""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import laformat as L


def test_quantize_blocks():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 64), dtype=np.float32) * 0.3
    raw = L.quantize_q8_0(x)
    assert len(raw) == 3 * 2 * L.BLOCK_Q8_0_BYTES
    y = L.dequantize_q8_0(raw, [64, 3]).reshape(3, 64)
    d = (np.abs(x.reshape(3, 2, 32)).max(axis=2) / 127.0).astype(np.float16).astype(np.float32)
    dper = np.repeat(d, 32, axis=1)
    err = np.abs(y - x)
    assert np.all(err <= dper * 0.6 + 1e-6), err.max()
    print("quantize_blocks ok  maxerr=%.3e" % err.max())


def test_quantize_known_values():
    x = np.zeros((1, 32), dtype=np.float32)
    x[0, 0] = 1.0
    x[0, 1] = -0.5
    raw = L.quantize_q8_0(x)
    d = np.frombuffer(raw[:2], dtype="<f2")[0]
    qs = np.frombuffer(raw[2:34], dtype="i1")
    assert d == np.float16(1.0 / 127.0), d
    assert qs[0] == 127 and qs[1] == -64, (qs[0], qs[1])
    print("quantize_known_values ok")


def test_roundf_semantics():
    a = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=np.float32)
    r = L._roundf(a)
    assert list(r) == [1.0, 2.0, 3.0, -1.0, -2.0, -3.0], r
    print("roundf ok")


def test_quantizable():
    assert L.quantizable([2048, 2048])
    assert not L.quantizable([4304, 1152])   # ne0 = 4304 is not a multiple of 32
    assert not L.quantizable([2048])         # 1-D
    assert not L.quantizable([1152, 3, 14, 14])
    print("quantizable ok")


def test_container_streaming_and_alias():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.laqt")
        rng = np.random.default_rng(1)
        wq = rng.standard_normal((3, 64), dtype=np.float32)
        wb = rng.standard_normal(10, dtype=np.float32)
        w = L.ContainerWriter(p, config={"lm.hidden": 2048}, tokens=["a", "b"],
                              token_types=[1, 4], merges=["a b"])
        w.plan("w1", "q8_0", [64, 3])
        w.plan("b1", "f32", [10])
        w.plan("w1_alias", "q8_0", [64, 3], alias_of="w1")
        size = w.begin()
        w.append("w1", L.quantize_q8_0(wq))
        w.append("b1", wb.astype("<f4").tobytes())
        w.finish()
        assert os.path.getsize(p) == size, (os.path.getsize(p), size)
        c = L.Container(p)
        assert c.config["lm.hidden"] == 2048 and c.tokens == ["a", "b"]
        assert sorted(c.names()) == ["b1", "w1", "w1_alias"]
        np.testing.assert_array_equal(c.f32("b1"), wb)
        np.testing.assert_array_equal(c.f32("w1"), c.f32("w1_alias"))
        assert c.tensors["w1"]["offset"] == c.tensors["w1_alias"]["offset"]
        # the aliased entry must not have cost another payload
        assert size < 4096 * 3 + len(L.quantize_q8_0(wq)) + 4096
        for t in c.header["tensors"]:
            assert t["offset"] % 4096 == 0, t
            assert t["offset"] + t["nbytes"] <= size
        c.close()
        print("container_streaming_and_alias ok  size=%d" % size)


if __name__ == "__main__":
    test_roundf_semantics()
    test_quantize_known_values()
    test_quantizable()
    test_quantize_blocks()
    test_container_streaming_and_alias()
    print("all laformat tests passed")
