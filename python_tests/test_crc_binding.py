import pytest

sp_vision_bindings = pytest.importorskip("sp_vision_bindings")


def test_crc8_roundtrip() -> None:
    data = bytes([0x01, 0x02, 0x03, 0x04])
    crc = sp_vision_bindings.crc8(data)
    assert sp_vision_bindings.check_crc8(data + bytes([crc]))


def test_crc16_roundtrip() -> None:
    data = bytes([0x01, 0x02, 0x03, 0x04])
    crc = sp_vision_bindings.crc16(data)
    payload = data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    assert sp_vision_bindings.check_crc16(payload)
