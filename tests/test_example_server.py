from tools.example_server import ping


def test_ping_default():
    assert ping() == "pong"


def test_ping_echoes_message():
    assert ping("hello") == "hello"
