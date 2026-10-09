"""url_utils.join_endpoint 测试。"""

from injection_guard.url_utils import join_endpoint


def test_appends_endpoint_to_base():
    assert (
        join_endpoint("http://127.1.1.1:8094/v1", "/chat/completions")
        == "http://127.1.1.1:8094/v1/chat/completions"
    )


def test_trailing_slash_on_base():
    assert (
        join_endpoint("http://127.1.1.1:8096/v1/", "/embeddings")
        == "http://127.1.1.1:8096/v1/embeddings"
    )


def test_path_without_leading_slash():
    assert (
        join_endpoint("http://127.1.1.1:8096/v1", "embeddings")
        == "http://127.1.1.1:8096/v1/embeddings"
    )


def test_idempotent_when_full_path_given():
    full = "http://127.1.1.1:8094/v1/chat/completions"
    assert join_endpoint(full, "/chat/completions") == full
