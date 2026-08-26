from pathlib import Path


SOURCE = (Path(__file__).parent / "vercel_frontdoor/index.html").read_text(encoding="utf-8")


def test_frontend_handles_platform_text_errors_and_large_uploads():
    assert "readJsonResponse" in SOURCE
    assert "response.text()" in SOURCE
    assert "Request returned a non-JSON response" not in SOURCE
    assert "Vercel's platform limit" in SOURCE
    assert "3 * 1024 * 1024" in SOURCE


if __name__ == "__main__":
    test_frontend_handles_platform_text_errors_and_large_uploads()
    print("Frontend API error handling checks passed")
