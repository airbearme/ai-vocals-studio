from pathlib import Path


SOURCE = (Path(__file__).parent / "vercel_frontdoor/api/clone-and-speak.js").read_text(encoding="utf-8")


def test_upstream_plain_text_errors_are_not_parsed_twice():
    assert "const body = await response.text();" in SOURCE
    assert "JSON.parse(body)" in SOURCE
    assert "readJsonResponse(voiceResponse" in SOURCE
    assert "response.json()" not in SOURCE


if __name__ == "__main__":
    test_upstream_plain_text_errors_are_not_parsed_twice()
    print("Upstream error handling checks passed")
