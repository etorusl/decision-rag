import json

import httpx
import pytest

from decision_rag.backends.jev import JevBackend


def make_backend(handler, **kwargs):
    return JevBackend(api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_binary_wire_format_and_result_order():
    def handle(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert body == {
            "model": "jev-1.13.0", "state": "Both candidates are listed here.",
            "questions": {"q0": {"type": "noul", "instructions": "First?"},
                          "q1": {"type": "noul", "instructions": "Second?"}},
        }
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "q1": {"noul": .2}, "q0": {"noul": .9}}, "usage": {"input_tokens": 40}})
    out = make_backend(handle).binary("Both candidates are listed here.", ["First?", "Second?"])
    assert out.probabilities == [.9, .2]
    assert out.input_tokens == 40


def test_choice_maps_options_and_restores_order():
    def handle(request):
        body = json.loads(request.content)
        assert body["questions"]["choice"]["criteria"] == {"o0": "Passage 9", "o1": "Passage 2"}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "choice": {"probabilities": {"o1": .7, "o0": .3}}}})
    assert make_backend(handle).choice("state", "Best?", ["Passage 9", "Passage 2"]).probabilities == [.3, .7]


def test_openrouter_uses_pinned_snapshot():
    def handle(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
        return httpx.Response(200, json={"model": "typesafe/jev-1.14", "answers": {"q0": {"noul": .9}}})
    with pytest.raises(ValueError, match="model changed"):
        make_backend(handle, provider="openrouter").binary("state", ["Relevant?"])


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), True, "0.2"])
def test_invalid_binary_probability_is_rejected(value):
    def handle(request):
        # JSON NaN is rejected by httpx's encoder, so supply a raw response.
        return httpx.Response(200, content=json.dumps({"model": "jev-1.13.0", "answers": {"q0": {"noul": value}}}))
    with pytest.raises(ValueError, match="probability"):
        make_backend(handle).binary("state", ["Relevant?"])


def test_missing_answer_is_rejected():
    with pytest.raises(ValueError, match="missing answer"):
        make_backend(lambda r: httpx.Response(200, json={"model": "jev-1.13.0", "answers": {}})).binary("", ["Q"])


def test_retries_rate_limit_but_not_bad_request(monkeypatch):
    monkeypatch.setattr("decision_rag.backends.jev.time.sleep", lambda n: None)
    requests = []
    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q0": {"noul": .5}}})
    assert make_backend(handle).binary("", ["Q"]).probabilities == [.5]
    assert len(requests) == 2
    with pytest.raises(httpx.HTTPStatusError):
        make_backend(lambda r: httpx.Response(400)).binary("", ["Q"])


def test_transport_error_on_final_attempt_propagates(monkeypatch):
    monkeypatch.setattr("decision_rag.backends.jev.time.sleep", lambda n: None)
    calls = []
    def handle(request):
        calls.append(request)
        raise httpx.ConnectError("unreachable", request=request)
    with pytest.raises(httpx.ConnectError):
        make_backend(handle, max_retries=1).binary("", ["Q"])
    assert len(calls) == 2


def test_empty_batch_has_no_request():
    def handle(request):
        raise AssertionError("No request expected")
    output = make_backend(handle).binary("", [])
    assert output.probabilities == [] and output.forward_passes == 0


def test_choice_count_and_distribution_validation():
    backend = make_backend(lambda r: httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
        "choice": {"probabilities": {"o0": .2, "o1": .2}}}}))
    with pytest.raises(ValueError, match="255"):
        backend.choice("", "Q", ["a"] * 256)
    with pytest.raises(ValueError, match="sum"):
        backend.choice("", "Q", ["a", "b"])
