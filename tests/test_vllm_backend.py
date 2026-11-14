import json
import math

import httpx
import pytest

from decision_rag.backends.vllm import VLLMDecisionBackend, server_root


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        labels = {"no": 10, "yes": 11, **{chr(65 + i): 100 + i for i in range(26)}}
        return [labels[text]] if text in labels else [ord(c) + 1000 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<user>" + messages[0]["content"] + "</user><assistant>"


class LocalServer:
    def __init__(self):
        self.requests = []
        self.mode = "processed_logprobs"
        self.model_id = "Qwen/Qwen3-8B"
        self.server_info = True
        self.response_mutator = None
        self.tokenizer_mismatch = False

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={"data": [{"id": self.model_id, "max_model_len": 8192}]})
        if request.url.path == '/version':
            return httpx.Response(200, json={"version": "0.11.2"})
        if request.url.path == '/server_info':
            if not self.server_info:
                return httpx.Response(404, json={"detail": "Not found"})
            return httpx.Response(200, json={"vllm_config": {
                "model_config": {"model": self.model_id, "logprobs_mode": self.mode,
                                 "max_logprobs": 26, "max_model_len": 8192, "hf_token": "secret"},
                "cache_config": {"enable_prefix_caching": True},
            }})
        if request.url.path == '/tokenize':
            tokens = TinyTokenizer().encode(body['prompt'])
            return httpx.Response(200, json={"tokens": tokens + ([999] if self.tokenizer_mismatch else [])})
        assert request.url.path == '/v1/completions'
        prompts, labels = body['prompt'], body['allowed_token_ids']
        choices = []
        for index in range(len(prompts)):
            weights = [i + 1 + index for i in range(len(labels))]
            probs = [value / sum(weights) for value in weights]
            choices.append({"index": index, "text": "ignored", "finish_reason": "length", "logprobs": {
                "tokens": [f"token_id:{labels[-1]}"],
                "top_logprobs": [{f"token_id:{token_id}": math.log(p)
                                  for token_id, p in zip(labels, probs, strict=True)}],
            }})
        prompt_tokens = sum(map(len, prompts))
        response = {"choices": list(reversed(choices)), "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": len(prompts),
            "prompt_tokens_details": {"cached_tokens": prompt_tokens // 2},
        }}
        if self.response_mutator:
            self.response_mutator(response)
        return httpx.Response(200, json=response)


def backend(server=None, **kwargs):
    server = server or LocalServer()
    client = httpx.Client(transport=httpx.MockTransport(server))
    return VLLMDecisionBackend(tokenizer=TinyTokenizer(), client=client, **kwargs), server


def completions(server):
    return [body for _, path, body in server.requests if path == '/v1/completions']


def test_binary_payload_normalized_logprobs_and_response_indexes():
    model, server = backend(prime_prefix=False)
    out = model.binary('context contains all candidates', ['question1', 'question2'])
    assert out.probabilities == pytest.approx([2/3, 3/5])
    assert out.forward_passes is None
    assert out.usage['requests'] == 1
    assert out.usage['prompt_requests'] == out.usage['completion_tokens'] == 2
    assert out.usage['cached_tokens'] == out.input_tokens // 2
    payload = completions(server)[0]
    assert payload['allowed_token_ids'] == [10, 11]
    assert payload['return_tokens_as_token_ids'] is True
    assert payload['max_tokens'] == payload['min_tokens'] == payload['n'] == 1
    assert payload['temperature'] == payload['top_p'] == payload['repetition_penalty'] == 1
    assert payload['top_k'] == -1
    assert payload['logprobs'] == 2
    assert payload['add_special_tokens'] is False
    assert len(payload['prompt']) == 2
    assert len(payload['cache_salt']) == 32
    assert 'secret' not in json.dumps(model.runtime_metadata)


def test_priming_counts_real_first_judgment_and_batched_work():
    model, server = backend(batch_size=2)
    out = model.binary('shared', ['q1','q2','q3','q4'])
    assert [len(row['prompt']) for row in completions(server)] == [1,2,1]
    assert out.usage['requests'] == 3
    assert out.usage['prompt_requests'] == out.usage['completion_tokens'] == 4
    assert out.probabilities == pytest.approx([2/3,2/3,3/5,2/3])
    assert len({row['cache_salt'] for row in completions(server)}) == 1


def test_cache_salt_is_stable_inside_query_and_fresh_for_next_query():
    model, server = backend(prime_prefix=False)
    model.prepare_query()
    model.binary('c', ['q'])
    model.binary('next stage', ['q'])
    model.prepare_query()
    model.binary('c', ['q'])
    salts = [row['cache_salt'] for row in completions(server)]
    assert salts[0] == salts[1]
    assert salts[2] != salts[0]


def test_warm_cache_policy_omits_salt():
    model, server = backend(cache_policy='warm')
    model.prepare_query()
    model.binary('c', ['q'])
    assert 'cache_salt' not in completions(server)[0]


def test_choice_uses_single_token_letter_probabilities():
    model, server = backend()
    out = model.choice('context', 'Which?', ['one','two','three'])
    assert out.probabilities == pytest.approx([1/6,2/6,3/6])
    assert completions(server)[0]['allowed_token_ids'] == [100,101,102]
    assert out.usage['completion_tokens'] == 1
    assert model.choice('c','q',['only']).probabilities == [1.]
    with pytest.raises(ValueError, match='26'):
        model.choice('c','q',['x']*27)


def test_server_info_not_required_in_production():
    server = LocalServer()
    server.server_info = False
    model, _ = backend(server)
    assert model.runtime_metadata['server_config_available'] is False
    assert model.binary('c',['q']).probabilities == pytest.approx([2/3])


def test_wrong_probability_mode_detected_before_scoring():
    server = LocalServer()
    server.mode = 'raw_logprobs'
    with pytest.raises(ValueError, match='processed_logprobs'):
        backend(server)
    assert not completions(server)


def test_model_and_tokenizer_mismatch_fail_explicitly():
    server = LocalServer()
    server.model_id = 'different'
    with pytest.raises(ValueError, match='does not serve'):
        backend(server)
    server = LocalServer()
    server.tokenizer_mismatch = True
    with pytest.raises(ValueError, match='tokenizers differ'):
        backend(server)


@pytest.mark.parametrize('error', ['missing','unnormalized','positive_logprob','duplicate_index','token_usage','cache_usage'])
def test_malformed_scoring_responses_are_not_silently_accepted(error):
    server = LocalServer()
    def mutate(body):
        first = body['choices'][0]['logprobs']['top_logprobs'][0]
        if error == 'missing':
            first.pop('token_id:10')
        elif error == 'unnormalized':
            first.update({'token_id:10': -10.,'token_id:11': -10.})
        elif error == 'positive_logprob':
            first['token_id:10'] = 1.0
        elif error == 'duplicate_index':
            body['choices'][0]['index'] = body['choices'][1]['index']
        elif error == 'token_usage':
            body['usage']['completion_tokens'] += 1
        else:
            body['usage']['prompt_tokens_details']['cached_tokens'] = -1
    server.response_mutator = mutate
    model, _ = backend(server, prime_prefix=False)
    with pytest.raises(ValueError):
        model.binary('c',['q1','q2'])


def test_token_budget_is_checked_without_truncation_or_scoring_request():
    model, server = backend(max_context_tokens=10)
    with pytest.raises(ValueError, match='no truncation'):
        model.binary('long context', ['question'])
    assert not completions(server)


def test_empty_binary_does_not_call_server_and_missing_cache_usage_stays_unknown():
    model, server = backend()
    out = model.binary('context', [])
    assert out.probabilities == []
    assert out.usage['requests'] == 0
    server.response_mutator = lambda body: body['usage'].pop('prompt_tokens_details')
    assert model.binary('c',['q']).usage['cached_tokens'] is None


def test_http_failure_has_no_model_fallback():
    model, _ = backend()
    model.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    with pytest.raises(RuntimeError, match='no HF fallback'):
        model.binary('c',['q'])


def test_server_root_accepts_openai_suffix_and_rejects_embedded_credentials():
    assert server_root('http://127.0.0.1:8000/v1/') == 'http://127.0.0.1:8000'
    with pytest.raises(ValueError):
        server_root('http://name:password@localhost:8000')


def test_prompt_token_ids_match_isolated_hf_reference_without_loading_weights():
    pytest.importorskip("torch")
    from decision_rag.backends.hf import HFLogitBackend
    reference = object.__new__(HFLogitBackend)
    reference.tokenizer = TinyTokenizer()
    model, _ = backend()
    questions = ["Is candidate 1 relevant?", "Is candidate 2 relevant?"]
    layout = reference._binary_layout("Both candidate descriptions", questions)
    actual = model._binary_prompts("Both candidate descriptions", questions)
    prefix = layout.input_ids[:layout.context_length]
    for index, ids in enumerate(actual):
        branch = [token for token, owner in zip(layout.input_ids, layout.owners) if owner == index]
        assert ids == prefix + branch


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logprob_parser_rejected(value):
    body = {"choices": [{"index": 0, "logprobs": {"top_logprobs": [
        {"token_id:10": value, "token_id:11": math.log(.7)}
    ]}}]}
    with pytest.raises(ValueError, match="invalid label"):
        VLLMDecisionBackend._parse_choices(body, 1, [10,11])
