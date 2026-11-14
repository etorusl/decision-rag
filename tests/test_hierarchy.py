import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from decision_rag import hierarchy
from decision_rag.schema import Corpus, Node


@pytest.fixture
def flat_corpus(tmp_path):
    corpus = Corpus([
        Node("root", "Wikipedia", "root"),
        Node("b", "Birds", "page", "root", "Birds", 1),
        Node("b1", "Birds can fly. Most birds lay eggs.", "passage", "b"),
        Node("a", "Aircraft", "page", "root", "Aircraft", 0),
        Node("a1", "Aircraft are vehicles. They travel in the air.", "passage", "a"),
        Node("c", "Cats", "page", "root", "Cats", 2),
        Node("c1", "Cats are mammals.", "passage", "c"),
    ])
    path = tmp_path / "flat.jsonl"
    corpus.save(path)
    return path


def install_fake_models(monkeypatch):
    def encode(texts, model_name, device, batch_size, revision):
        assert texts == ["query: Aircraft\nAircraft are vehicles.",
                         "query: Birds\nBirds can fly.", "query: Cats\nCats are mammals."]
        return np.array([[1., 0.], [.9, .1], [0., 1.]], dtype=np.float32), {"model": "fake-e5"}
    def fit(embeddings, clusters, seed, n_init, max_iter, *, device):
        assert clusters == 2 and seed == 17
        assert device == 'cpu'
        return np.array([0, 0, 1]), np.array([[1., 0.], [0., 1.]]), {"algorithm": "fake"}
    def summarize(titles, model_name, device, revision, max_input_tokens):
        assert titles == {"0": ["Aircraft", "Birds"], "1": ["Cats"]}
        return {"0": "Flying vehicles and birds.", "1": "Domestic cats."}, {"model": "fake-qwen"}
    monkeypatch.setattr(hierarchy, "encode_pages", encode)
    monkeypatch.setattr(hierarchy, "fit_clusters", fit)
    monkeypatch.setattr(hierarchy, "summarize_clusters", summarize)


def test_builder_keeps_evidence_ids_and_writes_bound_artifacts(flat_corpus, tmp_path, monkeypatch):
    install_fake_models(monkeypatch)
    output = tmp_path / "clustered.jsonl"
    metadata = hierarchy.cluster_corpus(flat_corpus, output, clusters=2, seed=17, device="cpu")
    corpus = Corpus.load(output)
    assert [n.kind for n in corpus.children("root")] == ["group", "group"]
    assert [p.id for p in corpus.passages] == ["a1", "b1", "c1"]
    assert corpus.nodes["a"].parent_id == corpus.nodes["b"].parent_id != corpus.nodes["c"].parent_id
    assert corpus.nodes["a1"].text == "Aircraft are vehicles. They travel in the air."
    assert metadata["cluster_sizes"] == {"min": 1, "median": 1.5, "max": 2}
    assert metadata["output_sha256"] == hierarchy.file_sha256(output)
    assert metadata["embedding"]["artifact_sha256"]
    artifacts = tmp_path / "clustered.hierarchy"
    assignments = json.loads((artifacts / "assignments.json").read_text())
    summaries = json.loads((artifacts / "summaries.json").read_text())
    assert assignments["page_ids"] == ["a", "b", "c"]
    assert assignments["assignment_sha256"] == summaries["assignment_sha256"]
    assert (artifacts / "metadata.json").exists()


def test_reused_assignments_and_summaries_need_no_models(flat_corpus, tmp_path, monkeypatch):
    install_fake_models(monkeypatch)
    hierarchy.cluster_corpus(flat_corpus, tmp_path / "first.jsonl", clusters=2, seed=17, device='cpu')
    def fail(*args, **kwargs):
        raise AssertionError("Reusing bound artifacts must not load a model")
    monkeypatch.setattr(hierarchy, "encode_pages", fail)
    monkeypatch.setattr(hierarchy, "fit_clusters", fail)
    monkeypatch.setattr(hierarchy, "summarize_clusters", fail)
    hierarchy.cluster_corpus(flat_corpus, tmp_path / "second.jsonl", clusters=2,
                             assignments_path=tmp_path / "first.hierarchy/assignments.json",
                             summaries_path=tmp_path / "first.hierarchy/summaries.json")
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()


def test_wrong_assignment_identity_is_rejected(flat_corpus, tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"labels": [0, 0, 1], "page_ids": ["a", "b", "c"], "cluster_count": 2}))
    with pytest.raises(ValueError, match="do not match"):
        hierarchy.cluster_corpus(flat_corpus, tmp_path / "result.jsonl", clusters=2, assignments_path=path)


def test_wrong_summary_binding_is_rejected(flat_corpus, tmp_path, monkeypatch):
    install_fake_models(monkeypatch)
    supplied = tmp_path / "wrong-summaries.json"
    supplied.write_text(json.dumps({"summaries": {"0": "Planes.", "1": "Cats."}, "assignment_sha256": "wrong"}))
    with pytest.raises(ValueError, match="do not match"):
        hierarchy.cluster_corpus(flat_corpus, tmp_path / "result.jsonl", clusters=2, seed=17,
                                 summaries_path=supplied, device='cpu')


def test_title_typicality_and_cap():
    pages = [Node(str(i), "", "page", "root", f"Title {i}") for i in range(400)]
    embeddings = np.arange(400, dtype=np.float32).reshape(-1, 1)
    titles = hierarchy.typical_titles(pages, embeddings, np.zeros(400, dtype=int), np.array([[200.]]), device='cpu')
    assert len(titles["0"]) == 300
    assert titles["0"][0] == "Title 200"
    assert set(titles["0"][1:3]) == {"Title 199", "Title 201"}


@pytest.mark.parametrize("labels", [np.array([0, 0, 0]), np.array([0, 1]), np.array([0., 0., 1.])])
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        hierarchy.validate_labels(labels, pages=3, clusters=2)


def test_summary_shortening_is_explicit_and_preserves_limit():
    summary, clipped = hierarchy._compact_summary(" ".join(["word"] * 35))
    assert clipped and len(summary.split()) == 30
    assert hierarchy._compact_summary("<think>hidden thoughts</think>Cats and dogs. Further sentence.") == (
        "Cats and dogs.", False)
    with pytest.raises(ValueError, match="empty"):
        hierarchy._compact_summary("   ")
    with pytest.raises(ValueError, match="unfinished"):
        hierarchy._compact_summary("<think>still thinking")


def test_input_and_output_cannot_be_same(flat_corpus):
    with pytest.raises(ValueError, match="must differ"):
        hierarchy.cluster_corpus(flat_corpus, flat_corpus, clusters=2)


def test_grouped_corpus_is_rejected():
    corpus = Corpus([Node("root", "", "root"), Node("group", "", "group", "root"),
                     Node("page", "Title", "page", "group"), Node("passage", "text", "passage", "page")])
    with pytest.raises(ValueError, match="unclustered"):
        hierarchy.page_features(corpus)


def test_excess_clusters_rejected_without_model_load(flat_corpus, tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        hierarchy.cluster_corpus(flat_corpus, tmp_path / "result.jsonl", clusters=4)


def test_hash_bound_embeddings_reuse(flat_corpus, tmp_path, monkeypatch):
    install_fake_models(monkeypatch)
    hierarchy.cluster_corpus(flat_corpus, tmp_path / "first.jsonl", clusters=2, seed=17, device='cpu')
    monkeypatch.setattr(hierarchy, "encode_pages", lambda *args: pytest.fail("Must reuse supplied embeddings"))
    second = hierarchy.cluster_corpus(flat_corpus, tmp_path / "second.jsonl", clusters=2, seed=17,
                                      embeddings_path=tmp_path / "first.hierarchy/embeddings.npy", device='cpu')
    assert "reused_embeddings" in second["embedding"]


def test_real_kmeans_is_seeded_and_separates_clear_clusters():
    pytest.importorskip("sklearn")
    vectors = np.array([[1., 0.], [.99, .01], [0., 1.], [.01, .99]], dtype=np.float32)
    first, centers, metadata = hierarchy.fit_clusters(vectors, 2, 42, n_init=2, device='cpu')
    second, _, _ = hierarchy.fit_clusters(vectors, 2, 42, n_init=2, device='cpu')
    assert np.array_equal(first, second)
    assert first[0] == first[1] and first[2] == first[3] and first[0] != first[2]
    assert centers.shape == (2, 2) and metadata["seed"] == 42
    assert metadata['backend'] == 'scikit-learn' and metadata['device'] == 'cpu'


def test_gpu_kmeans_routes_directly_to_cuml_and_synchronizes(monkeypatch):
    calls = []
    class Device:
        def __init__(self, index):
            calls.append(('device', index))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append(('exit',))
        def synchronize(self):
            calls.append(('sync',))
    class KMeans:
        def __init__(self, **kwargs):
            calls.append(('kmeans', kwargs))
        def fit(self, vectors):
            assert vectors.dtype == np.float32
            self.labels_ = np.array([0, 0, 1, 1])
            self.cluster_centers_ = np.array([[1, 0], [0, 1]], dtype=np.float32)
            self.inertia_ = .1
    monkeypatch.setattr(hierarchy, 'resolve_search_device', lambda device: 'cuda:0')
    pool = SimpleNamespace(free_all_blocks=lambda: calls.append(('free',)))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(Device=Device),
                         get_default_memory_pool=lambda: pool, get_default_pinned_memory_pool=lambda: pool))
    monkeypatch.setitem(sys.modules, 'cuml.cluster', SimpleNamespace(KMeans=KMeans))
    vectors = np.array([[1., 0.], [.99, .01], [0., 1.], [.01, .99]])
    labels, centers, info = hierarchy.fit_clusters(vectors, 2, 42, n_init=3, max_iter=10)
    assert calls[0] == ('device', 0)
    assert calls[1] == ('kmeans', {'n_clusters': 2, 'random_state': 42, 'n_init': 3,
                                  'max_iter': 10, 'init': 'scalable-k-means++', 'output_type': 'numpy'})
    assert calls.index(('sync',)) < calls.index(('free',))
    assert info['backend'] == 'cuML' and info['device'] == 'cuda:0'
    assert info['precision'] == 'float32' and info['iterations'] is None
    assert centers.shape == (2, 2) and labels.tolist() == [0, 0, 1, 1]


def test_missing_cuml_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(hierarchy, 'resolve_search_device', lambda device: 'cuda:0')
    monkeypatch.setitem(sys.modules, 'cupy', None)
    with pytest.raises(RuntimeError, match='gpu-clustering'):
        hierarchy.fit_clusters(np.eye(2, dtype=np.float32), 2, 42)


def test_centroid_recompute_and_title_ties_cpu():
    vectors = np.array([[0., 0.], [2., 0.], [10., 0.]], dtype=np.float32)
    labels = np.array([0, 0, 1])
    centers = hierarchy.recompute_centroids(vectors, labels, 2, device='cpu')
    np.testing.assert_allclose(centers, [[1, 0], [10, 0]])
    pages = [Node('z', '', 'page', 'root', 'Z'), Node('a', '', 'page', 'root', 'A'),
             Node('b', '', 'page', 'root', 'B')]
    assert hierarchy.typical_titles(pages, vectors, labels, centers, device='cpu') == {
        '0': ['A', 'Z'], '1': ['B']}


def test_gpu_centroids_and_title_order_match_cpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for GPU preprocessing parity')
    vectors = np.array([[0., 0.], [2., 0.], [10., 0.]], dtype=np.float32)
    labels = np.array([0, 0, 1])
    pages = [Node('z', '', 'page', 'root', 'Z'), Node('a', '', 'page', 'root', 'A'),
             Node('b', '', 'page', 'root', 'B')]
    cpu_centers = hierarchy.recompute_centroids(vectors, labels, 2, device='cpu')
    gpu_centers = hierarchy.recompute_centroids(vectors, labels, 2)
    np.testing.assert_allclose(gpu_centers, cpu_centers, atol=1e-6)
    assert hierarchy.typical_titles(pages, vectors, labels, gpu_centers) == hierarchy.typical_titles(
        pages, vectors, labels, cpu_centers, device='cpu')


def test_real_gpu_cuml_kmeans_separates_clear_clusters():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for cuML KMeans integration')
    pytest.importorskip('cuml.cluster')
    vectors = np.array([[1., 0.], [.99, .01], [.98, .02],
                        [0., 1.], [.01, .99], [.02, .98]], dtype=np.float32)
    labels, centers, metadata = hierarchy.fit_clusters(vectors, 2, 42, n_init=2, max_iter=50)
    assert labels.shape == (6,) and np.issubdtype(labels.dtype, np.integer)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert centers.shape == (2, 2) and np.isfinite(centers).all()
    assert metadata['backend'] == 'cuML'
    assert metadata['device'].startswith('cuda:')
    assert metadata['init'] == 'scalable-k-means++' and metadata['precision'] == 'float32'
    assert np.isfinite(metadata['inertia']) and metadata['inertia'] >= 0
