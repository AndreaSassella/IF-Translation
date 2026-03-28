from __future__ import annotations


def ensure_nltk_resource(resource_name: str, lookup_path: str) -> None:
    try:
        import nltk
        nltk.data.find(lookup_path)
        return
    except LookupError:
        pass
    except ImportError as exc:
        raise ImportError(
            "NLTK is required for the IFEval evaluator. Install requirements.txt before running."
        ) from exc

    import nltk

    ok = nltk.download(resource_name, quiet=True)
    if not ok:
        raise RuntimeError(
            f"Failed to download NLTK resource '{resource_name}'. "
            "Please run `python -c \"import nltk; nltk.download('"
            f"{resource_name}')\"` in an environment with internet access."
        )

    try:
        nltk.data.find(lookup_path)
    except LookupError as exc:
        raise RuntimeError(
            f"NLTK resource '{resource_name}' still not available after download."
        ) from exc


def ensure_ifeval_nltk_resources() -> None:
    ensure_nltk_resource("punkt", "tokenizers/punkt")
    ensure_nltk_resource("punkt_tab", "tokenizers/punkt_tab/english")
