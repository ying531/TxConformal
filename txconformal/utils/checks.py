def should_use_uniform_weights(provider) -> bool:
    has_embeddings = (provider.E_calib is not None) or (provider.embed_fn is not None)
    return not has_embeddings