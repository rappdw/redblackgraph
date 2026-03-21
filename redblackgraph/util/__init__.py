def __getattr__(name):
    if name in ('RedBlackGraphWriter', 'RelationshipFileReader'):
        from .relationship_file_io import RedBlackGraphWriter, RelationshipFileReader
        return locals()[name]
    if name == 'RbgGraphBuilder':
        from .graph_builder import RbgGraphBuilder
        return RbgGraphBuilder
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")
