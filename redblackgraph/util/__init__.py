from .relationship_file_io import RedBlackGraphWriter, RelationshipFileReader

# Lazy import to avoid issues when running graph_builder as __main__
def __getattr__(name):
    if name == 'RbgGraphBuilder':
        from .graph_builder import RbgGraphBuilder
        return RbgGraphBuilder
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
