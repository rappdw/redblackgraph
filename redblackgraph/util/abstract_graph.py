"""Abstract graph-builder interfaces for relationship-database ingestion.

Vendored from rappdw/fs-crawler (``fscrawler/util/abstract_graph.py``) at commit
fae46e76e3849924d6d1fb96f343f9a00a21ca98. fs-crawler is deprecated as a
dependency; only the graph-consumption side it provided lives on here. The
original FamilySearch crawler that produces the relationship databases remains
external. fs-crawler is licensed GPL-3.0-or-later, compatible with this
project's AGPLv3+.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from tqdm import tqdm


class VertexInfo(ABC):
    @abstractmethod
    def get_vertex_key(self) -> Dict[int,Tuple[str,str]]:
        """
        Get the "vertex key", a dictionary keyed by the vertex id (int) with values
        that are tuples of (external id, string designation)
        """
        pass


class AbstractGraphBuilder(ABC):
    def __init__(self, sparse_threshold: int = 1000, enable_status: bool = True,
                 status_interval: int = 10000, use_tqdm: bool = True):
        self.sparse_threshold = sparse_threshold
        self.enable_status = enable_status
        self.status_interval = status_interval
        self.use_tqdm = use_tqdm
        self.logger = logging.getLogger(self.__class__.__name__)
        self._vertex_count = 0
        self._edge_count = 0
        self._vertices_added = 0
        self._edges_added = 0
        self._vertex_pbar: Optional[tqdm] = None
        self._edge_pbar: Optional[tqdm] = None

    @abstractmethod
    def init_builder(self, vertex_count: int, edge_count: int):
        """
        Initialize the builder
        """
        pass

    def _init_status(self, vertex_count: int, edge_count: int):
        """
        Initialize status tracking. Call this from init_builder implementations.
        """
        # Close any existing progress bars
        self._close_progress_bars()

        self._vertex_count = vertex_count
        self._edge_count = edge_count
        self._vertices_added = 0
        self._edges_added = 0

        if self.enable_status:
            if self.use_tqdm:
                # Create progress bars for vertices and edges
                self._vertex_pbar = tqdm(total=vertex_count, desc="Vertices", unit="vertex", leave=True)
                self._edge_pbar = tqdm(total=edge_count, desc="Edges", unit="edge", leave=True)
            else:
                self.logger.info(f"Initializing graph builder: {vertex_count} vertices, {edge_count} edges")

    @abstractmethod
    def get_ordering(self) -> Sequence[int]:
        pass

    @abstractmethod
    def add_vertex(self, vertex_id: int, color: int):
        pass

    def _track_vertex(self):
        """
        Track vertex addition for status updates. Call this from add_vertex implementations.
        """
        self._vertices_added += 1
        if self.enable_status:
            if self.use_tqdm and self._vertex_pbar:
                self._vertex_pbar.update(1)
            elif self._vertices_added % self.status_interval == 0:
                progress = (self._vertices_added / self._vertex_count * 100) if self._vertex_count > 0 else 0
                self.logger.info(f"Added {self._vertices_added:,}/{self._vertex_count:,} vertices ({progress:.1f}%)")

    @abstractmethod
    def add_edge(self, source_id: int, dest_id: int):
        pass

    def _track_edge(self):
        """
        Track edge addition for status updates. Call this from add_edge implementations.
        """
        self._edges_added += 1
        if self.enable_status:
            if self.use_tqdm and self._edge_pbar:
                self._edge_pbar.update(1)
            elif self._edges_added % self.status_interval == 0:
                progress = (self._edges_added / self._edge_count * 100) if self._edge_count > 0 else 0
                self.logger.info(f"Added {self._edges_added:,}/{self._edge_count:,} edges ({progress:.1f}%)")

    @abstractmethod
    def add_gender(self, vertex_id: int, color: int):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def save_cache(self, graph, cache_path: Path, metadata: dict):
        """Save the built graph to a cache file.

        Args:
            graph: The graph object to cache (returned from build())
            cache_path: Path where the cache file should be saved
            metadata: Dictionary containing metadata like {'hops': int or None}
        """
        pass

    @abstractmethod
    def load_cache(self, cache_path: Path, expected_metadata: dict):
        """Load a graph from a cache file.

        Args:
            cache_path: Path to the cache file
            expected_metadata: Dictionary with expected metadata like {'hops': int or None}

        Returns:
            The loaded graph object

        Raises:
            ValueError: If metadata doesn't match expected values
        """
        pass

    @abstractmethod
    def is_cache_valid(self, cache_path: Path, reference_path: Path) -> bool:
        """Check if a cache file exists and is valid.

        Args:
            cache_path: Base path for the cache file
            reference_path: Path to reference file (e.g., database) to compare modification times

        Returns:
            True if cache exists and is newer than reference_path, False otherwise
        """
        pass

    def _close_progress_bars(self):
        """
        Close any open progress bars.
        """
        if self._vertex_pbar:
            self._vertex_pbar.close()
            self._vertex_pbar = None
        if self._edge_pbar:
            self._edge_pbar.close()
            self._edge_pbar = None

    def _build_status(self):
        """
        Log final build status and close progress bars. Call this from build implementations.
        """
        # Close progress bars before final status
        self._close_progress_bars()

        if self.enable_status:
            self.logger.info(f"Built graph with {self._vertices_added:,} vertices and {self._edges_added:,} edges")
