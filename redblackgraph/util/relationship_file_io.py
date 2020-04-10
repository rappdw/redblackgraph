"""
Relationship File IO is a collection of utilities to parse an input csv file
into a RB Graph.

It takes 2 input files. One that defines the vertices with 3 columns: external_id, gender, name.
One that defines the edges with 2 columns: source_external_id, destination_external_id
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import csv
import itertools
import logging
from typing import Optional, Dict, Tuple, List

import numpy as np
import redblackgraph as rb
import xlsxwriter

logger = logging.getLogger(__name__)

ROTATE_90 = 'rotate-90'
MAX_COLUMNS_EXCEL = 16384


class PersonIdentifier:
    def __init__(self):
        self.next_id = 0
        self.person_dictionary = {}

    def set_order(self, order):
        self.order = order

    def add_person(self, external_id: str, name: str) -> Optional[int]:
        """
        Given an external id and name, add the person into the dictionary if it's not already into the dictionary

        If the person tuple is empty, returns None
        :param person: tuple[ given name, surname ]
        """
        if external_id:
            if not external_id in self.person_dictionary:
                internal_id = self.next_id
                self.person_dictionary[external_id] = (internal_id, name)
                self.next_id += 1
                return internal_id
        return None

    def get_person_id(self, id: str) -> Optional[int]:
        """
        Given an external id, get the rbg id
        :return: person_id or None if person hasn't been added
        """
        if id and id in self.person_dictionary:
            return self.person_dictionary[id][0]
        return None

    def get_person_name(self, id: str) -> Optional[str]:
        """
        Given an external id, get the name used when added
        :return: person name or None if person hasn't been added
        """
        if id and id in self.person_dictionary:
            return self.person_dictionary[id][1]
        return None
    
    def get_vertex_key(self, order_lookup):
        return {order_lookup[person_tuple[0]]: (external_id, person_tuple[1]) for external_id, person_tuple in self.person_dictionary.items()}


class GraphBuilder:
    def __init__(self):
        self.count = 0 # count of how many data cells there should be (one for each edge)
        self.rbg_dictionary = {}
        self.vertex_color = None

    def max_vertices(self, count):
        if self.vertex_color is None:
            self.vertex_color = [0] * count
        elif count != len(self.vertex_color):
            if count < len(self.vertex_color):
                self.vertex_color = self.vertex_color[:count]
            else:
                raise ValueError(f"Unexpected condition. Setting max vertices, {count}, greater than initial estimate, {len(self.vertex_color)}.")

    def add_vertex(self, vertex_id:int, color:int, red_vertex_id:int=None, black_vertex_id:int=None):
        """
        Add a vertex to the graph (optionally include immediate ancestry vertices
        :param vertex_id: id of the vertex to add 
        :param color: color of the vertex to add
        :param red_vertex_id: (Optional) red vertex which vertex_id is connected to
        :param black_vertex_id: (Optional) black vertex which vertex_id is connected to
        :return: None
        """
        if vertex_id is not None:
            self.vertex_color[vertex_id] = color
            self.count += 1
            if red_vertex_id is not None:
                if vertex_id not in self.rbg_dictionary:
                    self.rbg_dictionary[vertex_id] = {}
                self.rbg_dictionary[vertex_id][red_vertex_id] = 2
                self.vertex_color[red_vertex_id] = -1
                self.count += 2
            if black_vertex_id is not None:
                if vertex_id not in self.rbg_dictionary:
                    self.rbg_dictionary[vertex_id] = {}
                self.rbg_dictionary[vertex_id][black_vertex_id] = 3
                self.vertex_color[black_vertex_id] = 1
                self.count += 2

    def add_edge(self, source_vertex: int, destination_vertex: int):
        if source_vertex not in self.rbg_dictionary:
            self.rbg_dictionary[source_vertex] = {}
        self.rbg_dictionary[source_vertex][destination_vertex] = 2 \
            if self.vertex_color[destination_vertex] == -1 else 3
        self.count += 1

    def _topological_visit(self, v, color, order):
        """Run iterative DFS from node V"""
        total = 0
        stack = [v]  # create stack with starting vertex, stack to replace recursion with loop
        while stack:  # while stack is not empty
            v = stack[-1]  # peek top of stack
            if color[v]:  # if already seen
                v = stack.pop()  # done with this node, pop it from stack
                if color[v] == 1:  # if GRAY, finish this node
                    order.append(v)
                    color[v] = 2  # BLACK, done
            else:  # seen for first time
                color[v] = 1  # GRAY: discovered
                total += 1
                if v in self.rbg_dictionary:
                    for w in self.rbg_dictionary[v].keys():  # for all neighbors
                        if not color[w]:
                            stack.append(w)
        return total

    def _topological_sort(self):
        """Run DFS on graph"""
        N = len(self.vertex_color)
        color = [0] * N
        order = []  # stack to hold topological ordering of graph
        for v in range(N):
            if not color[v]:
                self._topological_visit(v, color, order)
        return order[::-1]


    def _gen_graph_ordering(self):
        """
        Topologically order the graph. This ordering will then apply to both thee vertices of the
        graph as well as to the vertex keys
        :return: None
        """

        # order is the list that indicates the ordering of the vertices
        # e.g. [2, 0, 1] indicates that the first vertex is id 2, the second is id 0 and the last is id 1
        self.order = self._topological_sort()
        intermediate = {id: idx for idx, id in enumerate(self.order)}
        # order lookup is a list that indicates what the ordinal value is for a given vertex
        # e.g. [1, 2, 0] indicates that id 0 is in the 2nd position, id 1 is in the last position and id 3 is in
        # the first position
        self.order_lookup = [intermediate[idx] for idx in range(len(self.order))]


    def generate_graph(self):
        self._gen_graph_ordering()

        N = len(self.vertex_color)

        # create a rb_matrix (sparse)
        data = np.zeros(self.count, dtype=np.int32)
        indices = np.zeros(self.count, dtype=np.int32)
        indptr = np.zeros(N + 1, dtype=np.int32)

        ind_idx = 0
        idx = 0
        for i in range(N):
            vertex_id = self.order[i]
            vertex_color = self.vertex_color[vertex_id]
            indptr[ind_idx] = idx
            ind_idx += 1
            data[idx] = vertex_color
            indices[idx] = i
            idx += 1
            if vertex_id in self.rbg_dictionary:
                for vertex, relationship in self.rbg_dictionary[vertex_id].items():
                    data[idx] = relationship
                    indices[idx] = self.order_lookup[vertex]
                    idx += 1
        indptr[ind_idx] = idx

        return rb.sparse.rb_matrix((data, indices, indptr))


class VertexInfo(ABC):
    @abstractmethod
    def get_vertex_key(self) -> Dict[int,Tuple[str,str]]:
        """
        Get the "vertex key", a dictionary keyed by the vertex id (int) with values
        that are tuples of (external id, string designation)
        """
        pass


class RelationshipFileReader(VertexInfo):
    def __init__(self, persons_file, relationships_file, hop:int, filter:List[str], invalid_edges_file=None,
                 invalid_filter:List[str]=None, ignore_file=None):
        self.persons_file = persons_file
        self.relationships_file = relationships_file
        self.person_identifier = PersonIdentifier()
        self.graph_builder = GraphBuilder()
        self.hop = hop
        self.filter = filter
        self.invalid_edges_file = invalid_edges_file
        self.invalid_filter = invalid_filter if invalid_filter else []
        self.ignore_file = ignore_file

    def read(self):
        vertex_exclusions = set()
        vertex_count = 0 # from this first loop, the count is an estimate as we could have some "island" vertices still
        hop_limit_count = 0
        no_color_count = 0
        no_edges_count = 0
        duplicate_edge_count = 0

        # read through vertex file first and build a set of vertices to exclude. A vertex
        # can be excluded if it has no color or if it is outside the range of the hop limit.
        with open(self.persons_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                external_id = row[0]
                hop = int(row[3])
                color = row[1]
                if hop > self.hop:
                    hop_limit_count += 1
                    vertex_exclusions.add(external_id)
                elif color == '':
                    no_color_count += 1
                    vertex_exclusions.add(external_id)
                else:
                    vertex_count += 1

        self.graph_builder.max_vertices(vertex_count)

        # read through the edges file and identify all vertices that have edges
        # excluding edges that cross the hop limit frontier
        linked = set()
        with open(self.relationships_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                if row[2] in self.filter:
                    if row[0] not in vertex_exclusions and row[1] not in vertex_exclusions:
                        linked.add(row[0])
                        linked.add(row[1])

        # if we are also reading the invalid file, do so here
        if self.invalid_edges_file:
            with open(self.invalid_edges_file, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0].startswith("#"):
                        continue
                    if row[2] in self.filter and row[3] in self.invalid_filter:
                        if row[0] not in vertex_exclusions and row[1] not in vertex_exclusions:
                            linked.add(row[0])
                            linked.add(row[1])

        # read through the vertex file and for any vertices that are
        # linked and inside of the hop frontier, get a unique monotonically increasing
        # vertex id and add it as a vertex to the graph
        vertex_count = 0 # reset vertex count
        with open(self.persons_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                external_id = row[0]
                color = row[1]
                name = row[2]
                if external_id in vertex_exclusions:
                    pass
                else:
                    color = int(color)
                    vertex_id = self.person_identifier.add_person(external_id, name)
                    self.graph_builder.add_vertex(vertex_id, color)
                    vertex_count += 1
                    if external_id not in linked:
                        no_edges_count += 1

        self.graph_builder.max_vertices(vertex_count)

        # if ignore file exists, read it and build dictionary of src -> dest edges that should
        # be ignored
        ignore:Dict[str, set] = defaultdict(lambda : set())
        if self.ignore_file:
            with open(self.ignore_file, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0].startswith("#"):
                        continue
                    ignore[row[0]].add(row[1])

        # read through the edges file. if the edge type is in the accepted filter,
        # add the edge to the graph
        with open(self.relationships_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                if row[2] in self.filter:
                    source_vertex = self.person_identifier.get_person_id(row[0])
                    destination_vertex = self.person_identifier.get_person_id(row[1])
                    if not source_vertex is None and not destination_vertex is None:
                        if row[0] in ignore and row[1] in ignore[row[0]]:
                            continue
                        self.graph_builder.add_edge(source_vertex, destination_vertex)

        # if we are also reading the invalid file, do so here
        if self.invalid_edges_file:
            with open(self.invalid_edges_file, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0].startswith("#"):
                        continue
                    if row[2] in self.filter and row[3] != 'frontier':
                        source_vertex = self.person_identifier.get_person_id(row[0])
                        destination_vertex = self.person_identifier.get_person_id(row[1])
                        if not source_vertex is None and not destination_vertex is None:
                            if row[0] in ignore and row[1] in ignore[row[0]]:
                                continue
                            duplicate_edge_count += 1
                            self.graph_builder.add_edge(source_vertex, destination_vertex)

        logger.info(f"{vertex_count:,} vertices in graph. {hop_limit_count:,} vetices were"
                    f" outside the hop limit. {no_edges_count:,} have no edges."
                    f" {no_color_count:,} were removed due to no color."
                    f" {duplicate_edge_count:,} duplicate parent edges exist."
                    f" {len(ignore):,} were ignored.")
        return self.graph_builder.generate_graph()

    def get_vertex_key(self):
        return self.person_identifier.get_vertex_key(self.graph_builder.order_lookup)

    def get_person_id(self, person):
        return self.person_identifier.get_person_id(person)


class RedBlackGraphWriter:
    def __init__(self, vertex_info:VertexInfo=None):
        ''':parameter vertex_info - class providing information on the vertices'''
        self.vertex_key = vertex_info.get_vertex_key() if vertex_info else None

    @staticmethod
    def _open_workbook(output_file):
        workbook = xlsxwriter.Workbook(output_file)
        formats = {
            ROTATE_90: workbook.add_format({'rotation': 90})
        }
        return (workbook, formats)

    @staticmethod
    def _calc_width(len_of_max_string):
        return max(len_of_max_string + 0.83, 2.67)

    def write(self, R, output_file='/tmp/rbg.csv', key_permutation=None):
        workbook, formats = self._open_workbook(output_file)
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(hide_unused_rows=True)
        n = R.shape[0] if isinstance(R, rb.sparse.rb_matrix) else len(R)
        if n > MAX_COLUMNS_EXCEL:
            logging.error("Graph exceeds max size allowable by Excel")
        if key_permutation is None:
            key_permutation = np.arange(n)

        max_key = 0
        max_np = 0

        worksheet.write(0, 0, ' ')
        for idx in range(n):
            if self.vertex_key:
                vertex_key = self.vertex_key[key_permutation[idx]]
                cell_data = f"{vertex_key[0]} - {vertex_key[1]}"
            else:
                cell_data = f"{idx}"
            max_key = max(max_key, len(cell_data))
            worksheet.write(0, idx + 1, cell_data, formats[ROTATE_90])
            worksheet.write(idx + 1, 0, cell_data)

        if isinstance(R, rb.sparse.rb_matrix):
            for i, j in zip(*R.nonzero()):
                cell_data = R[i, j]
                max_np = max(max_np, cell_data)
                worksheet.write(i + 1, j + 1, cell_data)
        else:
            for i, j in itertools.product(range(n), repeat=2):
                cell_data = R[i][j]
                if cell_data != 0:
                    max_np = max(max_np, cell_data)
                    worksheet.write(i + 1, j + 1, cell_data)
        column_width = self._calc_width(len(f"{max_np}"))
        worksheet.freeze_panes(1, 1)
        worksheet.set_column(0, 0, self._calc_width(max_key))
        worksheet.set_column(1, n, column_width)
        worksheet.set_column(n + 1, MAX_COLUMNS_EXCEL - 1, None, None, {'hidden': True})
        workbook.close()

    def append_vertex_key(self, key):
        self.vertex_key[len(self.vertex_key)] = key
