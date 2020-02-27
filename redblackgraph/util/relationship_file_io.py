"""
Relationship File IO is a collection of utilities to parse an input csv file
into a RB Graph.

It takes 2 input files. One that defines the vertices with 3 columns: external_id, gender, name.
One that defines the edges with 2 columns: source_external_id, destination_external_id
"""
from abc import ABC, abstractmethod
import csv
import logging
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

import numpy as np
import redblackgraph as rb
import xlsxwriter

logger = logging.getLogger(__name__)

ROTATE_90 = 'rotate-90'
MAX_COLUMNS_EXCEL = 16384


class PersonIdentifier:
    def __init__(self):
        self.person_dictionary = {'p_id': (0,)}

    def add_person(self, external_id: str, name: str) -> Optional[int]:
        """
        Given an external id and name, add the person into the dictionary if it's not already into the dictionary

        If the person tuple is empty, returns None
        :param person: tuple[ given name, surname ]
        """
        if external_id:
            if not external_id in self.person_dictionary:
                internal_id = self.person_dictionary['p_id'][0]
                self.person_dictionary[external_id] = (internal_id, name)
                next_id = internal_id + 1
                self.person_dictionary['p_id'] = (next_id,)
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
    
    def get_vertex_key(self):
        return {person_tuple[0]: (external_id, person_tuple[1]) for external_id, person_tuple in self.person_dictionary.items() if external_id != 'p_id'}


class GraphBuilder:
    def __init__(self):
        self.rbg_dictionary = defaultdict(lambda: {})

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
            self.rbg_dictionary[vertex_id][vertex_id] = color
            if not red_vertex_id is None:
                self.rbg_dictionary[vertex_id][red_vertex_id] = 2
                self.rbg_dictionary[red_vertex_id][red_vertex_id] = -1
            if not black_vertex_id is None:
                self.rbg_dictionary[vertex_id][black_vertex_id] = 3
                self.rbg_dictionary[black_vertex_id][black_vertex_id] = 1

    def add_edge(self, source_vertex: int, destination_vertex: int):
        self.rbg_dictionary[source_vertex][destination_vertex] = 2 \
            if self.rbg_dictionary[destination_vertex][destination_vertex] == -1 else 3

    def generate_graph(self):
        # create an numpy array with the correct shape and load it with the tuples
        m = len(self.rbg_dictionary)
        R = np.zeros((m, m), dtype=np.int32).view(rb.array)
        for i in range(m):
            row = self.rbg_dictionary[i]
            for key in row.keys():
                R[i][key] = row[key]
        return R


class VertexInfo(ABC):
    @abstractmethod
    def get_vertex_key(self) -> Dict[int,Tuple[str,str]]:
        """
        Get the "vertex key", a dictionary keyed by the vertex id (int) with values
        that are tuples of (external id, string designation)
        """
        pass


class RelationshipFileReader(VertexInfo):
    def __init__(self, persons_file, relationships_file, hop:int, filter:List[str]):
        self.persons_file = persons_file
        self.relationships_file = relationships_file
        self.person_identifier = PersonIdentifier()
        self.graph_builder = GraphBuilder()
        self.hop = hop
        self.filter = filter

    def __call__(self, *args, **kwargs):
        linked = set()
        vertex_count = 0
        with open(self.relationships_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                if row[2] in self.filter:
                    linked.add(row[0])
                    linked.add(row[1])
        with open(self.persons_file, "r") as csvfile:
            skipped_count = 0
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                external_id = row[0]
                hop = int(row[3])
                color = row[1]
                if external_id in linked:
                    name = row[2]
                    if hop <= self.hop:
                        if color != '':
                            color = int(color)
                            vertex_id = self.person_identifier.add_person(external_id, name)
                            self.graph_builder.add_vertex(vertex_id, color)
                            vertex_count += 1
                        else:
                            skipped_count += 1
                else:
                    if hop <= self.hop:
                        skipped_count += 1
        with open(self.relationships_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#"):
                    continue
                if row[2] in self.filter:
                    source_vertex = self.person_identifier.get_person_id(row[0])
                    destination_vertex = self.person_identifier.get_person_id(row[1])
                    if not source_vertex is None and not destination_vertex is None:
                        self.graph_builder.add_edge(source_vertex, destination_vertex)
        logger.info(f"{vertex_count} vertices in graph. {skipped_count} vetices were"
                    f" removed from the graph as they either had no edges or no color.")
        return self.graph_builder.generate_graph()

    def get_vertex_key(self):
        return self.person_identifier.get_vertex_key()

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
    
    def __call__(self, *args, **kwargs):
        workbook, formats = self._open_workbook(kwargs.get('output_file', '/tmp/rbg.csv'))
        worksheet = workbook.add_worksheet()
        worksheet.set_default_row(hide_unused_rows=True)
        R = args[0].tolist()
        n = len(R)
        if n > MAX_COLUMNS_EXCEL:
            logging.error("Graph exceeds max size allowable by Excel")
        key_permutation = kwargs.get('key_permutation', np.arange(n))
        row = 0

        max_key = 0
        max_np = 0

        if self.vertex_key:
            column = 0
            worksheet.write(row, column, ' ')
            column += 1
            for column_idx in range(n):
                vertex_key = self.vertex_key[key_permutation[column_idx]]
                cell_data = f"{vertex_key[0]} - {vertex_key[1]}"
                max_key = max(max_key, len(cell_data))
                worksheet.write(row, column_idx + column, cell_data, formats[ROTATE_90])
            row += 1

        logger.debug(f"Graph size: {n}")
        for row_idx in range(n):
            if (row_idx + 1) % 500 == 0:
                logger.debug(f"completed row {row_idx + 1} of {n}")
            column = 0
            if self.vertex_key:
                vertex_key = self.vertex_key[key_permutation[row_idx]]
                cell_data = f"{vertex_key[0]} - {vertex_key[1]}"
                worksheet.write(row + row_idx, 0, cell_data)
                column += 1
            for column_idx in range(n):
                cell_data = R[row_idx][column_idx]
                if cell_data != 0:
                    max_np = max(max_np, cell_data)
                    worksheet.write(row + row_idx, column + column_idx, cell_data)
        column_width = self._calc_width(len(f"{max_np}"))
        worksheet.freeze_panes(1, 1)
        worksheet.set_column(0, 0, self._calc_width(max_key))
        worksheet.set_column(1, n, column_width)
        worksheet.set_column(n + 1, MAX_COLUMNS_EXCEL - 1, None, None, {'hidden': True})
        workbook.close()

    def append_vertex_key(self, key):
        self.vertex_key[len(self.vertex_key)] = key
