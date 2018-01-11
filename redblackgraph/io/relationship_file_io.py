"""
Relationship File IO is a collection of utilities to parse an input csv file
into a RB Graph.

The input file format consists of multiple rows of thw following format:
#First Name,Surname,Birthyear,Gender,Father FN,Father Sn,Father By,Mother FN,Mother Sn,Mother By

While it's possible for a given column (aside from gender) to be missing, it is required that
each individual in the input file can be uniquely identified by the tuple (FN,Sn,By)
"""
import csv
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np
import redblackgraph as rb
import xlsxwriter

ROTATE_90 = 'rotate-90'
COLUMNS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class PersonIdentifier:
    def __init__(self):
        self.person_dictionary = {'p_id': 0}
        
    def get_person_id(self, person: Tuple[str, str]) -> Optional[int]:
        """
        Given a person/bdate tuple, return the person_id (either looking up the person if
        already created an id, or generate a new one).

        If the person tuple is empty, returns None
        :param person: tuple[ given name, surname ]
        :return: person_id or None if tuple is empty
        """
        if person[0] or person[1]:
            if not person in self.person_dictionary:
                self.person_dictionary[person] = self.person_dictionary['p_id']
                self.person_dictionary['p_id'] += 1
            return self.person_dictionary[person]
        return None
    
    def get_vertex_key(self):
        return {v: k for k, v in self.person_dictionary.items()}


class GraphBuilder:
    def __init__(self):
        self.rbg_dictionary = defaultdict(lambda: {})
        
    def add_vertex(self, vertex_id, color, red_vertex_id: None, black_vertex_id: None):
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
    
    def generate_graph(self):
        # create an numpy array with the correct shape and load it with the tuples
        m = len(self.rbg_dictionary)
        R = np.zeros((m, m), dtype=np.int32).view(rb.array)
        for i in range(m):
            row = self.rbg_dictionary[i]
            for key in row.keys():
                R[i][key] = row[key]
        return R


class RelationshipFileReader:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.person_identifier = PersonIdentifier()
        self.graph_builder = GraphBuilder()
        
    def __call__(self, *args, **kwargs):
        with open(self.input_dir, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0].startswith("#") or row[0] in ["Vertex", "Fn"]:
                    continue
                vertex = tuple(row[0:3])
                red_vertex = tuple(row[4:7])
                black_vertex = tuple(row[7:])
                vertex_id = self.person_identifier.get_person_id(vertex)
                red_verex_id = self.person_identifier.get_person_id(red_vertex)
                black_vertex_id = self.person_identifier.get_person_id(black_vertex)
                self.graph_builder.add_vertex(vertex_id, int(row[3]), red_verex_id, black_vertex_id)
        return self.graph_builder.generate_graph()

    def get_vertex_key(self):
        return self.person_identifier.get_vertex_key()

    def get_person_id(self, person):
        return self.person_identifier.get_person_id(person)
    
class RedBlackGraphWriter:
    def __init__(self, vertex_key=None):
        self.vertex_key = vertex_key

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
        key_transpose = kwargs.get('key_transpose', np.arange(n))
        row = 0

        max_key = 0
        max_np = 0

        if self.vertex_key:
            column = 0
            worksheet.write(row, column, ' ')
            column += 1
            for column_idx in range(n):
                vertex_key = self.vertex_key[key_transpose[column_idx]]
                cell_data = f"{vertex_key[0]}{vertex_key[1]} - {vertex_key[2]}"
                max_key = max(max_key, len(cell_data))
                worksheet.write(row, column_idx + column, cell_data, formats[ROTATE_90])
            row += 1

        for row_idx in range(n):
            column = 0
            if self.vertex_key:
                vertex_key = self.vertex_key[key_transpose[row_idx]]
                cell_data = f"{vertex_key[0]}{vertex_key[1]} - {vertex_key[2]}"
                worksheet.write(row + row_idx, 0, cell_data)
                column += 1
            for column_idx in range(n):
                cell_data = R[row_idx][column_idx]
                max_np = max(max_np, cell_data)
                worksheet.write(row + row_idx, column + column_idx, cell_data)
        a = n // 26
        b = n % 26
        column_width = self._calc_width(len(f"{max_np}"))
        if self.vertex_key:
            worksheet.freeze_panes(1, 1)
            worksheet.set_column('A:A', self._calc_width(max_key))
            if a > 0:
                worksheet.set_column(f'B:{COLUMNS[a-1]}{COLUMNS[b]}', column_width)
                worksheet.set_column(f'{COLUMNS[a-1]}{COLUMNS[b+1]}:XFD', None, None, {'hidden': True})
            else:
                worksheet.set_column(f'B:{COLUMNS[b]}', column_width)
                worksheet.set_column(f'{COLUMNS[b+1]}:XFD', None, None, {'hidden': True})
        else:
            worksheet.set_column(f'A:{COLUMNS[b]}', column_width)
            worksheet.set_column(f'{COLUMNS[b+1]}:XFD', None, None, {'hidden': True})

    def append_vertex_key(self, key):
        self.vertex_key[len(self.vertex_key) - 1] = key
