import csv
from collections import defaultdict
from typing import Tuple, Optional

import numpy as np
import redblackgraph as rb
import redblackgraph.simple as smp


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
    
    def __call__(self, *args, **kwargs):
        R = args[0].tolist()
        m = len(R)
        with open(kwargs.get('output_file', '/tmp/rbg.csv'), "w") as csvfile:
            writer = csv.writer(csvfile)
            if self.vertex_key:
                header = [' '] + \
                         [f"{self.vertex_key[x][0]}{self.vertex_key[x][1]} - {self.vertex_key[x][2]}" for x in range(m)]
                writer.writerow(header)
            for i in range(m):
                if self.vertex_key:
                    row = [f"{self.vertex_key[i][0]}{self.vertex_key[i][1]} - {self.vertex_key[i][2]}"] + R[i]
                else:
                    row = R[i]
                writer.writerow(row)

    def append_vertex_key(self, key):
        self.vertex_key[len(self.vertex_key)] = key


if __name__ == "__main__":
    reader = RelationshipFileReader("../../tests/resources/person-relationship.csv")
    R = reader()

    writer = RedBlackGraphWriter(reader.get_vertex_key())
    writer(R, output_file="../../tests/resources/r.csv")

    # compute the transitive closure
    R_star, diameter = R.transitive_closure()
    cardinality = R.cardinality()

    # write out the results
    writer(R_star, output_file="../../tests/resources/closure.results.csv")
    components = smp.find_components(R_star.tolist())
    print(f"Found {components[1]} connected components")

    # Now perform a relational composition
    u = np.zeros((R.shape[0],), dtype=np.int32).view(rb.array)
    u[reader.get_person_id(('D', 'R', '1963'))] = 2
    u[reader.get_person_id(('B', 'V', '1960'))] = 3
    v = np.zeros((R.shape[0],), dtype=np.int32).view(rb.array)
    writer.append_vertex_key(('B', 'M-R', '2001'))

    R_lambda = R_star.vertex_relational_composition(u, v, 1)
    writer(R_lambda, output_file="../../tests/resources/composition.results.csv")
    components = smp.find_components(R_lambda.tolist())
    print(f"Found {components[1]} connected components")
