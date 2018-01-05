import csv
from collections import defaultdict
import time
import numpy as np
import redblackgraph as rb
from redblackgraph.simple.warshall import warshall


def get_person_id(p, d):
    if not p in d:
        d[p] = d['p_id']
        d['p_id'] += 1
    return d[p]

def add_person(p_id, c, d, f_id=None, m_id=None):
    row = d[p_id]
    row[p_id] = c
    if not f_id is None:
        row[f_id] = 2
    if not m_id is None:
        row[m_id] = 3

def write_resutls(filename, inv_map, A_list, m):
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile)
        header = [' '] + [f"{inv_map[x][0]}{inv_map[x][1]} - {inv_map[x][2]}" for x in range(m)]
        writer.writerow(header)
        for i in range(m):
            row = [f"{inv_map[i][0]}{inv_map[i][1]} - {inv_map[i][2]}"] + A_list[i]
            writer.writerow(row)


if __name__ == "__main__":
    p_ids = {
        'p_id': 0
    }
    rows = defaultdict(lambda: {})
    # read the relationship tuples into python dictionaries
    with open("resources/person-relationship.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].startswith("#") or row[0] in ["Vertex", "Fn"]:
                continue
            p = tuple(row[0:3])
            f = tuple(row[4:7])
            m = tuple(row[7:])
            p_id = get_person_id(p, p_ids)
            f_id = get_person_id(f, p_ids)
            m_id = get_person_id(m, p_ids)
            add_person(p_id, int(row[3]), rows, f_id, m_id)
            add_person(f_id, -1, rows)
            add_person(m_id, 1, rows)
    # create an numpy array with the correct shape and load it with the tuples
    m = len(rows)
    A = np.zeros((m, m), dtype=np.int32).view(rb.array)
    for i in range(len(rows)):
        row = rows[i]
        for key in row.keys():
            A[i][key] = row[key]

    # compute the transitive closure: Compare execution tie of Numpy and pure Python
    start_time = time.time()
    for i in range(100):
        A_star = A.transitive_closure()
    duration_numpy = time.time() - start_time
    A_star = A.transitive_closure()

    duration_python = 1.5 * 3
    if False:
        start_time = time.time()
        for i in range(3):
            A_star = warshall(A)
        duration_python = time.time() - start_time

    cardinality = A.cardinality()

    print(f"Diameter: {A_star[1]}, # of male: {cardinality['red']}, # of female: {cardinality['black']}, "
          f"Average execution time for closure - Python: {duration_python/3}, Numpy: {duration_numpy/100}")

    # write out the results
    inv_map = {v: k for k, v in p_ids.items()}
    A_list = A_star[0].tolist()
    write_resutls("resources/closure.results.csv", inv_map, A_list, m)

    # Now perform a relational composition
    u = np.zeros((1, m), dtype=np.int32).view(rb.array)
    u[0][p_ids[('D', 'R', '1963')]] = 2
    u[0][p_ids[('B', 'V', '1960')]] = 3
    v = np.zeros((m + 1), dtype=np.int32).view(rb.array)
    inv_map[m] = ('B', 'M-R', '2001')

    start_time = time.time()
    for i in range(100):
        A_lambda = A_star[0].vertex_relational_composition(u, v, 1)
    duration_numpy = time.time() - start_time
    print(f"Average execution time for composition - Numpy: {duration_numpy/100}")

    A_lambda = A_star[0].vertex_relational_composition(u, v, 1)
    A_list = A_lambda.tolist()
    write_resutls("resources/composition.results.csv", inv_map, A_list, m + 1)
