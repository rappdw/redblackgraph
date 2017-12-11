import csv
import os
from collections import defaultdict
import numpy as np
import redblackgraph as rb


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

if __name__ == "__main__":
    p_ids = {
        'p_id': 0
    }
    rows = defaultdict(lambda: {})
    with open("resources/person-relationship.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].startswith("#"):
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
    m = len(rows)
    A = np.zeros((m, m), dtype=np.int32).view(rb.array)
    for i in range(len(rows)):
        row = rows[i]
        for key in row.keys():
            A[i][key] = row[key]
    A_star = A.transitive_closure()
    cardinality = A.cardinality()

    print(f"Diameter: {A_star[1]}, # of male: {cardinality['red']}, # of female: {cardinality['black']}")

    inv_map = {v: k for k, v in p_ids.items()}

    A_list = A_star[0].tolist()
    with open("resources/results.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        header = [' '] + [f"{inv_map[x][0]}{inv_map[x][1]} - {inv_map[x][2]}" for x in range(m)]
        writer.writerow(header)
        for i in range(m):
            row = [f"{inv_map[i][0]}{inv_map[i][1]} - {inv_map[i][2]}"] + A_list[i]
            writer.writerow(row)

