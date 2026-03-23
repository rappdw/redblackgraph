"""
Name generation for the RedBlackGraph family DAG synthesizer.

Provides random European-style first and last names for synthetic individuals.
"""

import numpy as np

MALE_FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
    "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory", "Frank",
    "Alexander", "Patrick", "Jack", "Dennis", "Peter", "Henry", "Carl",
    "Arthur", "Albert", "Eugene", "Ralph", "Roy", "Louis", "Russell", "Philip",
    # German
    "Hans", "Friedrich", "Heinrich", "Wilhelm", "Karl", "Otto", "Franz", "Ludwig",
    "Werner", "Helmut", "Gerhard", "Dieter", "Klaus", "Manfred", "Rolf",
    "Wolfgang", "Siegfried", "Bernhard", "Erwin", "Konrad",
    # French
    "Jean", "Pierre", "Jacques", "Michel", "Andre", "Philippe", "Alain", "Rene",
    "Claude", "Marcel", "Francois", "Henri", "Yves", "Bernard", "Lucien",
    "Olivier", "Laurent", "Thierry", "Christophe", "Nicolas",
    # Italian
    "Giovanni", "Giuseppe", "Marco", "Antonio", "Francesco", "Alessandro",
    "Matteo", "Lorenzo", "Luca", "Andrea", "Roberto", "Stefano", "Massimo",
    "Fabio", "Paolo", "Gianluca", "Claudio", "Enrico", "Sergio", "Vittorio",
    # Scandinavian
    "Erik", "Lars", "Olaf", "Sven", "Anders", "Magnus", "Bjorn", "Gunnar",
    "Nils", "Leif", "Harald", "Ragnar", "Torsten", "Axel", "Gustaf",
    "Mikael", "Johan", "Henrik", "Kristian", "Oskar",
    # Spanish
    "Carlos", "Miguel", "Fernando", "Rafael", "Alejandro", "Javier", "Pablo",
    "Diego", "Manuel", "Rodrigo", "Sergio", "Andres", "Enrique", "Luis", "Ramon",
    # Dutch
    "Jan", "Pieter", "Willem", "Hendrik", "Cornelis", "Dirk", "Maarten",
    # Portuguese
    "Joao", "Pedro", "Tiago", "Gonçalo",
    # Irish/Scottish
    "Sean", "Patrick", "Liam", "Connor", "Declan", "Callum", "Angus", "Duncan",
    # Polish
    "Stanislaw", "Wojciech", "Tomasz", "Krzysztof", "Marek", "Janusz",
    # Hungarian
    "Istvan", "Laszlo", "Gabor", "Zoltan",
]

FEMALE_FIRST_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda",
    "Pamela", "Emma", "Nicole", "Helen", "Samantha", "Katherine", "Christine",
    "Debra", "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
    "Diane", "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria",
    "Kelly", "Lauren", "Christina", "Joan", "Evelyn", "Judith", "Andrea",
    # German
    "Ingrid", "Helga", "Ursula", "Hildegard", "Gertrud", "Elke", "Renate",
    "Monika", "Brigitte", "Hannelore", "Christa", "Erika", "Lieselotte",
    "Marlene", "Gisela", "Anneliese", "Frieda", "Waltraud", "Elfriede", "Ilse",
    # French
    "Marie", "Isabelle", "Nathalie", "Sophie", "Sylvie", "Monique", "Colette",
    "Simone", "Genevieve", "Marguerite", "Madeleine", "Camille", "Juliette",
    "Amelie", "Aurelie", "Celine", "Claire", "Dominique", "Veronique", "Brigitte",
    # Italian
    "Giulia", "Francesca", "Chiara", "Alessandra", "Valentina", "Elena",
    "Silvia", "Paola", "Cristina", "Roberta", "Daniela", "Federica",
    "Giovanna", "Antonella", "Claudia", "Patrizia", "Luisa", "Rosa", "Teresa",
    # Scandinavian
    "Ingeborg", "Astrid", "Sigrid", "Solveig", "Freya", "Greta", "Linnea",
    "Elsa", "Maja", "Karin", "Birgit", "Annika", "Hilda", "Dagny", "Liv",
    "Kristina", "Helena", "Margareta", "Gunhild", "Thyra",
    # Spanish
    "Carmen", "Isabel", "Pilar", "Dolores", "Lucia", "Sofia", "Elena",
    "Catalina", "Beatriz", "Esperanza", "Rosario", "Consuelo", "Paloma",
    # Dutch
    "Johanna", "Wilhelmina", "Cornelia", "Hendrika", "Grietje",
    # Irish/Scottish
    "Siobhan", "Niamh", "Fiona", "Eileen", "Moira", "Bridget",
    # Polish
    "Agnieszka", "Katarzyna", "Malgorzata", "Ewa", "Jadwiga",
    # Hungarian
    "Katalin", "Erzsebet", "Ilona", "Magdolna",
]

SURNAMES = [
    # English
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Robinson", "Clark", "Lewis", "Walker", "Hall",
    "Allen", "Young", "King", "Wright", "Hill", "Scott", "Green", "Adams",
    "Baker", "Nelson", "Carter", "Mitchell", "Roberts", "Turner", "Phillips",
    "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Morris",
    "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey", "Cooper",
    "Richardson", "Cox", "Howard", "Ward", "Brooks", "Gray", "Watson", "Wood",
    # German
    "Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner",
    "Becker", "Schulz", "Hoffmann", "Schaefer", "Koch", "Bauer", "Richter",
    "Klein", "Wolf", "Schroeder", "Neumann", "Schwarz", "Zimmermann",
    "Braun", "Krueger", "Hartmann", "Lange", "Werner", "Lehmann", "Krause",
    # French
    "Dupont", "Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Simon",
    "Michel", "Lefevre", "Leroy", "Roux", "David", "Bertrand", "Morel",
    "Girard", "Fournier", "Lambert", "Fontaine", "Rousseau", "Vincent",
    "Blanc", "Chevalier", "Garnier", "Bonnet", "Mercier",
    # Italian
    "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo",
    "Ricci", "Marino", "Greco", "Bruno", "Gallo", "Conti", "DeLuca", "Costa",
    "Giordano", "Mancini", "Rizzo", "Lombardi", "Moretti",
    # Scandinavian
    "Johansson", "Andersson", "Karlsson", "Nilsson", "Eriksson", "Larsson",
    "Olsson", "Persson", "Svensson", "Gustafsson", "Lindberg", "Lindqvist",
    "Nordstrom", "Bergstrom", "Holm", "Lindgren", "Lundberg", "Hansen",
    "Pedersen", "Christensen",
    # Spanish
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Hernandez",
    "Perez", "Sanchez", "Ramirez", "Torres", "Flores", "Rivera", "Gomez",
    "Diaz", "Cruz", "Reyes", "Morales", "Ortiz", "Gutierrez", "Ruiz",
    # Dutch
    "DeVries", "VanDijk", "Bakker", "Jansen", "Visser", "Smit", "Meijer",
    "DeBoer", "Mulder", "DeGroot",
    # Polish
    "Nowak", "Kowalski", "Wisniewski", "Kaminski", "Lewandowski",
    "Zielinski", "Szymanski", "Wozniak", "Kozlowski", "Jankowski",
    # Portuguese
    "Silva", "Santos", "Oliveira", "Souza", "Pereira", "Ferreira",
    # Irish
    "O'Brien", "O'Sullivan", "O'Connor", "O'Neill", "McCarthy",
    "Fitzgerald", "Walsh", "Byrne", "Ryan", "Kelly",
    # Hungarian
    "Nagy", "Kovacs", "Toth", "Szabo", "Horvath", "Varga", "Kiss", "Molnar",
]


class NameGenerator:
    """Generate random European-style names for synthetic individuals."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def random_male_name(self, surname: str | None = None) -> tuple:
        """Return (first_name, last_name). Uses given surname or picks random."""
        first = self.rng.choice(MALE_FIRST_NAMES)
        last = surname if surname is not None else self.rng.choice(SURNAMES)
        return (first, last)

    def random_female_name(self, surname: str | None = None) -> tuple:
        """Return (first_name, last_name). Uses given surname or picks random."""
        first = self.rng.choice(FEMALE_FIRST_NAMES)
        last = surname if surname is not None else self.rng.choice(SURNAMES)
        return (first, last)
