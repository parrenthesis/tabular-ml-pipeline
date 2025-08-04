import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Drop tables if they exist
c.execute('DROP TABLE IF EXISTS Patients')
c.execute('DROP TABLE IF EXISTS Teeth')
c.execute('DROP TABLE IF EXISTS Cavities')
c.execute('DROP TABLE IF EXISTS Treatments')

# Create tables
c.execute('''CREATE TABLE Patients (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    sex TEXT
)''')
c.execute('''CREATE TABLE Teeth (
    id INTEGER PRIMARY KEY,
    patientId INTEGER,
    toothType REAL,
    area REAL
)''')
c.execute('''CREATE TABLE Cavities (
    id INTEGER PRIMARY KEY,
    toothId INTEGER,
    area REAL,
    centerX REAL,
    centerY REAL
)''')
c.execute('''CREATE TABLE Treatments (
    id INTEGER PRIMARY KEY,
    toothId INTEGER,
    cost REAL
)''')

# Insert fake data
# Ensure at least 20 teeth with a treatment and 20 without (for stratified splitting and CV)
patients = [
    (1, 35, 'M'),
    (2, 42, 'F'),
    (3, 29, 'F'),
    (4, 50, 'M'),
]
teeth = []
cavities = []
treatments = []

# Teeth 1-20: with treatments
for i in range(1, 21):
    patient_id = (i % 4) + 1
    teeth.append((i, patient_id, float(i % 3 + 1), 10.0 + i))
    cavities.append((i, i, 1.0 + 0.1 * i, 0.5, 0.5))
    treatments.append((i, i, 100.0 + 10 * i))
# Teeth 21-40: without treatments
for i in range(21, 41):
    patient_id = (i % 4) + 1
    teeth.append((i, patient_id, float(i % 3 + 1), 10.0 + i))
    cavities.append((i, i, 1.0 + 0.1 * i, 0.5, 0.5))

c.executemany('INSERT INTO Patients (id, age, sex) VALUES (?, ?, ?)', patients)
c.executemany('INSERT INTO Teeth (id, patientId, toothType, area) VALUES (?, ?, ?, ?)', teeth)
c.executemany('INSERT INTO Cavities (id, toothId, area, centerX, centerY) VALUES (?, ?, ?, ?, ?)', cavities)
c.executemany('INSERT INTO Treatments (id, toothId, cost) VALUES (?, ?, ?)', treatments)

conn.commit()
conn.close()

print(f"Synthetic database created at {DB_PATH}")