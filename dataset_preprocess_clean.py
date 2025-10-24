# -*- coding: utf-8 -*-
"""
Cleaned script auto-generated from notebook: datase-traitment.ipynb
Purpose: Dataset cleaning/merging for Arabic NER (ANERcorp, AQMAR, WDC, Tweets, ...).
Notes:
  - Jupyter magics removed, imports deduplicated.
  - Consider parameterizing input/output paths via environment variables:
        DATA_IN (default: ./data_raw)
        DATA_OUT (default: ./data_clean)
  - If the original notebook used /kaggle/input or /kaggle/working, adapt to local folders.
"""
import os
DATA_IN  = os.getenv("DATA_IN", "./data_raw")
DATA_OUT = os.getenv("DATA_OUT", "./data_clean")
os.makedirs(DATA_OUT, exist_ok=True)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from collections import Counter
import random
import pandas as pd
import re
import shutil

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# === [CELL SEPARATOR] ===

# Définition des chemins des fichiers d'entrée
file1_path = "/kaggle/input/wojood-fine/WojoodFine/flat/train.txt"
file2_path = "/kaggle/input/wojood-fine/WojoodFine/flat/test.txt"
file3_path = "/kaggle/input/wojood-fine/WojoodFine/flat/val.txt"
# Fichier de sortie combiné
output_file = "/kaggle/working/wojood.txt"

# Liste des fichiers à combiner
input_files = [file1_path, file2_path, file3_path]

# Ouvrir le fichier de sortie en mode écriture
with open(output_file, "w", encoding="utf-8") as output_f:
    # Lire chaque fichier et les ajouter au fichier de sortie
    for file_path in input_files:
        # Vérifier si le fichier existe
        if isinstance(file_path, str) and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Écrire le contenu du fichier dans le fichier de sortie
            output_f.write(content + "\n")  # Ajouter une nouvelle ligne entre les fichiers
            print(f"Contenu de {file_path} ajouté au fichier combiné.")
        else:
            print(f"Fichier introuvable ou chemin invalide : {file_path}")

# Afficher l'emplacement du fichier combiné
print("\nLes fichiers ont été combinés avec succès dans :")
print(output_file)

# === [CELL SEPARATOR] ===

def compter_toutes_les_lignes(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()
    return len(lignes)

# Exemple d'utilisation
fichier_entree = "/kaggle/working/wojood.txt"
nb_lignes = compter_toutes_les_lignes(fichier_entree)
print(f"Nombre total de lignes (y compris vides) : {nb_lignes}")


# === [CELL SEPARATOR] ===


def compter_entites_par_type(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    compteur = Counter()

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue  # ignorer les lignes vides
        parts = ligne.split()
        if len(parts) != 2:
            continue  # ignorer les lignes mal formatées
        mot, etiquette = parts

        if etiquette.startswith("B-"):
            type_entite = etiquette[2:]
            compteur[type_entite] += 1

    return compteur

# Exemple d'utilisation
fichier_entree = "/kaggle/working/wojood.txt"
stats = compter_entites_par_type(fichier_entree)

# Affichage trié par nombre décroissant
for type_entite, nombre in stats.most_common():
    print(f"{type_entite}: {nombre}")


# === [CELL SEPARATOR] ===

def convert_bio_tag(original_tag):
    """
    Convertit une étiquette BIO selon les règles données.
    """
    if original_tag == 'O':
        return 'O'

    try:
        prefix, label = original_tag.split('-', 1)
    except ValueError:
        return 'O'  # étiquette invalide

    label_map = {
        'GPE': 'LOC',
        'FAC': 'LOC',
        'WEBSITE': 'ORG'
    }

    if label in ['PERS', 'LOC', 'ORG']:
        return original_tag
    elif label in label_map:
        return f"{prefix}-{label_map[label]}"
    else:
        return 'O'


def process_bio_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if line.strip() == '':
                outfile.write('\n')
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                outfile.write(line)  # ligne invalide ou commentaire
                continue

            token, tag = parts[0], parts[-1]
            new_tag = convert_bio_tag(tag)
            outfile.write(f"{token} {new_tag}\n")


# Exemple d’utilisation
if __name__ == "__main__":
    input_path = "wojood.txt"      # remplace par ton fichier original
    output_path = "wojood-1.txt"    # fichier annoté nettoyé
    process_bio_file(input_path, output_path)


# === [CELL SEPARATOR] ===


def count_entity_labels(file_path):
    counter = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Ignorer les lignes vides
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            tag = parts[-1]

            if tag != 'O':
                try:
                    prefix, label = tag.split('-', 1)
                    counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé
                    counter[tag] += 1

    return counter


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "wojood-1.txt"  # Remplace par ton fichier BIO
    entity_counts = count_entity_labels(input_file)

    print("📊 Statistiques des entités nommées :")
    for entity, count in entity_counts.most_common():
        print(f"{entity}: {count}")


# === [CELL SEPARATOR] ===


def analyze_bio_file(file_path):
    entity_counter = Counter()
    empty_lines = 0
    o_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':
                empty_lines += 1
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            tag = parts[-1]

            if tag == 'O':
                o_count += 1
            else:
                try:
                    _, label = tag.split('-', 1)
                    entity_counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé (ex: juste un label sans B- ou I-)
                    entity_counter[tag] += 1

    # Affichage
    print("📊 Statistiques du fichier BIO :")
    print(f"- Lignes vides : {empty_lines}")
    print(f"- Occurrences de 'O' : {o_count}")
    print("- Occurrences par entité :")
    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "wojood-1.txt"  # Remplace par ton fichier BIO
    analyze_bio_file(input_file)


# === [CELL SEPARATOR] ===


def read_sentences(file_path):
    sentences = []
    current = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line)
        if current:  # dernière phrase sans saut final
            sentences.append(current)

    return sentences

def write_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line)
            f.write('\n')

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    # Lire et mélanger les phrases
    sentences = read_sentences(input_file)
    random.seed(seed)
    random.shuffle(sentences)

    total = len(sentences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = sentences[:train_end]
    val_set = sentences[train_end:val_end]
    test_set = sentences[val_end:]

    write_sentences(train_set, 'train-w.txt')
    write_sentences(val_set, 'val-w.txt')
    write_sentences(test_set, 'test-w.txt')

    print("✅ Fichiers générés :")
    print(f"- train.txt : {len(train_set)} phrases")
    print(f"- val.txt   : {len(val_set)} phrases")
    print(f"- test.txt  : {len(test_set)} phrases")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "wojood-1.txt"  # ton fichier BIO complet
    split_dataset(input_file)


# === [CELL SEPARATOR] ===


# Charger le fichier Excel
file_path = "/kaggle/input/anercorpus/ANERCorp.xlsx"  # Remplace par le chemin de ton fichier
df = pd.read_excel(file_path, engine="openpyxl")

# Convertir en texte avec un espace comme séparateur
text_data = df.to_csv(sep=" ", index=False, header=False)

# Sauvegarder dans un fichier texte
with open("anercopus.txt", "w", encoding="utf-8") as f:
    f.write(text_data)

print("Conversion terminée. Le fichier est créé.")

# === [CELL SEPARATOR] ===


def compter_entites_par_type(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    compteur = Counter()

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue  # ignorer les lignes vides
        parts = ligne.split()
        if len(parts) != 2:
            continue  # ignorer les lignes mal formatées
        mot, etiquette = parts

        if etiquette.startswith("B-") or etiquette.startswith("I-"):
            type_entite = etiquette.split("-")[1]
            compteur[type_entite] += 1

    return compteur

# Exemple d'utilisation
fichier_entree = "/kaggle/working/anercopus.txt"
stats = compter_entites_par_type(fichier_entree)

# Affichage trié par nombre décroissant
for type_entite, nombre in stats.most_common():
    print(f"{type_entite}: {nombre}")


# === [CELL SEPARATOR] ===

def process_anercorp(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                outfile.write('\n')
                continue

            parts = line.split()
            if len(parts) < 2:
                outfile.write(line + '\n')
                continue

            token, tag = parts[0], parts[-1]

            # Convertir MISC → O
            if tag in ['B-MISC', 'I-MISC']:
                tag = 'O'

            outfile.write(f"{token} {tag}\n")

            # Ajouter une ligne vide après ". O"
            if token == '.' and tag == 'O':
                outfile.write('\n')

    print("✅ Traitement terminé. Fichier nettoyé écrit dans :", output_file)


# Exemple d'utilisation
if __name__ == "__main__":
    input_path = "/kaggle/working/anercopus.txt"      # Fichier d'entrée
    output_path = "anercorpus.txt"  # Fichier de sortie
    process_anercorp(input_path, output_path)


# === [CELL SEPARATOR] ===


def compter_entites_par_type(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    compteur = Counter()

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue  # ignorer les lignes vides
        parts = ligne.split()
        if len(parts) != 2:
            continue  # ignorer les lignes mal formatées
        mot, etiquette = parts

        if etiquette.startswith("B-") or etiquette.startswith("I-"):
            type_entite = etiquette.split("-")[1]
            compteur[type_entite] += 1

    return compteur

# Exemple d'utilisation
fichier_entree = "/kaggle/working/anercorpus.txt"
stats = compter_entites_par_type(fichier_entree)

# Affichage trié par nombre décroissant
for type_entite, nombre in stats.most_common():
    print(f"{type_entite}: {nombre}")


# === [CELL SEPARATOR] ===


def is_valid_tag(tag):
    if tag == 'O':
        return True
    # Expression régulière pour les tags valides BIO
    return re.match(r'^[BI]-(PERS|LOC|ORG|MISC)$', tag) is not None

def extract_invalid_tags(file_path):
    invalid_tags = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            tag = parts[-1]
            if not is_valid_tag(tag):
                invalid_tags[tag] += 1

    print("🚨 Étiquettes malformées détectées :")
    for tag, count in invalid_tags.most_common():
        print(f"{tag}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "/kaggle/working/anercorpus.txt"  # Remplace par ton fichier
    extract_invalid_tags(input_file)


# === [CELL SEPARATOR] ===


def is_valid_tag(tag):
    if tag == 'O':
        return True
    return re.match(r'^[BI]-(PERS|LOC|ORG)$', tag) is not None

def clean_bio_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            stripped = line.strip()

            if not stripped:
                outfile.write('\n')
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue  # ligne incomplète

            tag = parts[-1]

            if is_valid_tag(tag):
                outfile.write(line)
            else:
                continue  # ligne mal formée supprimée

    print(f"✅ Fichier nettoyé écrit dans : {output_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "/kaggle/working/anercorpus.txt"          # Fichier source
    output_file = "anercorp-1.txt"  # Fichier nettoyé
    clean_bio_file(input_file, output_file)


# === [CELL SEPARATOR] ===


def analyze_bio_file(file_path):
    entity_counter = Counter()
    empty_lines = 0
    o_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':
                empty_lines += 1
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            tag = parts[-1]

            if tag == 'O':
                o_count += 1
            else:
                try:
                    _, label = tag.split('-', 1)
                    entity_counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé (ex: juste un label sans B- ou I-)
                    entity_counter[tag] += 1

    # Affichage
    print("📊 Statistiques du fichier BIO :")
    print(f"- Lignes vides : {empty_lines}")
    print(f"- Occurrences de 'O' : {o_count}")
    print("- Occurrences par entité :")
    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "anercorp-1.txt"  # Remplace par ton fichier BIO
    analyze_bio_file(input_file)


# === [CELL SEPARATOR] ===


def read_sentences(file_path):
    sentences = []
    current = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line)
        if current:  # dernière phrase sans saut final
            sentences.append(current)

    return sentences

def write_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line)
            f.write('\n')

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    # Lire et mélanger les phrases
    sentences = read_sentences(input_file)
    random.seed(seed)
    random.shuffle(sentences)

    total = len(sentences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = sentences[:train_end]
    val_set = sentences[train_end:val_end]
    test_set = sentences[val_end:]

    write_sentences(train_set, 'train-an.txt')
    write_sentences(val_set, 'val-an.txt')
    write_sentences(test_set, 'test-an.txt')

    print("✅ Fichiers générés :")
    print(f"- train.txt : {len(train_set)} phrases")
    print(f"- val.txt   : {len(val_set)} phrases")
    print(f"- test.txt  : {len(test_set)} phrases")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "anercorp-1.txt"  # ton fichier BIO complet
    split_dataset(input_file)


# === [CELL SEPARATOR] ===


# Définition du dossier contenant les fichiers du dataset
dataset_dir = "/kaggle/input/aqmarcorp"  # Modifier avec le bon chemin
output_file = "/kaggle/working/aqmar.txt"  # Fichier de sortie combiné

# Liste des fichiers du dataset
files = [
    "Crusades.txt", "Damascus.txt", "Ibn_Tolun_Mosque.txt", "Imam_Hussein_Shrine.txt",
    "Islamic_Golden_Age.txt", "Islamic_History.txt", "Ummaya_Mosque.txt", "Atom.txt",
    "Enrico_Fermi.txt", "Light.txt", "Nuclear_Power.txt", "Periodic_Table.txt",
    "Physics.txt", "Razi.txt", "Christiano_Ronaldo.txt", "Football.txt",
    "Portugal_football_team.txt", "Raul_Gonzales.txt", "Real_Madrid.txt",
    "Soccer_Worldcup.txt", "Summer_Olympics2004.txt", "Computer_Software.txt",
    "Computer.txt", "Internet.txt", "Linux.txt", "Richard_Stallman.txt",
    "Solaris.txt", "X_window_system.txt"
]

# Vérification de l'existence des fichiers
existing_files = [file for file in files if os.path.exists(os.path.join(dataset_dir, file))]

# Affichage des fichiers trouvés et ceux absents
print("Fichiers trouvés :")
print(existing_files)

# Ouvrir le fichier de sortie en mode écriture
with open(output_file, "w", encoding="utf-8") as output_f:
    # Lire chaque fichier, appliquer les modifications et les ajouter au fichier de sortie
    for file in existing_files:
        # Chemin du fichier source
        file_path = os.path.join(dataset_dir, file)

        # Lecture du contenu du fichier
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remplacer les mots B-PER et I-PER par B-PERS et I-PERS
        content = content.replace("B-PER", "B-PERS").replace("I-PER", "I-PERS")

        # Écrire le contenu modifié dans le fichier de sortie
        output_f.write(content + "\n")  # Ajouter une nouvelle ligne entre les fichiers

        print(f"Modifications appliquées à : {file}")

# Afficher l'emplacement du fichier combiné
print("\nTous les fichiers ont été combinés et enregistrés dans :")
print(output_file)


# === [CELL SEPARATOR] ===


def compter_entites_par_type(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    compteur = Counter()

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue  # ignorer les lignes vides
        parts = ligne.split()
        if len(parts) != 2:
            continue  # ignorer les lignes mal formatées
        mot, etiquette = parts

        if etiquette.startswith("B-"):
            type_entite = etiquette[2:]
            compteur[type_entite] += 1

    return compteur

# Exemple d'utilisation
fichier_entree = "/kaggle/working/aqmar.txt"
stats = compter_entites_par_type(fichier_entree)

# Affichage trié par nombre décroissant
for type_entite, nombre in stats.most_common():
    print(f"{type_entite}: {nombre}")


# === [CELL SEPARATOR] ===


def filter_pers_loc_org(input_file, output_file):
    valid_labels = {"PERS", "LOC", "ORG"}

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            stripped = line.strip()

            if not stripped:
                outfile.write('\n')
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            token, tag = parts[0], parts[-1]

            if tag == "O":
                new_tag = "O"
            else:
                match = re.match(r'^([BI])-(.+)$', tag)
                if match:
                    prefix, label = match.groups()
                    if label in valid_labels:
                        new_tag = f"{prefix}-{label}"
                    else:
                        new_tag = "O"
                else:
                    new_tag = "O"  # tag malformé

            outfile.write(f"{token} {new_tag}\n")

    print(f"✅ Fichier nettoyé et enregistré dans : {output_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "aqmar.txt"                # Ton fichier d'origine
    output_file = "aqmar-1.txt"  # Fichier nettoyé
    filter_pers_loc_org(input_file, output_file)


# === [CELL SEPARATOR] ===


def analyze_bio_file(file_path):
    entity_counter = Counter()
    empty_lines = 0
    o_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':
                empty_lines += 1
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            tag = parts[-1]

            if tag == 'O':
                o_count += 1
            else:
                try:
                    _, label = tag.split('-', 1)
                    entity_counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé (ex: juste un label sans B- ou I-)
                    entity_counter[tag] += 1

    # Affichage
    print("📊 Statistiques du fichier BIO :")
    print(f"- Lignes vides : {empty_lines}")
    print(f"- Occurrences de 'O' : {o_count}")
    print("- Occurrences par entité :")
    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "aqmar-1.txt"  # Remplace par ton fichier BIO
    analyze_bio_file(input_file)


# === [CELL SEPARATOR] ===


def count_all_entity_tags(file_path):
    tag_counter = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped == 'O':
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue  # ligne incomplète ou mal formée

            tag = parts[-1]

            # Extraire la partie sans le préfixe BIO s’il y en a
            if '-' in tag:
                _, label = tag.split('-', 1)
            else:
                label = tag  # mal formé ou sans préfixe

            tag_counter[label] += 1

    print("📊 Statistiques des entités (y compris mal formées) :")
    for label, count in tag_counter.most_common():
        print(f"{label}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "aqmar-1.txt"  # Remplace par ton fichier
    count_all_entity_tags(input_file)


# === [CELL SEPARATOR] ===


def is_valid_bio_tag(tag):
    """
    Retourne True si le tag est de type O ou BIO correct (B-PERS, I-LOC, etc.)
    """
    if tag == 'O':
        return True
    return re.fullmatch(r'[BI]-(PERS|LOC|ORG)', tag) is not None

def find_malformed_lines(file_path):
    malformed_lines = []
    line_number = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            stripped = line.strip()

            if not stripped:
                continue  # ligne vide OK

            parts = stripped.split()

            # Ligne sans assez de colonnes
            if len(parts) < 2:
                malformed_lines.append((line_number, line.strip(), '🔴 Moins de 2 éléments'))
                continue

            token, tag = parts[0], parts[-1]

            if not is_valid_bio_tag(tag):
                malformed_lines.append((line_number, line.strip(), f'⚠️ Tag invalide : {tag}'))

    # Affichage
    print("🧾 Lignes mal formées détectées :")
    for lineno, content, reason in malformed_lines:
        print(f"Ligne {lineno}: {content} ({reason})")

    print(f"\n✅ Total : {len(malformed_lines)} lignes mal formées trouvées.")

    return malformed_lines


# Exemple d’utilisation
if __name__ == "__main__":
    input_file = "aqmar-1.txt"  # Remplace par ton fichier
    find_malformed_lines(input_file)


# === [CELL SEPARATOR] ===

def compter_toutes_les_lignes(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()
    return len(lignes)

# Exemple d'utilisation
fichier_entree = "/kaggle/working/anercorp-1.txt"
nb_lignes = compter_toutes_les_lignes(fichier_entree)
print(f"Nombre total de lignes (y compris vides) : {nb_lignes}")

# === [CELL SEPARATOR] ===


def read_sentences(file_path):
    sentences = []
    current = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line)
        if current:  # dernière phrase sans saut final
            sentences.append(current)

    return sentences

def write_sentences(sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line)
            f.write('\n')

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    # Lire et mélanger les phrases
    sentences = read_sentences(input_file)
    random.seed(seed)
    random.shuffle(sentences)

    total = len(sentences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = sentences[:train_end]
    val_set = sentences[train_end:val_end]
    test_set = sentences[val_end:]

    write_sentences(train_set, 'train-aq.txt')
    write_sentences(val_set, 'val-aq.txt')
    write_sentences(test_set, 'test-aq.txt')

    print("✅ Fichiers générés :")
    print(f"- train.txt : {len(train_set)} phrases")
    print(f"- val.txt   : {len(val_set)} phrases")
    print(f"- test.txt  : {len(test_set)} phrases")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "aqmar-1.txt"  # ton fichier BIO complet
    split_dataset(input_file)


# === [CELL SEPARATOR] ===

def combine_files(file1, file2, file3, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in [file1, file2, file3]:
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                if not outfile.tell() == 0:
                    outfile.write('\n')  # Séparateur de fichier

    print(f"✅ Fichiers combinés dans : {output_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    file1 = "/kaggle/working/val-w.txt"
    file2 = "/kaggle/working/val-aq.txt"
    file3 = "/kaggle/working/val-an.txt"
    output = "val.txt"
    combine_files(file1, file2, file3, output)


# === [CELL SEPARATOR] ===

def combine_files(file1, file2, file3, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in [file1, file2, file3]:
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                if not outfile.tell() == 0:
                    outfile.write('\n')  # Séparateur de fichier

    print(f"✅ Fichiers combinés dans : {output_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    file1 = "/kaggle/working/train-w.txt"
    file2 = "/kaggle/working/train-aq.txt"
    file3 = "/kaggle/working/train-an.txt"
    output = "train.txt"
    combine_files(file1, file2, file3, output)


# === [CELL SEPARATOR] ===

def combine_files(file1, file2, file3, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in [file1, file2, file3]:
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                if not outfile.tell() == 0:
                    outfile.write('\n')  # Séparateur de fichier

    print(f"✅ Fichiers combinés dans : {output_file}")


# Exemple d'utilisation
if __name__ == "__main__":
    file1 = "/kaggle/working/test-w.txt"
    file2 = "/kaggle/working/test-aq.txt"
    file3 = "/kaggle/working/test-an.txt"
    output = "test.txt"
    combine_files(file1, file2, file3, output)


# === [CELL SEPARATOR] ===


def is_valid_entity(tag):
    """Vérifie si le tag est conforme à BIO avec PERS, LOC, ORG ou O"""
    return tag == "O" or re.fullmatch(r"[BI]-(PERS|LOC|ORG)", tag) is not None

def clean_and_filter_bio(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            stripped = line.strip()

            # Conserver les lignes vides (séparation des phrases)
            if not stripped:
                outfile.write("\n")
                continue

            parts = stripped.split()

            if len(parts) != 2:
                continue  # ligne mal formée → ignorée

            token, tag = parts

            if is_valid_entity(tag):
                outfile.write(f"{token} {tag}\n")
            # Sinon, on ignore la ligne

    print(f"✅ Cleaned file saved to: {output_file}")


# Exemple d’utilisation
if __name__ == "__main__":
    input_path = "train.txt"         # fichier source
    output_path = "train-c.txt"    # fichier nettoyé
    clean_and_filter_bio(input_path, output_path)


# === [CELL SEPARATOR] ===


def is_valid_entity(tag):
    """Vérifie si le tag est conforme à BIO avec PERS, LOC, ORG ou O"""
    return tag == "O" or re.fullmatch(r"[BI]-(PERS|LOC|ORG)", tag) is not None

def clean_and_filter_bio(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            stripped = line.strip()

            # Conserver les lignes vides (séparation des phrases)
            if not stripped:
                outfile.write("\n")
                continue

            parts = stripped.split()

            if len(parts) != 2:
                continue  # ligne mal formée → ignorée

            token, tag = parts

            if is_valid_entity(tag):
                outfile.write(f"{token} {tag}\n")
            # Sinon, on ignore la ligne

    print(f"✅ Cleaned file saved to: {output_file}")


# Exemple d’utilisation
if __name__ == "__main__":
    input_path = "test.txt"         # fichier source
    output_path = "test-c.txt"    # fichier nettoyé
    clean_and_filter_bio(input_path, output_path)


# === [CELL SEPARATOR] ===


def is_valid_entity(tag):
    """Vérifie si le tag est conforme à BIO avec PERS, LOC, ORG ou O"""
    return tag == "O" or re.fullmatch(r"[BI]-(PERS|LOC|ORG)", tag) is not None

def clean_and_filter_bio(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            stripped = line.strip()

            # Conserver les lignes vides (séparation des phrases)
            if not stripped:
                outfile.write("\n")
                continue

            parts = stripped.split()

            if len(parts) != 2:
                continue  # ligne mal formée → ignorée

            token, tag = parts

            if is_valid_entity(tag):
                outfile.write(f"{token} {tag}\n")
            # Sinon, on ignore la ligne

    print(f"✅ Cleaned file saved to: {output_file}")


# Exemple d’utilisation
if __name__ == "__main__":
    input_path = "val.txt"         # fichier source
    output_path = "val-c.txt"    # fichier nettoyé
    clean_and_filter_bio(input_path, output_path)


# === [CELL SEPARATOR] ===


def analyze_bio_file(file_path):
    entity_counter = Counter()
    empty_lines = 0
    o_count = 0
    total_tokens = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':
                empty_lines += 1
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            total_tokens += 1  # Chaque ligne non vide avec 2 éléments est un token

            tag = parts[-1]

            if tag == 'O':
                o_count += 1
            else:
                try:
                    _, label = tag.split('-', 1)
                    entity_counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé (ex: juste un label sans B- ou I-)
                    entity_counter[tag] += 1

    # Affichage
    print("📊 Statistiques du fichier BIO :")
    print(f"- Nombre total de tokens       : {total_tokens}")
    print(f"- Nombre de phrases (vides)    : {empty_lines}")
    print(f"- Occurrences de 'O'           : {o_count}")
    print("- Occurrences par entité :")
    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "train-c.txt"  # Remplace par ton fichier BIO
    analyze_bio_file(input_file)


# === [CELL SEPARATOR] ===


def analyze_bio_file(file_path):
    entity_counter = Counter()
    empty_lines = 0
    o_count = 0
    total_tokens = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':
                empty_lines += 1
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            total_tokens += 1  # Chaque ligne non vide avec 2 éléments est un token

            tag = parts[-1]

            if tag == 'O':
                o_count += 1
            else:
                try:
                    _, label = tag.split('-', 1)
                    entity_counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé (ex: juste un label sans B- ou I-)
                    entity_counter[tag] += 1

    # Affichage
    print("📊 Statistiques du fichier BIO :")
    print(f"- Nombre total de tokens       : {total_tokens}")
    print(f"- Nombre de phrases (vides)    : {empty_lines}")
    print(f"- Occurrences de 'O'           : {o_count}")
    print("- Occurrences par entité :")
    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "test-c.txt"  # Remplace par ton fichier BIO
    analyze_bio_file(input_file)


# === [CELL SEPARATOR] ===


def analyze_bio_file(file_path):
    entity_counter = Counter()
    empty_lines = 0
    o_count = 0
    total_tokens = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line == '':
                empty_lines += 1
                continue

            parts = line.split()
            if len(parts) < 2:
                continue  # ligne mal formée

            total_tokens += 1  # Chaque ligne non vide avec 2 éléments est un token

            tag = parts[-1]

            if tag == 'O':
                o_count += 1
            else:
                try:
                    _, label = tag.split('-', 1)
                    entity_counter[label] += 1
                except ValueError:
                    # Cas où le tag est mal formé (ex: juste un label sans B- ou I-)
                    entity_counter[tag] += 1

    # Affichage
    print("📊 Statistiques du fichier BIO :")
    print(f"- Nombre total de tokens       : {total_tokens}")
    print(f"- Nombre de phrases (vides)    : {empty_lines}")
    print(f"- Occurrences de 'O'           : {o_count}")
    print("- Occurrences par entité :")
    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}")


# Exemple d'utilisation
if __name__ == "__main__":
    input_file = "val-c.txt"  # Remplace par ton fichier BIO
    analyze_bio_file(input_file)
