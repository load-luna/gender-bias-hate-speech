# Analyse und Reduktion von Gender Bias in Hate-Speech-Klassifikationsmodellen

Dieses Repository enthält die Codebasis, Notebooks, Konfigurationen und Ergebnisse
für eine Bachelorarbeit zur Untersuchung von Gender Bias in BERT-basierten
Hate-Speech-Klassifikationsmodellen.

## Projektziel

Untersucht werden insbesondere:

- eine Baseline ohne Bias-Mitigation
- Pre-Processing durch Gender-Balancing
- Data Augmentation durch Gender-Swapping
- In-Processing durch Fairness Constraints
- Performance-Metriken
- Fairness-Metriken
- Counterfactual Testing als zusätzliche Analyse

## Modell

Als Ausgangspunkt ist `bert-base-uncased` vorgesehen. Der konkrete Modellname
kann in den YAML-Konfigurationen geändert werden.

## Repository-Struktur

```text
gender-bias-hate-speech/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline.yaml
│   ├── balancing.yaml
│   ├── gender_swap.yaml
│   └── fairness_constraints.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_gender_labeling.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_balancing.ipynb
│   ├── 05_gender_swapping.ipynb
│   ├── 06_fairness_constraints.ipynb
│   └── 07_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── models/
│   ├── baseline/
│   ├── balancing/
│   ├── gender_swap/
│   └── fairness_constraints/
│
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── plots/
│   └── tables/
│
└── docs/
```

## Empfohlener Workflow in Google Colab

1. Repository klonen:

```python
!git clone https://github.com/DEIN-USERNAME/gender-bias-hate-speech.git
%cd gender-bias-hate-speech
```

2. Abhängigkeiten installieren:

```python
!pip install -r requirements.txt
```

3. Einen GPU-Runtime-Typ in Colab aktivieren.

4. Ein Notebook aus `notebooks/` öffnen bzw. ausführen.

5. Große Datensätze und trainierte Modelle nicht ungeprüft in GitHub speichern.
   Nutze dafür beispielsweise Google Drive oder einen geeigneten Artefakt-/Cloud-Speicher.

## Reproduzierbarkeit

Experimente sollten möglichst über die YAML-Dateien in `configs/` gesteuert werden.
So können zentrale Trainingsparameter dokumentiert und zwischen Experimenten
konsistent gehalten werden.

## Vor dem ersten Lauf

Die vorbereiteten Python-Dateien enthalten bewusst noch Platzhalter bzw.
einfache Ausgangsfunktionen. Dataset-spezifische Details, Gender-Labeling-Logik,
Mitigation-Implementierungen und die konkrete Hugging-Face-Trainingskonfiguration
müssen an den verwendeten Datensatz und das finale Studiendesign angepasst werden.

## GitHub-Hinweise

Die `.gitignore` verhindert standardmäßig das Einchecken großer bzw. generierter
Dateien aus `data/`, `models/` und `results/`.

Die `.gitkeep`-Dateien sorgen dafür, dass die vorgesehenen Verzeichnisse trotz
fehlender Dateien im Repository sichtbar bleiben.

## Status

Projekt-Template für die Bachelorarbeit.
