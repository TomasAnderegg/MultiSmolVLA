# Handoff : entraînement du pipeline MultiSmolVLA

**Rédigé par** : Tomas  
**Date** : 29 avril 2026  
**Objectif** : lancer et superviser les runs d'entraînement du pipeline sur le cluster EPFL (Izar)

---

## Contexte rapide

Le projet remplace l'encodeur visuel de SmolVLA par **4M-21** (encodeur multimodal EPFL-VILAB) et ajoute un curriculum de **modality dropout** pour rendre le modèle robuste aux capteurs manquants. L'architecture est en deux blocs :

- **Block1** : prend du RGB + depth + seg + thermal → applique un dropout progressif de modalités → encode le thermal via ImageBind (embedding 1024-d)
- **Block2** : encode le RGB avec 4M-21 → projette via un MLP connector → SmolVLA prédit les actions

---

## Ce dont tu as besoin

### 1. Le code — git clone

```bash
git clone --recurse-submodules https://github.com/TomasAnderegg/MultiSmolVLA.git
cd MultiSmolVLA
```

Puis setup de l'environnement : **lire [SETUP_IZAR.md](../SETUP_IZAR.md) en entier** avant de continuer.

### 2. Le dataset — Tomas doit te le transférer

Le dataset est un ensemble de 500 fichiers Parquet pré-générés qui contiennent les **4 modalités** :
- `observation.images.image` → RGB
- `observation.images.image_depth` → profondeur
- `observation.images.image_mask` → segmentation
- `observation.images.thermal` → thermique synthétique (pré-calculé)

Tomas doit copier ces fichiers vers ton scratch sur Izar. Il lancera :

```bash
rsync -av /home/garate/MultiSmolVLA/data/parquet_scratch/parquet_thermal/ \
          /scratch/izar/<ton_username>/data/parquet_thermal/
```

Une fois copié, tes données seront à :
```
/scratch/izar/<ton_username>/data/parquet_thermal/
```

> **Pourquoi le thermal est pré-calculé ?** ThermalGen est un modèle de diffusion — le faire tourner à chaque step d'entraînement serait très lent. On a pré-généré les images thermiques une fois pour toutes (job SLURM de 7h sur 4 GPUs en parallèle) et on les stocke directement dans le Parquet.

### 3. Le script SLURM — déjà dans le repo

Le fichier [scripts/run_training.sh](../scripts/run_training.sh) est prêt. Tu dois juste vérifier/adapter la ligne `DATA_DIR` :

```bash
# Dans run_training.sh, ligne ~22 :
DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
```

Si Tomas a copié les données à un autre endroit, change ce chemin.

---

## Lancer l'entraînement

```bash
# Depuis ton home sur Izar
cd /home/$USER/MultiSmolVLA
mkdir -p logs

# Soumettre le job
sbatch scripts/run_training.sh

# Suivre le log en temps réel
tail -f logs/training_<JOBID>.out
```

Le job demande **1 GPU, 64G RAM, 24h**. Les checkpoints sont sauvegardés toutes les 1000 steps dans `checkpoints/full_pipeline/`.

---

## Ce que fait le script d'entraînement

`train_full_pipeline.py` avec `--data_dir` :

1. Charge les parquets depuis `DATA_DIR` (dataset custom `ParquetThermalDataset`)
2. Pour chaque batch : lit RGB, depth, seg, thermal depuis le parquet
3. **Block1** : reçoit les 4 modalités → saute ThermalGen (thermal déjà dispo) → applique le curriculum dropout → encode le thermal via ImageBind
4. **Block2** : encode RGB via 4M-21 (gelé) → MLP connector → SmolVLA → prédit les actions
5. Backprop sur le MLP connector + action expert uniquement

Le **curriculum de dropout** augmente progressivement la probabilité de zeroing de chaque modalité (0% → 50% sur 20 000 steps). Cela force le modèle à ne pas dépendre d'un seul capteur.

---

## Ce qui est gelé / entraîné

| Composant | Status | Raison |
|---|---|---|
| ThermalGen | Gelé + bypassé | Thermal déjà dans le parquet |
| ImageBind | Gelé | Encodeur pré-entraîné, on ne veut pas le dégrader |
| 4M-21 | Gelé | Encodeur pré-entraîné EPFL-VILAB |
| **MLP connector** | **Entraîné** | Pont entre 4M-21 et SmolLM2 — pièce clé |
| **Action expert** | **Entraîné** | Prédit les actions motrices |
| SmolVLM | Gelé | LLM de base, on fine-tune seulement l'expert |

---

## Vérifier que l'entraînement converge

Dans les logs, cherche les lignes `loss=` :

```
2026-04-29 10:15:00 INFO step=50/20000  loss=2.4312
2026-04-29 10:17:00 INFO step=100/20000 loss=2.1854
...
```

La loss doit **descendre régulièrement** dans les premiers milliers de steps. Si elle stagne dès le début :
- Essayer `--lr_mlp 1e-3` (MLP apprend plus vite)
- Vérifier que le dataset est bien chargé (log `Dataset size: N samples`)

---

## Fichiers importants

```
MultiSmolVLA/
├── scripts/
│   ├── run_training.sh            ← soumettre ce script sur Izar
│   └── train_full_pipeline.py     ← script Python d'entraînement
├── src/pipeline/
│   ├── block1.py                  ← Block1 (thermal bypass si pré-calculé)
│   ├── block2.py                  ← Block2 (4M + MLP + SmolVLA)
│   ├── full_pipeline.py           ← pipeline complet
│   └── modality_dropout.py        ← curriculum de dropout
├── utils/
│   └── parquet_dataset.py         ← dataset custom pour les parquets
├── checkpoints/full_pipeline/     ← checkpoints sauvegardés ici
├── logs/                          ← logs SLURM
└── SETUP_IZAR.md                  ← setup complet du cluster (lire en premier)
```
