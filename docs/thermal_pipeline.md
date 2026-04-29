# Pipeline de génération thermique

## Vue d'ensemble

Le pipeline thermal a pour but d'**augmenter le dataset robotique existant** avec des images thermiques synthétiques. Comme on n'a pas de caméra thermique réelle, on utilise un modèle de deep learning pour **prédire** à quoi ressemblerait une image thermique à partir d'une image RGB normale.

---

## Concepts de base

### Qu'est-ce qu'un fichier Parquet ?

Un fichier **Parquet** est un format de stockage de données tabulaires (comme un tableau Excel ou un DataFrame pandas), mais optimisé pour les grosses quantités de données.

Dans ce projet, chaque fichier Parquet contient **un shard du dataset** — c'est-à-dire une tranche du dataset complet. Chaque ligne correspond à un pas de temps d'une trajectoire robotique et contient, entre autres colonnes :

| Colonne | Contenu |
|---|---|
| `observation.images.image` | Image RGB de la caméra (encodée en bytes PNG) |
| `action` | Commande moteur à ce pas de temps |
| `observation.state` | État proprioceptif du robot |
| `observation.images.thermal` | **(ajoutée par ce pipeline)** Image thermique synthétique |

Les images ne sont pas stockées comme des fichiers séparés, mais **directement dans le tableau** sous forme de dictionnaires `{"bytes": <PNG brut>, "path": <nom fictif>}`.

### Qu'est-ce qu'un worker (ou job SLURM) ?

Sur le cluster EPFL (Izar), on ne lance pas les scripts directement — on les soumet au **scheduler SLURM**, qui les met en file d'attente et les distribue sur les nœuds GPU disponibles.

Un **worker** ici est simplement un **processus indépendant** qui tourne sur un nœud GPU. On lance plusieurs workers en parallèle pour traiter le dataset plus vite. Chaque worker :
- charge le modèle ThermalGen sur son propre GPU
- traite un sous-ensemble des fichiers Parquet
- écrit ses résultats indépendamment des autres

Le script utilise la fonctionnalité **job array** de SLURM (`--array=0-3`), qui crée 4 jobs identiques mais avec un identifiant différent (`SLURM_ARRAY_TASK_ID` = 0, 1, 2, ou 3). C'est cet identifiant qui détermine quels shards chaque worker traite.

---

## Architecture détaillée

### Répertoires

```
/scratch/izar/garate/data/
├── parquet/           ← input : shards originaux (RGB uniquement)
└── parquet_thermal/   ← output : shards augmentés (RGB + thermal)
```

Le répertoire `/scratch` sur Izar est un stockage rapide temporaire — les données ne sont pas persistées indéfiniment, mais les I/O sont bien plus rapides que sur le home directory.

### Partitionnement des shards entre workers

Supposons qu'il y ait 20 fichiers Parquet (`shard_0000.parquet` ... `shard_0019.parquet`) et 4 workers. La distribution se fait par **slicing interleaved** :

```python
shards = all_shards[job_id::n_jobs]
```

| Worker (job_id) | Shards traités |
|---|---|
| 0 | 0, 4, 8, 12, 16 |
| 1 | 1, 5, 9, 13, 17 |
| 2 | 2, 6, 10, 14, 18 |
| 3 | 3, 7, 11, 15, 19 |

Cette approche est simple et robuste : pas de communication entre workers, pas de risk de conflit d'écriture sur le même fichier.

Si un shard de sortie existe déjà (`if os.path.exists(output_path): continue`), il est ignoré — ce qui permet de **relancer le job** sans tout recalculer en cas d'interruption.

---

## Le modèle : ThermalGen-XL-2

### Qu'est-ce que c'est ?

ThermalGen est un **modèle génératif conditionné sur le RGB** qui prédit une image thermique réaliste. Il est publié sur HuggingFace sous `xjh19972/ThermalGen-XL-2` et chargé automatiquement au démarrage.

Il est basé sur une architecture **SiT** (Scalable interpolant Transformer), similaire aux DiT (Diffusion Transformers) utilisés dans Stable Diffusion 3 ou FLUX.

### Architecture interne

```
RGB (256×256)
    │
    ▼
RGB KL-VAE encoder        ← compresse l'image en espace latent (32×32×4)
    │
    ▼
SiT Transformer           ← diffusion dans l'espace latent, conditionné sur le latent RGB
    │
    ▼
Thermal KL-VAE decoder    ← décode le latent thermal en image (256×256×1)
    │
    ▼
Image thermique (niveaux de gris)
```

Deux VAEs distincts sont utilisés : un pour le RGB (pré-entraîné, `stabilityai/sd-vae-ft-mse`), un pour le thermal (entraîné par les auteurs de ThermalGen).

### Mode de génération utilisé

On utilise le mode **unconditional par rapport au dataset** (`dataset_idx = 1000`), c'est-à-dire qu'on ne précise pas de quelle source de données vient l'image. La génération reste conditionnée sur le contenu RGB via l'encodeur.

La génération se fait par **ODE sampling** (résolution d'une équation différentielle ordinaire), contrairement au DDPM classique qui utilise des étapes de bruit/dénuit stochastiques. C'est plus rapide et déterministe.

---

## Traitement d'un shard : étape par étape

```
process_shard(input_path, output_path)
```

1. **Lecture** : charger le fichier Parquet en DataFrame pandas
2. **Extraction** : récupérer la colonne `observation.images.image` (liste de dicts `{bytes, path}`)
3. **Boucle par batch de 16 images** :
   - Décoder les bytes PNG → `PIL.Image`
   - Resize vers 256×256, normaliser vers `[-1, 1]`
   - Empiler en tensor `(B, 3, 256, 256)` et envoyer sur GPU
   - Passer dans ThermalGen → tensor thermal `(B, 1, 256, 256)` en `[-1, 1]`
   - Remettre à l'échelle `[0, 1]`, multiplier par 255, convertir en `uint8`
   - Encoder en PNG bytes → dict `{bytes, path}`
4. **Écriture** : ajouter la colonne `observation.images.thermal` au DataFrame et sauvegarder en Parquet

### Gestion des erreurs

Chaque image et chaque batch sont protégés par un `try/except` — si une image est corrompue ou si un batch échoue (OOM GPU par exemple), le reste du shard continue. Les images qui échouent restent à `None` dans la colonne thermal.

---

## Le wrapper runtime : `thermal_wrapper.py`

Ce fichier ([src/pipeline/thermal_wrapper.py](../src/pipeline/thermal_wrapper.py)) est une **version allégée** pour l'inférence en temps réel pendant un rollout du VLA.

Différences avec `thermal_pipeline.py` :
- Prend du RGB en `224×224` (résolution ImageBind) au lieu de fichiers Parquet
- Upscale à 256 pour ThermalGen, puis redescend à 224 après
- Convertit la sortie 1 canal → 3 canaux (RGB répété) pour compatibilité avec ImageBind
- Utilisé à chaque pas de temps pendant l'exécution de la politique

```
RGB (224×224) → upsample → ThermalGen → thermal (256×256×1) → downsample → repeat canaux → (224×224×3)
```

---

## Résumé du flux global

```
Dataset original (RGB)
        │
        ▼
run_thermal_pipeline.sh     ← soumet 4 jobs SLURM en parallèle
        │
        ├── Worker 0 → thermal_pipeline.py --job_id 0 --n_jobs 4
        ├── Worker 1 → thermal_pipeline.py --job_id 1 --n_jobs 4
        ├── Worker 2 → thermal_pipeline.py --job_id 2 --n_jobs 4
        └── Worker 3 → thermal_pipeline.py --job_id 3 --n_jobs 4
                │
                ▼
        Dataset augmenté (RGB + Thermal synthétique)
                │
                ▼
        Entraînement MultiSmolVLA avec modalité thermique
```
