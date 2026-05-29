# MultiSmolVLA — Stratégie complète

## Objectif de recherche

Rendre SmolVLA robuste aux **obstructions visuelles** (caméra bouchée, occlusion partielle, bruit sur le capteur) en utilisant 4M comme module de reconstruction multimodale.

**Hypothèse centrale** : quand le RGB est dégradé ou manquant, les autres modalités disponibles (depth, segmentation) permettent à 4M de reconstruire un RGB suffisamment bon pour que SmolVLA génère une action correcte.

---

## Pipeline complet

```
[Simulation LIBERO]
        ↓
   RGB + depth estimé + seg estimé
        ↓
[4M LoRA]  ←— prend toutes les modalités disponibles en entrée
        ↓
   RGB reconstruit (~20-25 dB)
        ↓
[SmolVLA]  ←— agit toujours sur un RGB reconstruit par 4M
        ↓
   Action robot
```

### Rôle de chaque composant

| Composant | Rôle |
|-----------|------|
| **Depth Anything V2 Small** | Estime depth depuis RGB en temps réel |
| **K-means (RGB+D)** | Produit pseudo-segmentation (16 clusters) |
| **DiVAE** | Tokenise/détokenise RGB (16k vocab, 14×14 tokens à 224px) |
| **VQ depth** | Tokenise depth (8k vocab, 14×14 tokens) |
| **4M-21-XL + LoRA** | Reconstruit RGB depuis toutes modalités disponibles |
| **SmolVLA** | Génère les actions depuis le RGB reconstruit |

---

## État actuel

### LoRA fine-tuning 4M (en cours)
- **But** : domain adaptation — faire générer à 4M des images style LIBERO (pas ImageNet)
- **Checkpoint** : step ~22k / 172k (12.9%)
- **Job RunAI** : `finetune-4m-lora`
- **Modalités entraînées** : depth→RGB, seg→RGB, depth+seg→RGB
- **Qualité actuelle** : PSNR ~17-19 dB (blurry, acceptable à ~50k steps)
- **Sans LoRA** : PSNR ~7 dB (hallucinations ImageNet, inutilisable)
- **Avec LoRA + CFG=2.0** : meilleur résultat actuel

### Résultats de génération
- DiVAE passthrough (upper bound à 224px) : ~18 dB
- LoRA step 22k + CFG=2.0, T=3.0 : ~18 dB
- Base 4M sans LoRA : ~7 dB (inutilisable)
- **Note** : Case A (depth→RGB) ≈ Case B (RGB+depth→RGB) car le LoRA n'a jamais vu RGB dans l'encodeur

---

## Décisions architecturales prises

### Pourquoi le LoRA ?
4M pré-entraîné sur ImageNet/COCO génère du bruit bleu/cyan sur les images LIBERO. Le LoRA (rank=8, ~14.9M params sur 4.5B) adapte le domaine de génération.

### Pourquoi pas de passthrough ?
Le passthrough (tokenize → decode sans 4M) donne ~18 dB à 224px — similaire au LoRA actuel. Mais il **ignore complètement les autres modalités** (depth, seg). Quand RGB est obstrué, le passthrough ne peut rien faire. Le 4M, lui, peut reconstruire depuis depth+seg.

### Pourquoi SmolVLA doit être fine-tuné ?
SmolVLA actuel est entraîné sur RGB parfait → performances mauvaises quand RGB est dégradé. Il faut le fine-tuner sur du RGB qualité 4M (~20 dB) pour qu'il apprenne à agir avec cette qualité.

### Convention target_mask dans 4M
`target_mask = False` → position à générer (decoder target)  
`target_mask = True` → position déjà générée / connue  
**Utiliser `GenerationSampler.maskgit_step_batched` / `guided_maskgit_step_batched`** — ne pas réimplémenter MaskGIT manuellement.

---

## Questions ouvertes (à valider empiriquement)

### Q1 : 4M est-il naturellement robuste aux entrées légèrement corrompues ?

4M a été pré-entraîné avec masking **binaire** (token présent ou absent), jamais avec des tokens bruités. On ne sait pas si :
- Depth légèrement bruité → RGB dégradé gracieusement ou catastrophiquement
- RGB partiellement obstrué (patches noirs) → depth+seg compensent bien

**Test à faire** (après LoRA ~50k steps) :
```
Test A : depth propre → 4M → RGB                   (baseline)
Test B : depth + 10% bruit token → 4M → RGB        (légère dégradation)
Test C : depth + 50% bruit token → 4M → RGB        (forte dégradation)
Test D : RGB corrompu + depth propre → 4M → RGB    (depth compense ?)
```

→ Si dégradation gracieuse : pas besoin de modifier l'entraînement LoRA  
→ Si dégradation catastrophique : ajouter corruption augmentation dans le LoRA

### Q2 : Faut-il inclure RGB dans l'encodeur 4M ?

Le LoRA actuel n'a jamais vu RGB comme entrée encodeur. Pour la pipeline complète (RGB+depth+seg → RGB), il faudrait peut-être réentraîner avec RGB dans l'encodeur + dropout par modalité.

**À décider après Q1.**

---

## Plan d'implémentation (séquentiel, validé étape par étape)

### Étape 1 — Attendre LoRA ~50k steps *(en cours)*
- Qualité attendue : PSNR ~22-25 dB
- Aucune modification de code nécessaire

### Étape 2 — Test de robustesse empirique *(Q1)*
- Script : modifier `test_4m_generation.py` pour tester des entrées corrompues
- Valide si modification du LoRA est nécessaire

### Étape 3 — (Conditionnel) Modifier entraînement LoRA
**Seulement si Q1 montre une dégradation catastrophique.**
- Ajouter RGB comme modalité encodeur possible
- Ajouter dropout par modalité (p_drop_rgb, p_drop_depth, p_drop_seg)
- Ajouter corruption token-level (remplacer X% tokens par tokens aléatoires)
- Repartir du checkpoint step 50k

### Étape 4 — Fine-tuning SmolVLA *(après LoRA validé)*
- Prendre le checkpoint SmolVLA existant
- À chaque batch : appliquer modality_dropout → 4M reconstruit RGB → SmolVLA
- SmolVLA apprend à agir sur RGB qualité 4M (~20-25 dB)
- Script à créer : `scripts/finetune_smolvla_robust.py`

### Étape 5 — Évaluation comparative
Scripts déjà prêts : `scripts/run_eval_modalities.sh`

| Mode | Ce que SmolVLA reçoit | Question |
|------|----------------------|----------|
| `rgb` | RGB parfait (passthrough) | Performance baseline |
| `rgb_depth` | RGB reconstruit par 4M depuis depth | Dégradation ? |
| `rgb_depth_seg` | RGB reconstruit par 4M depuis depth+seg | Meilleur ? |
| `p_drop_rgb=1.0` | RGB généré depuis depth+seg seulement | Obstruction totale |

**Métrique principale** : taux de succès sur LIBERO-90 (10 épisodes par tâche).

---

## Paramètres clés

```python
# LoRA
lora_rank = 8
lora_scale = 1.0
lora_target = "attention"   # Attention + CrossAttention encoder + decoder

# 4M MaskGIT (inférence)
maskgit_steps = 8
temperature = 3.0           # Officiel 4M pour X2RGB
cfg_scale = 2.0             # Classifier-Free Guidance
divae_steps = 50            # DDIM steps pour DiVAE

# Depth estimator
model = "depth-anything/Depth-Anything-V2-Small-hf"

# Seg estimator
n_clusters = 16             # K-means sur (R,G,B,D)

# Chemins
lora_ckpt = "/scratch/finetune_4m_lora_rgb/lora_step0XXXXX.pt"
hf_repo = "tjga98/4m_fine_tune_libero"
```

---

## Fichiers clés

| Fichier | Rôle |
|---------|------|
| `scripts/finetune_4m_libero.py` | Entraînement LoRA 4M sur LIBERO |
| `scripts/test_4m_generation.py` | Test qualité génération RGB |
| `scripts/test_divae_roundtrip.py` | Test qualité tokenizer seul |
| `src/pipeline/fourm_image_processor_v2.py` | Module 4M dans pipeline SmolVLA |
| `src/pipeline/realtime_modality_estimator.py` | Depth + seg depuis RGB |
| `src/pipeline/modality_dropout.py` | Corruption curriculum par modalité |
| `scripts/eval_native_smolvla_v2.py` | Évaluation SmolVLA sur LIBERO |
| `scripts/run_eval_modalities.sh` | Lance les 3 modes d'éval |
| `utils/parquet_dataset.py` | Dataset LIBERO parquet |

---

## Contribution scientifique visée

> En fine-tunant SmolVLA avec un module de reconstruction multimodale basé sur 4M, on maintient **X% du taux de succès** de la politique originale malgré une obstruction totale du RGB, grâce à la compensation par depth et segmentation estimés en temps réel.
