# MultiSmolVLA — Analyse des problèmes, métriques et résultats

## 1. Contexte et objectif

Le projet MultiSmolVLA vise à enrichir SmolVLA (un VLA — Vision-Language-Action model) avec des modalités supplémentaires (profondeur, segmentation, thermique) pour la manipulation robotique sur le benchmark LIBERO-10. L'architecture proposée est un pipeline en deux blocs :

- **Block1** : RGB → ThermalGen (génération d'image thermique) → ImageBind (encodeur multimodal, sortie 1024-dim)
- **Block2** : {RGB, depth, seg, thermal} → 4M-21 (encodeur visuel) → MLP connector → SmolVLA → actions

La baseline de référence est SmolVLA natif (sans modifications) : **60% de succès sur LIBERO-10**.

---

## 2. Résultats de toutes les évaluations

| Modèle / Stage | Succès global LIBERO-10 | Détails |
|---|---|---|
| SmolVLA natif (baseline) | **60%** | Tasks : 0/80/100/100/40/100/40/20/60/60 |
| Stage 1 (distill mean-pooled) | **0%** | MLP aligné globalement, pas par token |
| Stage 1b (+ modality dropout) | **0%** | Même problème d'alignement |
| Stage 2 LoRA (fine-tune action) | **0%** | Représentations mal alignées au départ |
| Stage 3 (joint loss, 50k steps) | **0%** (step 50k) / **4%** (final) | Amélioration marginale, trop tardive |
| Ablation no-thermal | **0%** | Confirms thermal ne contribue pas |
| Stage 1 RGB per-token (Run A) | **0%** | cos_sim=0.784, mlp_norm=59.6 ⚠️ |
| Stage 1 RGB+depth+seg (Run B) | **TBD** | cos_sim=0.779, eval à lancer |

---

## 3. Problèmes découverts et diagnostic

### 3.1 Problème principal : loss de distillation mean-poolée inefficace

**Découverte :** Après le Stage 1 (distillation MLP), les métriques d'alignement montraient :

```
mean_cosine_similarity : 0.2707   ← alignement per-token très faible
cka_score              : 0.9119   ← structure globale excellente
```

**Analyse :** La loss originale calculait le cosinus sur les tokens *moyennés* (`tokens.mean(dim=1)`) avant normalisation. Cela optimise l'alignement du vecteur global moyen, pas de chaque token spatial individuellement.

SmolVLA utilise de l'attention cross-attention *par token* : chaque token textuel attend chaque token visuel séparément. Un CKA élevé signifie que la structure relative entre tokens est préservée (bon signe), mais un cosine_similarity faible par token signifie que les directions individuelles ne correspondent pas à celles de SigLIP que SmolVLA a appris à utiliser.

**Impact :** Le MLP produit des tokens structurellement cohérents (CKA=0.91) mais directionnellement mal alignés (cos_sim=0.27) avec ce que SmolVLA attend → 0% de succès malgré un bon alignement apparent.

**Résolution :** Réécriture de la loss pour opérer sur tous les `B×196` tokens aplatis :
```python
# Avant (mean-pooled)
mlp_mean = F.normalize(mlp_tokens.mean(dim=1), dim=-1)   # (B, D)
sig_mean = F.normalize(siglip_tokens.mean(dim=1), dim=-1)
loss = 1 - (mlp_mean * sig_mean).sum(dim=-1).mean()

# Après (per-token)
B, N, D = mlp_tokens.shape
mlp_flat = F.normalize(mlp_tokens.reshape(B*N, D), dim=-1)  # (B*196, D)
sig_flat = F.normalize(siglip_proj.reshape(B*N, D), dim=-1)
loss = F.cosine_embedding_loss(mlp_flat, sig_flat, torch.ones(B*N))
```

**Résultat :** cos_sim per-token passe de **0.27 → 0.78** avec la nouvelle loss.

---

### 3.2 Problème : 4M-21 est un encodeur RGB-uniquement

**Découverte :** En inspectant le code de l'encodeur `Encoder4M`, la profondeur, la segmentation et le thermique étaient commentés :

```python
mod_dict = {
    "rgb@224": {"tensor": rgb, ...},
    # depth, seg, thermal : "ignored for now"
}
```

**Impact :** L'intégralité de Block1 (ThermalGen + ImageBind) était du calcul inutile. Les 1024-dim thermal embeddings d'ImageBind n'étaient jamais injectés dans 4M. Le thermique était généré et encodé, mais jeté silencieusement.

**Cause architecturale :** 4M-21 attend des modalités tokenisées via VQ-VAE pour la profondeur et la segmentation. Brancher une image continue directement dans 4M nécessite des adaptateurs spécifiques.

**Résolution :** Abandon de Block1 (ThermalGen + ImageBind) car son output — un vecteur thermique 1024-dim — était architecturalement incompatible avec 4M (qui attend des entrées tokenisées via VQ-VAE, pas des vecteurs continus). La modalité thermique a donc été abandonnée entièrement.

Pour depth/seg, une approche plus simple a été adoptée : un `PatchEmbedder` apprenable qui opère directement sur les images brutes (sans passer par ImageBind), ajouté comme résidu aux tokens RGB de 4M :

```python
class PatchEmbedder(nn.Module):
    PATCH_SIZE = 16
    def forward(self, x):  # (B,1,224,224) → (B,196,D)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//16, 16, W//16, 16)
        x = x.permute(0,2,4,1,3,5).reshape(B,-1,256)
        return self.proj(x)

# Dans Encoder4M.forward:
tokens = fourm_rgb_tokens           # (B, 196, D)
if use_depth:
    tokens = tokens + self.depth_embed(depth)
if use_seg:
    tokens = tokens + self.seg_embed(seg)
```

---

### 3.3 Problème : norm inflation du MLP (Stage 2)

**Découverte :** Durant le Stage 2 (fine-tuning de l'action head avec les tokens 4M), la norme des tokens MLP a explosé sans améliorer le cosine :

```
Stage 2 step 0     : mlp_norm=29,  cos_sim=0.27
Stage 2 step 50000 : mlp_norm=41,  cos_sim=0.27
```

**Analyse :** La loss d'action pousse le MLP à produire des tokens avec plus de magnitude (plus de signal pour l'action head) sans contrainte directionnelle. Le cosine ne change pas car la direction n'est pas optimisée.

**Impact :** La dot-product attention dans SmolVLA est sensible à la magnitude. Des tokens MLP de norme 41 vs SigLIP de norme 36 amplifient artificiellement l'attention portée aux tokens 4M, ce qui déstabilise le modèle.

**Observation nouveau pipeline :**
- Run A (RGB only) : mlp_norm = 11 (step 500) → 59.6 (step 20000) — croissance continue ⚠️
- Run B (RGB+depth+seg) : mlp_norm = 18 → 23.3 — croissant mais stable ✅

---

### 3.4 Problème : divergence structurelle 4M ↔ SigLIP

**Observation :** Malgré 20 000 steps avec la loss per-token, cos_sim plafonne à **~0.78** au lieu d'atteindre >0.90.

**Analyse :** 4M-21 et SigLIP sont deux encodeurs pré-entraînés sur des données et objectifs différents :
- SigLIP : contrastive image-text, optimisé pour la sémantique globale
- 4M-21 : reconstruction multi-tâche (RGB, profondeur, segmentation, etc.), optimisé pour la structure spatiale locale

Un MLP 2-couches ne peut pas transformer complètement l'espace 4M en espace SigLIP — il y a une limite structurelle. Le plateau à 0.78 représente cette limite.

**Implication pour le rapport :** L'hypothèse initiale (« aligner 4M sur SigLIP via MLP suffit pour que SmolVLA fonctionne ») est peut-être trop optimiste. La vraie question est : est-ce que 0.78 est suffisant pour que SmolVLA produce de bonnes actions ?

---

## 4. Chronologie des expériences

```
Baseline SmolVLA           : 60% succès  (référence)
Stage 1 (mean-pooled)      : 0%  succès  | cos_sim=0.27 (mean-pooled), CKA=0.91
Stage 1b (+ dropout)       : 0%  succès  | cos_sim=0.27, CKA=0.87
Stage 2 LoRA               : 0%  succès  | mlp_norm 29→41, cos_sim stable à 0.27
Stage 3 (joint, 50k steps) : 0-4% succès | action loss + distill loss
Ablation no-thermal        : 0%  succès  | confirme thermal inutile
─────────────────────────────────────────────────────────────────
Diagnostic root cause      : loss mean-poolée + 4M RGB-only + Block1 inutile
─────────────────────────────────────────────────────────────────
Run A: RGB, per-token       : cos_sim=0.784 @ 20k steps | mlp_norm=59.6 ⚠️ → eval: 0%
Run B: RGB+depth+seg        : cos_sim=0.779 @ 20k steps | mlp_norm=23.3 ✅ → eval: TBD
```

---

## 5. Métriques d'alignement — tableau récapitulatif

| Expérience | Loss type | cos_sim final | CKA | mlp_norm | siglip_norm |
|---|---|---|---|---|---|
| Stage 1 (original) | mean-pooled cosine | 0.27 | 0.91 | ~36 | 36 |
| Stage 1b dropout | mean-pooled cosine | 0.27 | 0.87 | 51.3 | 36.3 |
| Run A (RGB, per-token) | per-token cosine | **0.784** | — | 59.6 ⚠️ | 36 |
| Run B (RGB+D+S, per-token) | per-token cosine | **0.779** | — | 23.3 ✅ | 36 |

**Interprétation :**
- cos_sim 0.27 → 0.78 : amélioration ×2.9 de l'alignement directionnel per-token
- mlp_norm Run A (59.6 >> 36) : tokens suramplifiés, peut nuire à l'attention cross-attention
- mlp_norm Run B (23.3 < 36) : tokens sous-amplifiés mais direction correcte — plus sain

---

## 6. Analyse architecturale

### Pourquoi Block1 (ThermalGen + ImageBind) ne fonctionnait pas

1. **ThermalGen** génère une image thermique RGB → thermique (via CycleGAN ou similaire)
2. **ImageBind** encode cette image thermique en un vecteur 1024-dim
3. Ce vecteur **n'était jamais injecté dans 4M-21** (code commenté)
4. 4M-21 opérait **uniquement sur RGB** dans tous les stages précédents

Conclusion : Block1 représentait du calcul additionnel (~30% de temps d'entraînement) pour zéro bénéfice.

### Pourquoi le MLP seul ne suffit probablement pas

SmolVLA a été pré-entraîné avec des tokens SigLIP. Sa cross-attention a appris des patterns spécifiques à l'espace SigLIP. Remplacer SigLIP par 4M+MLP revient à "changer de langue" pour le backbone : même si l'alignement cosinus est bon (0.78), les patterns d'activation internes peuvent différer suffisamment pour que l'action head ne généralise pas.

Une solution plus robuste aurait été de **fine-tuner l'action head** avec les tokens 4M dès le départ (ce que Stage 3 tentait), mais l'alignement initial était trop faible pour que l'action head s'adapte correctement.

---

## 7. Solutions implémentées

| Problème | Solution | Statut |
|---|---|---|
| Loss mean-poolée | Per-token cosine sur B×196 tokens | ✅ Implémenté, cos_sim 0.27→0.78 |
| 4M RGB-only | PatchEmbedder depth+seg, résidus additifs | ✅ Implémenté |
| Block1 inutile | Flag `--no_block1`, skip ThermalGen+ImageBind | ✅ Implémenté |
| Freeze incorrect | `freeze_4m` cible `model.parameters()` (pas les PatchEmbedders) | ✅ Corrigé |
| Vitesse d'entraînement | bfloat16 autocast, batch_size 16, skip Block1 | ✅ ~4x plus rapide |
| norm inflation | Monitoring continu (mlp_norm vs siglip_norm) | ⚠️ Non résolu, Run B plus stable |

---

## 8. Conclusion et perspectives

### Ce qui a été validé
- La loss per-token améliore drastiquement l'alignement directionnel (0.27 → 0.78)
- Run B (RGB+depth+seg) présente une norme MLP plus stable que Run A
- L'architecture Block1 (thermal) n'apporte pas de bénéfice mesurable

### Questions ouvertes (résultats eval en attente)
- cos_sim = 0.78 est-il suffisant pour que SmolVLA produise de bonnes actions ?
- Depth+seg (Run B) améliorent-ils les performances par rapport à RGB seul (Run A) ?

### Résultat confirmé : Run A (RGB, per-token) → 0% (évalué)

**Analyse détaillée des logs d'évaluation :**
- Tous les épisodes atteignent `steps=520` (maximum autorisé) → le robot ne termine jamais une tâche
- Les actions sont dans une plage très étroite et répétitive → comportement figé / boucle
- Gripper incohérent selon les tâches : parfois toujours ouvert (tasks 2,5,8 : 80-92% open), parfois toujours fermé (tasks 1,7 : 7-20% open)
- Aucune corrélation avec la tâche → le modèle ne comprend pas les instructions

**Conclusion finale :** cos_sim=0.78 n'est pas suffisant. La limite n'est pas l'alignement cosinus mais l'incompatibilité fondamentale entre l'espace de représentation 4M et ce que SmolVLA a appris avec SigLIP. Un MLP 2-couches ne peut pas combler cet écart sans fine-tuning end-to-end du backbone.

Ce negative result est en soi une contribution scientifique : il démontre qu'un connecteur MLP simple entre deux encodeurs pré-entraînés sur des objectifs différents (reconstruction multi-tâche vs contrastif image-texte) ne suffit pas pour la substitution de modalité dans un VLA pré-entraîné.

---

*Généré le 2026-05-25 — Expériences sur cluster EPFL RCP, GPU A100-40G*
