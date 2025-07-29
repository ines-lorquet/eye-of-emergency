# 🧠 Eye of Emergency – Classification de Tweets de Catastrophes

## 📌 Contexte

Les réseaux sociaux, et notamment Twitter, sont devenus des sources critiques d’informations en temps réel lors de catastrophes. Cependant, la désinformation y est aussi très présente.  
**Objectif :** développer un modèle de classification de tweets capable de distinguer les messages liés à des catastrophes réelles de ceux qui ne le sont pas, pour faciliter la diffusion d’alertes fiables.

---

## 🗃️ Jeu de données

Le dataset "Disaster Tweets" contient les colonnes suivantes :

| Colonne   | Description                                          |
|-----------|------------------------------------------------------|
| `id`      | Identifiant unique du tweet                          |
| `text`    | Contenu textuel                                      |
| `location`| Localisation du tweet (souvent bruitée)              |
| `keyword` | Mot-clé assigné lors de la collecte                  |
| `target`  | 1 = catastrophe réelle, 0 = non-catastrophe          |

Des étapes de nettoyage ont été appliquées : suppression des doublons, des valeurs nulles et des colonnes non informatives (`id`, `location`).

---

## 🔍 Analyse exploratoire des données (EDA)

- Vérification de la distribution des classes (`target`)
- Analyse des longueurs des tweets (caractères, mots)
- Analyse lexicale : mots fréquents, comparaison des classes
- Visualisations :
  - Nuages de mots
  - Barplots des mots dominants
  - Distribution des longueurs
  - Fréquences des keywords
- Observation de différences lexicales notables entre les classes

---

## 📚 Veille sur le NLP

### 1. Text Mining vs NLP
- **Text Mining** : extraction d'informations exploitables à partir de texte brut (fréquence, cooccurrence, etc.)
- **NLP** : ensemble de techniques pour faire comprendre, manipuler ou générer du langage naturel par une machine.
- Le NLP englobe le text mining, mais vise des traitements plus intelligents (compréhension, sens).

### 2. Sous-domaines du NLP
- **Analyse de sentiments** : détecter l’opinion (ex : avis positifs/négatifs)
- **NER (Named Entity Recognition)** : extraire lieux, dates, personnes (ex : “Paris”, “ONU”)
- **POS tagging** : identifier la fonction grammaticale des mots (ex : "ferme" = nom ou verbe)

### 3. Applications concrètes
- Détection de fake news, alertes automatiques, support client (chatbots), modération de contenu, analyse de tendances (Twitter), traduction automatique.

### 4. Stop-words
- Mots très fréquents sans intérêt pour la classification (ex : "the", "and")
- Supprimés pour éviter le bruit et améliorer la représentativité du texte

### 5. Ponctuation et caractères spéciaux
- Nettoyés (ex : URLs, mentions, hashtags, émojis) pour obtenir un texte exploitable
- On garde parfois les hashtags s’ils sont informatifs (ex : "#earthquake")

### 6. Tokens et N-grams
- **Token** : mot ou unité du texte
- **N-gram** : suite de N mots (ex : "feu de forêt")
- Permettent de capturer le **contexte** (ex : "suicide bomber" plus informatif que "suicide" seul)

### 7. Stemming vs Lemmatization
- **Stemming** : découpe brutale (ex : "flooded" → "flood")
- **Lemmatisation** : conversion linguistique précise (ex : "was" → "be")
- Le stemming est plus rapide, la lemmatisation plus précise

### 8. Représentation vectorielle
- **Bag-of-Words (BoW)** : représentation par fréquences brutes
- **TF-IDF** : fréquence pondérée par la rareté (met en valeur les mots spécifiques)
- Ces vecteurs sont utilisés comme entrée dans les modèles de ML

---

## 🧹 Prétraitement des textes

Pipeline appliqué :

1. Conversion en minuscules
2. Suppression :
   - URL, mentions (@), hashtags
   - Ponctuation, chiffres, caractères spéciaux
3. Suppression des stop-words
4. Lemmatisation
5. Vectorisation : TF-IDF

---

## 🤖 Modélisation

### Modèles testés :
- Régression Logistique
- Arbre de Décision (y compris implémentation manuelle)
- Random Forest
- XGBoost
- SVM

### Enjeux :
- Comparer bagging (Random Forest) et boosting (XGBoost)
- Optimiser les hyperparamètres avec **GridSearchCV**
- Tester différents pipelines de nettoyage

---

## 📊 Évaluation

- Métriques : Accuracy, Recall, F1-score
- Objectif prioritaire : **maximiser le F1-score**
- Visualisation : matrice de confusion, classification report

---

## ✅ Résultats & Conclusion

- Meilleur modèle : [à compléter avec ton résultat]
- Justification du choix : [ex : bon compromis précision/recall]
- Limites du modèle : tweets ambigus, ironie, usages figuratifs
- Pistes d'amélioration :
  - Word embeddings (Word2Vec, GloVe)
  - Modèles contextuels (BERT, RoBERTa)

---

## 🗂️ Contenu du repository

- `README.md` : ce fichier
- `notebook.ipynb` : code structuré et commenté
- `presentation.pdf` : slides de soutenance
- Dossier `data/` : fichiers CSV
- Dossier `outputs/` : figures et résultats du modèle

---

## 🔗 Références clés

- [nltk.org/book](https://www.nltk.org/book/)
- [spacy.io](https://spacy.io)
- [Bagging vs Boosting – Quantdare](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
- [GridSearch explained](https://www.lovelyanalytics.com/2017/10/16/grid-search/)
- [Text analytics – Datacamp](https://www.datacamp.com/tutorial/text-analytics-beginners-nltk)

