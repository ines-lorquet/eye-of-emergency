
# Documentation – Classe `TextMining`

## Objectif
Pipeline NLP complet pour nettoyer et vectoriser du texte (tweets, posts, etc.).

## Initialisation
```python
tm = TextMining(df, text_column="text", language="english")
```
**Paramètres :**
- **`df`** *(pd.DataFrame)* : Données source.
- **`text_column`** *(str)* : Colonne texte à traiter *(par défaut `"text"`)*.
- **`language`** *(str)* : Langue pour stopwords et stemming *(par défaut `"english"`)*.

---

## Méthodes de nettoyage
### `lowercase()`
Convertit toutes les colonnes texte en minuscules.
```python
tm.lowercase()
```

### `extract_target_char(char, new_column)`
Extrait les mots commençant par un caractère spécifique (`#`, `@`, etc.) et les supprime du texte original.
```python
tm.extract_target_char("#", "hashtags")
```

### `extract_url(url_column="urls")`
Extrait toutes les URLs présentes et les supprime du texte original.
```python
tm.extract_url()
```

### `clean_regex()`
Supprime tout caractère non alphanumérique.
```python
tm.clean_regex()
```

---

## Pipeline lexical
### `tokenize()`
Tokenise le texte en une liste de mots, stockée dans `tokens`.
```python
tm.tokenize()
```

### `remove_stopwords()`
Supprime les stopwords (définis par NLTK).
```python
tm.remove_stopwords()
```

### `apply_stemmer()`
Applique un stemming sur les tokens (Snowball).
```python
tm.apply_stemmer()
```

### `apply_lemmatizer()`
Applique une lemmatisation (WordNet).
```python
tm.apply_lemmatizer()
```

---

## Vectorisation
### `vectorize(mode="bow"|"tfidf", new_column="vector")`
Crée une représentation numérique du texte :
- `"bow"` → Bag of Words
- `"tfidf"` → TF-IDF  
Les vecteurs sont stockés dans une nouvelle colonne.
```python
tm.vectorize(mode="tfidf")
```

---

## Export CSV
### `export_csv(name=None)`
Exporte le DataFrame final vers `data/` :
- Nom par défaut → `export_YYYY_MM_DD_HH_MM_SS.csv`
- Retourne le chemin complet du fichier.
```python
csv_path = tm.export_csv()
print(csv_path)
```

---

## Sortie finale
### `get_df()`
Retourne le DataFrame traité.
```python
df_final = tm.get_df()
```

---

## Exemple complet d’utilisation
```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

tm = TextMining(data_train)
tm.lowercase()\
  .extract_target_char("#", "hashtags")\
  .extract_target_char("@", "mentions")\
  .extract_url()\
  .clean_regex()\
  .tokenize()\
  .remove_stopwords()\
  .apply_lemmatizer()\
  .vectorize(mode="tfidf")

# Export CSV dans le dossier "data"
csv_path = tm.export_csv()
print(f"Fichier exporté : {csv_path}")

# Récupérer le DataFrame final
df_final = tm.get_df()
print(df_final.head())
```
