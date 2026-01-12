# scalatest (Spark Sentiment Demo)

This is a tiny Scala + SBT project that demonstrates basic NLP feature engineering and a simple sentiment classifier using **Apache Spark MLlib**.

## What’s in here?

- `src/main/scala/SentimentAnalysis.scala`
  - A **feature extraction** demo: tokenization → stopword removal → TF‑IDF.
  - Prints the resulting numeric feature vectors.

- `src/main/scala/SentimentClassifier.scala`
  - A **real classifier** demo built on top of the same features.
  - Trains a multiclass **Logistic Regression** model (positive/negative/neutral), evaluates it, and predicts sentiment for a few new example sentences.

## Requirements

- Java (JDK 11 works with Spark 3.5.x)
- SBT (this repo pins it via `project/build.properties`)

Dependencies are handled by SBT via `build.sbt` (Spark Core/SQL/MLlib).

## Run

This project contains **two main classes**, so `sbt run` will prompt you to choose one.

### Run the feature extraction demo

```bash
cd /home/xps15/IdeaProjects/scalatest
sbt "runMain SentimentAnalysis"
```

### Run the classifier demo

```bash
cd /home/xps15/IdeaProjects/scalatest
sbt "runMain SentimentClassifier"
```

### Alternative: choose interactively

```bash
cd /home/xps15/IdeaProjects/scalatest
sbt run
```

## Notes / gotchas

- The classifier is trained on a **very small sample dataset** (10 rows). Metrics can vary a lot depending on the train/test split.
- Spark runs locally via `master("local[*]")`.
- While the app is running, the Spark UI is usually available at: http://localhost:4040

## Project structure (why it looks more complex than Python)

- `build.sbt`: build definition (similar role to `pyproject.toml` / `requirements.txt`)
- `project/`: SBT’s own build configuration (includes `project/build.properties` for the sbt version)
- `src/main/scala/`: application code
- `src/test/scala/`: tests (empty right now)
- `target/`: generated output (compiled `.class` files, incremental compilation caches, resolved dependency metadata)

It’s normal to delete `target/` if you want a clean rebuild — it will be regenerated automatically.

