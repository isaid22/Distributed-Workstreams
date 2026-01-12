import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, IndexToString, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * A real sentiment classifier example using Spark ML.
  *
  * Keeps the original SentimentAnalysis.scala intact by providing a separate entry point.
  */
object SentimentClassifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Sentiment Classifier")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()

    import spark.implicits._

    val reviews = Seq(
      (1, "This product is absolutely amazing! Love it so much!", "positive"),
      (2, "Terrible quality. Broke after one day. Complete waste of money!", "negative"),
      (3, "It's okay, nothing special but does what it says", "neutral"),
      (4, "Best purchase ever! Highly recommend to everyone!", "positive"),
      (5, "Very disappointed. Poor quality and bad customer service.", "negative"),
      (6, "Good value for the price. Works well.", "positive"),
      (7, "Don't waste your money on this garbage!", "negative"),
      (8, "Decent product. Not great but not terrible either.", "neutral"),
      (9, "Exceeded my expectations! Will buy again!", "positive"),
      (10, "Arrived damaged and seller won't respond to messages.", "negative")
    ).toDF("id", "text", "label")

    // Features: text -> TF-IDF
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val hashingTF = new HashingTF()
      .setInputCol("filtered_words")
      .setOutputCol("raw_features")
      .setNumFeatures(2000)
    val idf = new IDF().setInputCol("raw_features").setOutputCol("features")

    // Labels: string -> double
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
      .setHandleInvalid("error")

    val lr = new LogisticRegression()
      .setLabelCol("labelIndex")
      .setFeaturesCol("features")
      .setMaxIter(100)
      .setRegParam(0.01)

    // Decode prediction back to original label strings.
    // We fit a tiny StringIndexer here just to capture consistent label ordering.
    val labelOrdering = labelIndexer.fit(reviews)
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelOrdering.labels)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, labelIndexer, lr, labelConverter))

    val Array(train, test) = reviews.randomSplit(Array(0.8, 0.2), seed = 42L)

    println(s"Training rows: ${train.count()}, Test rows: ${test.count()}")

    val model = pipeline.fit(train)
    val predictions = model.transform(test)

    println("\n=== Predictions (test set) ===")
    predictions.select("id", "text", "label", "predictedLabel", "probability").show(truncate = false)

    val accuracy = new MulticlassClassificationEvaluator()
      .setLabelCol("labelIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
      .evaluate(predictions)

    val f1 = new MulticlassClassificationEvaluator()
      .setLabelCol("labelIndex")
      .setPredictionCol("prediction")
      .setMetricName("f1")
      .evaluate(predictions)

    println(f"\nAccuracy: ${accuracy}%.4f")
    println(f"F1 (weighted): ${f1}%.4f")

    val examples = Seq(
      (1001, "I love this! Super helpful and great quality."),
      (1002, "Awful experience. It arrived broken and support was useless."),
      (1003, "It's fine. Does the job, but nothing impressive.")
    ).toDF("id", "text")

    println("\n=== Example predictions (new text) ===")
    model.transform(examples).select("id", "text", "predictedLabel", "probability").show(truncate = false)

    println("\nSpark UI (while the app is running): http://localhost:4040")
    spark.stop()
  }
}

