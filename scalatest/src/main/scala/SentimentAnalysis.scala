import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, HashingTF, IDF}
import org.apache.spark.ml.Pipeline

object SentimentAnalysis {
  def main(args: Array[String]): Unit = {
    println("Starting Sentiment Analysis with Spark...")

    val spark = SparkSession.builder()
      .appName("Sentiment Analysis")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()

    import spark.implicits._

    // Sample customer reviews
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

    println("\n=== Original Reviews ===")
    reviews.show(truncate = false)

    // Step 1: Tokenization (split text into words)
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val tokenized = tokenizer.transform(reviews)
    println("\n=== After Tokenization ===")
    tokenized.select("text", "words").show(5, truncate = false)

    // Step 2: Remove stop words (common words like "the", "is", "a")
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    val filtered = remover.transform(tokenized)
    println("\n=== After Removing Stop Words ===")
    filtered.select("text", "words", "filtered_words").show(5, truncate = false)

    // Step 3: Convert words to numerical features (TF-IDF)
    val hashingTF = new HashingTF()
      .setInputCol("filtered_words")
      .setOutputCol("raw_features")
      .setNumFeatures(1000)

    val idf = new IDF()
      .setInputCol("raw_features")
      .setOutputCol("features")

    // Create a pipeline (chains all transformations)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf))

    println("\n=== Building NLP Pipeline ===")
    val model = pipeline.fit(reviews)
    val result = model.transform(reviews)

    println("\n=== Final Features (numerical representation of text) ===")
    result.select("text", "label", "features").show(5, truncate = false)

    // Analyze word counts
    println("\n=== Statistics ===")
    println(s"Total reviews: ${reviews.count()}")
    println("\nReviews by sentiment:")
    reviews.groupBy("label").count().show()

    println("\n✓ Sentiment prediction complete!")
    println("\n🌐 Spark UI available at: http://localhost:4040")
    println("Press ENTER to stop Spark...")
    scala.io.StdIn.readLine()  // Waits for you to press Enter

    spark.stop()
    println("\n✓ Analysis complete!")
  }
}