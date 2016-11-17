package com.example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}


object Main {
  def main(args: Array[String]): Unit = {
    val logFile = "zp3.csv"
    val conf = new SparkConf().setAppName("zrada-peremoga").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()


    val df = spark.read.csv(logFile)

    val toDouble = udf[Double, String]( _.toDouble)
    val mlData = df.select("_c0", "_c1").toDF("text", "label")

    val data_all = mlData.withColumn("label", toDouble(mlData("label"))).select("text", "label").cache()

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val cleaner = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("cleaned")
    val hashTF = new HashingTF().setInputCol(cleaner.getOutputCol).setOutputCol("features").setNumFeatures(11000)

    val naiveBayes = new NaiveBayes().setSmoothing(7)

    val pipeline = new Pipeline().setStages(Array(tokenizer, cleaner, hashTF, naiveBayes))

    val paramgrid = new ParamGridBuilder()
      .addGrid(hashTF.numFeatures, Array(1000, 5000, 10000, 11000, 12000, 13000))
      .addGrid(naiveBayes.smoothing, Array(1.0,2,3,4,5,6,7,8,9,10))
      .build()


    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramgrid)
      .setNumFolds(4)

    val splits = data_all.randomSplit(Array(0.7, 0.3))
    val (data_training, data_testing) = (splits(0), splits(1))

    val cvModel = cv.fit(data_training)

    val pal = cvModel.transform(data_testing)
      .select("prediction", "label")
      .collect()

      val accuracy = 1.0 * pal.count(r => r.getDouble(0) == r.getDouble(1)) / data_testing.count()
    println("accuracy: " + accuracy)

    println(cvModel.explainParams())
    println("-----------------")

    cvModel.write.overwrite().save("cvModel")
  }
}
