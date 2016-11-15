package com.example

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionWithLBFGS, NaiveBayes}
import org.apache.spark.ml.feature.{StopWordsRemover, HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Variance, Gini, Entropy}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, SparkContext}

object Main {
  def main(args: Array[String]): Unit = {
    val logFile = "/Users/andrey/dev/zp3.csv"
    val conf = new SparkConf().setAppName("zrada-peremoga").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()

    import spark.implicits._


    val df = spark.read.csv(logFile)

    val mlData = df.select("_c0", "_c1").toDF("text", "label")

    val wordsData = new Tokenizer().setInputCol("text").setOutputCol("words").transform(mlData)
    val cleaned = new StopWordsRemover().setInputCol("words").setOutputCol("cleaned").transform(wordsData)
    cleaned.show(30)
    val hashTF = new HashingTF().setInputCol("cleaned").setOutputCol("features").setNumFeatures(11000)
    val featureData = hashTF.transform(cleaned)

    val toDouble = udf[Double, String]( _.toDouble)
    val data_all = featureData.withColumn("label", toDouble(featureData("label"))).select("features", "label")


    val data_points = data_all.map(row => LabeledPoint(row.getDouble(1), Vectors.dense(row.getAs[Vector](0).toArray))).rdd
    val splits = data_points.randomSplit(Array(0.7, 0.3))
    val (data_training, data_testing) = (splits(0), splits(1))
    data_training.cache()


    val model = NaiveBayes.train(data_training, 7)   //0.7277

    //val model = DecisionTree.train(data_training, Algo.Classification, Entropy.instance, 3)
   // val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data_training)
    val predictionAndLabel = data_testing.map{ point =>
      (model.predict(point.features), point.label)
    }

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / data_testing.count()

    println("accuracy: " + accuracy)

    //model.save(sc, "/Users/andrey/dev/zp3model")
  }
}
