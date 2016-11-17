package com.example

import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


object ModelLoad {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("zrada-peremoga").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()

    val cv = CrossValidatorModel.load("cvModel")

    for (ln <- Source.stdin.getLines){
      val training = spark.createDataFrame(Seq(
        (0d, ln)
      )).toDF("label", "text")

      val row = cv.transform(training).select("prediction").collect().head

      val lb = if(row.getDouble(0)==1 ) "перемога" else "зрада" //win or betray
      println(row.getDouble(0) + " =" + lb)
    }


  }

}
