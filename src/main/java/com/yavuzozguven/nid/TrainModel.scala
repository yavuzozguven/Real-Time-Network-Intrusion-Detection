package com.yavuzozguven.nid

import com.linkedin.relevance.isolationforest.IsolationForest
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.sql.SparkSession

object TrainModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Project").getOrCreate

    val data = PreProcess.getDataFrame("kddcup.data_10_percent.gz")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)


    val dt = new DecisionTreeClassifier()
      .setMaxDepth(30)


    val rf = new RandomForestClassifier()
      .setMaxDepth(30)
      .setNumTrees(30)


    val layers = Array[Int](14, 5, 4, 2)
    val mlp = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)


    val contamination = 0.01
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setScoreCol("outlierScore")
      .setContamination(contamination)
      .setContaminationError(0.01 * contamination)
      .setRandomSeed(2018)


    val algorithm_arr = Array(lr,dt,rf,mlp,isolationForest)

    algorithm_arr.foreach{item=>
      val t1 = System.nanoTime

      val model = item.fit(data)

      model.write.overwrite().save("models/"+item)

      val duration = (System.nanoTime - t1) / 1e9d
      println(item.toString+s"execution time in ${duration}")

    }

  }
}
