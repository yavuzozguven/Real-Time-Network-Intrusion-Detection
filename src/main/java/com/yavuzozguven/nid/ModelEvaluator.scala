package com.yavuzozguven.nid

import com.linkedin.relevance.isolationforest.IsolationForestModel
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, LogisticRegressionModel, MultilayerPerceptronClassificationModel, OneVsRestModel, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.sql.{ SparkSession}
import org.apache.spark.sql.functions.{col, when}

object ModelEvaluator {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Project").getOrCreate

    val data = PreProcess.getDataFrame("corrected.gz")

    val lr = LogisticRegressionModel.load("models/logreg*")
    val dt = DecisionTreeClassificationModel.load("models/dt*")
    val rf = RandomForestClassificationModel.load("models/rf*")
    val mlp = MultilayerPerceptronClassificationModel.load("models/mlp*")
    val iso = IsolationForestModel.load("models/iso*")

    val arr = Seq(lr,dt,rf,mlp,iso)

    arr.foreach{item=>
      val t1 = System.nanoTime
      val frame = item.transform(data)
      val duration = (System.nanoTime - t1) / 1e9d

      val evaluator = new BinaryClassificationEvaluator()
      if(item.toString().matches("isolation"))
        evaluator.setMetricName("outlierScore")

      val roc = evaluator.getMetrics(frame).roc()

      spark.createDataFrame(roc).coalesce(1).write
        .option("header","true").option("sep",",").mode("overwrite").csv("scores/roc"+item.toString())

      frame.withColumn("label", when(col("label").>(0), 1).otherwise(0))
        .select("label","prediction").coalesce(1).write
        .option("header","true").option("sep",",").mode("overwrite").csv("scores/predslabels"+item.toString())

      println(item + s" transforms in ${duration}")
    }

  }

}
