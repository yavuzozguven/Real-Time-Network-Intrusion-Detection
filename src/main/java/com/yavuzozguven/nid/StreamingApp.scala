package com.yavuzozguven.nid

import java.net.InetAddress
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import com.mongodb.spark.MongoSpark
import org.apache.spark.ml.classification.{LogisticRegressionModel}
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions}

object StreamingApp {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.master("spark://<ipaddress>:7077")
      .config("spark.mongodb.output.uri","mongodb://<ipaddress>/nid.data")
      .appName("Project").getOrCreate

    val frame = spark.readStream.format("kafka")
      .option("kafka.bootstrap.servers", "<ipaddress>:9092")
      .option("subscribe", "test-first").load()

    val schema = new StructType().
      add("protocol_type",StringType,true).
      add("service",StringType,true).
      add("flag",StringType,true).
      add("src_bytes",StringType,true).
      add("dst_bytes",StringType,true).
      add("wrong_fragment",StringType,true).
      add("count",StringType,true).
      add("srv_count",StringType,true).
      add("diff_srv_rate",StringType,true).
      add("dst_host_diff_srv_rate",StringType,true).
      add("dst_host_same_src_port_rate",StringType,true).
      add("dst_host_srv_diff_host_rate",StringType,true).
      add("dst_host_rerror_rate",StringType,true).
      add("logged_in",StringType,true).
      add("label",StringType,true)


    val rowFrame = frame.selectExpr("CAST(value AS STRING)")

    val df = rowFrame.select(functions.from_json(rowFrame.col("value"), schema).as("data")).select("data.*")


    val localhost : InetAddress = InetAddress.getLocalHost
    val localIpAddress: String = localhost.getHostAddress
    val collectionName = args(0)

    //.trigger(Trigger.ProcessingTime(5000))
    df.writeStream.outputMode("append").format("console")
      .foreachBatch{(batchDF: DataFrame,batchId: Long) =>
        if(!batchDF.isEmpty){

          val t1 = System.nanoTime()
          val local_time = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd_HHmmss"))

          val data = PreProcess.process(batchDF)

          val model = LogisticRegressionModel.load("models/logreg*")

          val result = model.transform(data)
          var predictionAndLabels = result.select("prediction", "label")


          predictionAndLabels = predictionAndLabels.withColumn("localip",functions.lit(localIpAddress))
          predictionAndLabels = predictionAndLabels.withColumn("localtime",functions.lit(local_time))
          val duration = (System.nanoTime() -t1) / 1e9d
          predictionAndLabels = predictionAndLabels.withColumn("runtime",functions.lit(duration))
          MongoSpark.write(predictionAndLabels).option("collection","data"+collectionName).mode("append").save()


        }
      }.start().awaitTermination()


  }
}
