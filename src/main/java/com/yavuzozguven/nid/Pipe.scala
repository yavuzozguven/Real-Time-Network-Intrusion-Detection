package com.yavuzozguven.nid

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.{StringType, StructType}

object Pipe {
  def save(path: String): Unit = {
    val spark = SparkSession.builder.master("local").appName("Project").getOrCreate

    val schema = new StructType().
      add("duration",StringType,true).
      add("protocol_type",StringType,true).
      add("service",StringType,true).
      add("flag",StringType,true).
      add("src_bytes",StringType,true).
      add("dst_bytes",StringType,true).
      add("land",StringType,true).
      add("wrong_fragment",StringType,true).
      add("urgent",StringType,true).
      add("hot",StringType,true).
      add("num_failed_logins",StringType,true).
      add("logged_in",StringType,true).
      add("num_compromised",StringType,true).
      add("root_shell",StringType,true).
      add("su_attempted",StringType,true).
      add("num_root",StringType,true).
      add("num_file_creations",StringType,true).
      add("num_shells",StringType,true).
      add("num_access_files",StringType,true).
      add("num_outbound_cmds",StringType,true).
      add("is_host_login",StringType,true).
      add("is_guest_login",StringType,true).
      add("count",StringType,true).
      add("srv_count",StringType,true).
      add("serror_rate",StringType,true).
      add("srv_serror_rate",StringType,true).
      add("rerror_rate",StringType,true).
      add("srv_rerror_rate",StringType,true).
      add("same_srv_rate",StringType,true).
      add("diff_srv_rate",StringType,true).
      add("srv_diff_host_rate",StringType,true).
      add("dst_host_count",StringType,true).
      add("dst_host_srv_count",StringType,true).
      add("dst_host_same_srv_rate",StringType,true).
      add("dst_host_diff_srv_rate",StringType,true).
      add("dst_host_same_src_port_rate",StringType,true).
      add("dst_host_srv_diff_host_rate",StringType,true).
      add("dst_host_serror_rate",StringType,true).
      add("dst_host_srv_serror_rate",StringType,true).
      add("dst_host_rerror_rate",StringType,true).
      add("dst_host_srv_rerror_rate",StringType,true).
      add("label",StringType,true)

    var df = spark.read.options(Map("sep"->",", "header"-> "true"))
      .schema(schema).csv(path)

    df = df.filter(df("service").equalTo("http") || df("service").equalTo("private")
      || df("service").equalTo("ecr_i") || df("service").equalTo("smtp"))


    val selected_cols = List("service","src_bytes", "dst_bytes", "wrong_fragment",
      "count", "srv_count", "diff_srv_rate",
      "dst_host_diff_srv_rate",
      "dst_host_same_src_port_rate",
      "dst_host_srv_diff_host_rate",
      "dst_host_rerror_rate","logged_in","protocol_type","flag","label")

    df = df.select(selected_cols.map(c=> col(c)):_*)
    var numeric_cols = selected_cols.filterNot(elm => elm.equals("label"))
    numeric_cols = numeric_cols.filterNot(elm => elm.equals("protocol_type"))
    numeric_cols = numeric_cols.filterNot(elm => elm.equals("service"))
    numeric_cols = numeric_cols.filterNot(elm => elm.equals("flag"))

    df = df.withColumn("label", when(col("label").equalTo("normal."), "normal.").otherwise("attack."))

    val categoric_cols = Array("protocol_type","service", "flag", "label")

    numeric_cols.foreach(r=>
      df = df.withColumn(r,df(r).cast("double"))
    )

    val strIndexer = new StringIndexer()
      .setInputCols(categoric_cols)
      .setOutputCols(categoric_cols.map(_ + "_"))


    numeric_cols = "service_" :: numeric_cols
    numeric_cols = "protocol_type_" :: numeric_cols
    numeric_cols = "flag_" :: numeric_cols

    val assembler = new VectorAssembler()
      .setInputCols(numeric_cols.toArray)
      .setOutputCol("featuress")
      .setHandleInvalid("skip")

    val scaler = new StandardScaler()
      .setInputCol("featuress")
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(strIndexer,assembler,scaler))
    pipeline.save("models/pipeline")
  }
}
