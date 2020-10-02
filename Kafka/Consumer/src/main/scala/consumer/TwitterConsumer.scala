package consumer

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.kafka010.KafkaUtils
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.{Seconds, StreamingContext}


object TwitterConsumer {

  def createStream(ssc: StreamingContext, bootstrapServers: String, groupId: String, topic: String):
  InputDStream[ConsumerRecord[String, String]] = {

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> bootstrapServers,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> groupId,
      "auto.offset.reset" -> "latest",
      "enable.auto.commit" -> "false",
      "max.poll.records" -> "100"
    )
    val topics = Array(topic)

    val twitterStream = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )

    // This returns a Stream of ConsumerRecords
    // Each ConsumerRecord contains a key and a value
    twitterStream
  }

  def extractText(jsonString: String): String = {
    null
  }

  def run(): Unit = {
    // Initialize streaming context
    print("Creating streaming context")
    val conf = new SparkConf().setMaster("local[*]").setAppName("TwitterConsumer")
    val ssc = new StreamingContext(conf, Seconds(1))

    val bootstrapServers = "localhost:9092"
    val groupId = "twitter_group"
    val topic = "twitter_topic"

    print("Streaming data from Kafka")
    val tweetStream = createStream(ssc, bootstrapServers, groupId, topic)
    val keysAndValues = tweetStream.map(record => (record.key, record.value))
    keysAndValues.print()

    val checkpointDir = "C:/checkpoint/"
    ssc.checkpoint(checkpointDir)
    ssc.start()
    ssc.awaitTermination()
  }

  def main(args: Array[String]): Unit = {
    run()
  }
}
