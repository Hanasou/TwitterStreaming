package consumer

import java.io.NotSerializableException
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkException}
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.kafka010.KafkaUtils
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.json4s._
import org.json4s.jackson.JsonMethods._


class TwitterConsumer extends Serializable{

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

  case class User(verified: Boolean, followers_count: Int, friends_count: Int) extends Serializable
  case class Message(id_str: String, text: String, user: User) extends Serializable

  /**
   * Takes the fields that I'm interested in and puts it into a Message object
   * @param value a JSON string
   * @return
   */
  def extractFields(value: String): Message = {
    val jsonValue = parse(value)
    implicit val formats = DefaultFormats
    val msg = jsonValue.extract[Message]
    msg
  }

  /**
   * Get the average number of followers for people tweeting from producer
   * @param stream Stream of Message objects
   */
  def averageFollowers(stream: DStream[Message]) = {
    // Just get a stream of the followers
    val followers = stream.map(message => message.user.followers_count)
    // There might be multiple processes adding to this number
    // Atomic means that it's thread safe
    val totalUsers = new AtomicLong(0)
    val totalFollowers = new AtomicLong(0)
    followers.foreachRDD((rdd, time) => {
      val count = rdd.count()
      if (count > 0) {
        totalUsers.getAndAdd(count)
        totalFollowers.getAndAdd(rdd.reduce((x,y) => x + y).toLong)
        val average = totalFollowers.get() / totalUsers.get()
        println("Total users:", totalUsers.get())
        println("Average followers:", average)
      }
    })
  }

  /**
   * Get the number of people that are verified tweeting from
   * @param stream Stream of Message objects
   */
  def countVerified(stream: DStream[Message]): DStream[(Boolean, Long)] = {
    val verifiedStream = stream.map(message => message.user.verified)
    val countVerified = verifiedStream.countByValueAndWindow(Seconds(300), Seconds(1))
    countVerified
  }

  /**
   * Get the most popular people mentioned from producer
   * @param stream Stream of Message objects
   */
  def popularMentions(stream: DStream[String]): DStream[(String, Int)] = {
    // extract the right fields
    val extracted = stream.map(extractFields)
    // get the text of the message
    val textStream = extracted.map(message => message.text)
    // flatmap text into words
    val wordStream = textStream.flatMap(text => text.split(" "))
    // Filter mentions
    val mentions = wordStream.filter(word => word.startsWith("@"))
    // Map them into key/value pairs
    val keyValues = mentions.map(mention => (mention, 1))
    // Reduce by key and window
    val counts = keyValues.reduceByKeyAndWindow((x,y) => x + y, (x,y) => x + y, Seconds(300), Seconds(1))
    // Sort results
    val sorted = counts.transform(rdd => rdd.sortBy(x => x._2, ascending=false))
    sorted
  }


  def run(): Unit = {
    // Initialize streaming context
    println("Creating streaming context")
    val conf = new SparkConf().setMaster("local[*]").setAppName("TwitterConsumer")
    val ssc = new StreamingContext(conf, Seconds(1))

    // Get rid of log spam
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val bootstrapServers = "localhost:9092"
    val groupId = "twitter_group_2"
    val topic = "twitter_topic_3"

    println("Streaming data from Kafka")
    val tweetStream = createStream(ssc, bootstrapServers, groupId, topic)
    val values = tweetStream.map(record => record.value)

    // Making some other streams here
    val popMentions = popularMentions(values)
    //val verifiedCounts = countVerified(values)

    // Print the relevant stuff

    popMentions.print()
    //verifiedCounts.print()
    // The prints are inside this function
    // Maybe if I used mapWithState then I can do this the same way as the other two
    //averageFollowers(extracted)


    val checkpointDir = "C:/checkpoint/"
    ssc.checkpoint(checkpointDir)
    ssc.start()
    ssc.awaitTermination()
  }
}
