package app

import org.apache.spark.streaming._
import org.apache.spark.streaming.twitter._
import app.Utilities._
import org.apache.spark.streaming.dstream.DStream

object TwitterStream {

  def popularHashtags(text: DStream[String]): DStream[(String, Int)] = {
    // flatmap the text of the tweets into words
    val words = text.flatMap(text => text.split(" "))
    // filter out the hashtags
    val hashtags = words.filter(word => word.startsWith("#"))
    // map the words to a key/value pair
    val hashtagValues = hashtags.map(hashtag => (hashtag, 1))
    // reduce the words to get the counts
    val hashtagCounts = hashtagValues.reduceByKeyAndWindow((x,y) => x + y, (x,y) => x - y, Seconds(300), Seconds(1))
    // sort the results
    val sortedResults = hashtagCounts.transform(rdd => rdd.sortBy(x => x._2, false))

    sortedResults
  }

  def popularMentions(text: DStream[String]): DStream[(String, Int)] = {
    // flatmap the text of the tweets into words
    val words = text.flatMap(text => text.split(" "))
    // filter out the mentions
    val mentions = words.filter(word => word.startsWith("@"))
    // map them to key/value pairs
    val mentionValues = mentions.map(mention => (mention, 1))
    // reduce by key and window
    val mentionCounts = mentionValues.reduceByKeyAndWindow((x,y) => x + y, (x,y) => x - y, Seconds(300), Seconds(1))
    // sort results
    val sortedResults = mentionCounts.transform(rdd => rdd.sortBy(x => x._2, false))

    sortedResults
  }

  def main(args: Array[String]){

    // Configure twitter credentials
    setupTwitter()

    /*
    Configure streaming context
    Use all cores in local CPU
    Name of app is TwitterStream
    Batch size of one second
     */
    val ssc = new StreamingContext("local[*]", "TwitterStream", Seconds(1))

    // Get rid of log spam
    setupLogging()

    // Create DStream
    val tweets = TwitterUtils.createStream(ssc, None)

    // Get English Tweets
    val english = tweets.filter(status => status.getLang() == "en")

    // This is where we map the status to all the information we want
    // We'll use this section to test out the Status APIs
    val text = english.map(status => status.getText())

    // Get the most popular words
    val popHashtags: DStream[(String,Int)] = popularHashtags(text)
    // Get the most popular mentions
    val popMentions: DStream[(String,Int)] = popularMentions(text)

    // Print the first 10 statuses to see what they contain
    text.print
    popHashtags.print
    popMentions.print

    ssc.checkpoint("C:/checkpoint/")
    ssc.start()
    ssc.awaitTermination()
  }
}