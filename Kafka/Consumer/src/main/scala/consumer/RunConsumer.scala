package consumer

object RunConsumer {
  def main(args: Array[String]): Unit = {
    val consumer = new TwitterConsumer()
    consumer.run()
  }
}
