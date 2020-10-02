package producer;

import com.google.common.collect.Lists;
import com.twitter.hbc.ClientBuilder;
import com.twitter.hbc.core.Client;
import com.twitter.hbc.core.Constants;
import com.twitter.hbc.core.Hosts;
import com.twitter.hbc.core.HttpHosts;
import com.twitter.hbc.core.endpoint.StatusesFilterEndpoint;
import com.twitter.hbc.core.processor.StringDelimitedProcessor;
import com.twitter.hbc.httpclient.auth.Authentication;
import com.twitter.hbc.httpclient.auth.OAuth1;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class TwitterProducer {
    private static Logger logger = LoggerFactory.getLogger(TwitterProducer.class.getName());
    private static final String BOOTSTRAP_SERVERS = "localhost:9092";
    private static final String TOPIC = "twitter_topic";

    private static Map<String, String> getTwitterCredentials(String path) {
        Map<String, String> creds = new HashMap<String, String>();
        try {
            File file = new File(path);
            Scanner reader = new Scanner(file);
            while (reader.hasNextLine()) {
                String[] line = reader.nextLine().split(" ");
                String label = line[0];
                String cred = line[1];
                creds.put(label, cred);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return creds;
    }

    private static KafkaProducer<String, String> createProducer() {
        Properties properties = new Properties();
        properties.setProperty(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
        properties.setProperty(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        properties.setProperty(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        // Safe producer settings
        properties.setProperty(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, "true");
        properties.setProperty(ProducerConfig.ACKS_CONFIG, "all");
        properties.setProperty(ProducerConfig.RETRIES_CONFIG, Integer.toString(Integer.MAX_VALUE));
        properties.setProperty(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, "5");

        // High throughput settings
        properties.setProperty(ProducerConfig.COMPRESSION_TYPE_CONFIG, "snappy");
        properties.setProperty(ProducerConfig.LINGER_MS_CONFIG, "20");
        properties.setProperty(ProducerConfig.BATCH_SIZE_CONFIG, Integer.toString(32 * 1024));

        return new KafkaProducer<String, String>(properties);
    }

    private static Client createTwitterClient(BlockingQueue<String> msgQueue) {
        Map<String, String> creds = getTwitterCredentials("twitter.txt");
        String consumerKey = creds.get("consumerKey");
        String consumerSecret = creds.get("consumerSecret");
        String accessToken = creds.get("accessToken");
        String accessTokenSecret = creds.get("accessTokenSecret");

        // Set up Host, Endpoint, and Authentication
        Hosts hosebirdHosts = new HttpHosts(Constants.STREAM_HOST);
        StatusesFilterEndpoint hosebirdEndpoint = new StatusesFilterEndpoint();

        // Uncomment if I want to track terms and followers
        // List<Long> followings = Lists.newArrayList(1234L, 566788L);
        // List<String> terms = Lists.newArrayList("twitter", "api");
        // hosebirdEndpoint.followings(followings);
        // hosebirdEndpoint.trackTerms(terms);

        // Authenticate
        Authentication hosebirdAuth = new OAuth1(consumerKey, consumerSecret, accessToken, accessTokenSecret);

        ClientBuilder builder = new ClientBuilder()
                .name("Hosebird-Client-01")
                .hosts(hosebirdHosts)
                .authentication(hosebirdAuth)
                .endpoint(hosebirdEndpoint)
                .processor(new StringDelimitedProcessor(msgQueue));
        return builder.build();
    }

    private static void run() {
        // Set up message queue
        BlockingQueue<String> msgQueue = new LinkedBlockingQueue<String>(1000);

        // Client
        logger.info("Starting Twitter Client");
        Client hosebirdClient = createTwitterClient(msgQueue);
        // Producer
        logger.info("Starting Kafka Producer");
        KafkaProducer<String, String> producer = createProducer();

        // Connect
        logger.info("Connecting Client");
        hosebirdClient.connect();

        // Add shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Caught shutdown hook");
            hosebirdClient.stop();
            producer.close();
        }));

        // on a different thread, or multiple different threads....
        while (!hosebirdClient.isDone()) {
            String msg = null;
            try {
                msg = msgQueue.poll(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
                hosebirdClient.stop();
            }
            if (msg != null) {
                logger.info(msg);
                ProducerRecord<String, String> record = new ProducerRecord<String, String>(TOPIC, null, msg);
                producer.send(record);
            }
        }
        hosebirdClient.stop();
    }

    public static void main(String[] args) {
        run();
    }
}
