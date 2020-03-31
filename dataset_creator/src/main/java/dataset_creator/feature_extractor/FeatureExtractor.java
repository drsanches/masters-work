package dataset_creator.feature_extractor;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FeatureExtractor {

    private final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");

    /**
     * CSV fields: time,direction,address,contract,value,price
     * */
    public Map<String, Double> extractFeaturesFromCSV(List<String> csvLines) throws ParseException {
        Map<String, Double> features = new HashMap<>();
        List<String> transactions = csvLines.subList(1, csvLines.size());
        List<String> receivedTransactions = getReceivedTransactionsList(transactions);
        List<String> sentTransactions = getSentTransactionsList(transactions);

        features.put(FeatureName.AVG_TIME_BETWEEN_TRANS.toString(), getAvgTimeBetweenTrans(transactions));
        features.put(FeatureName.AVG_TIME_BETWEEN_RECEIVED_TRANS.toString(), getAvgTimeBetweenTrans(receivedTransactions));
        features.put(FeatureName.AVG_TIME_BETWEEN_SENT_TRANS.toString(), getAvgTimeBetweenTrans(sentTransactions));
        features.put(FeatureName.DEVIATION_TIME_BETWEEN_TRANS.toString(), getDeviationTimeBetweenTrans(transactions));
        features.put(FeatureName.DEVIATION_TIME_BETWEEN_RECEIVED_TRANS.toString(), getDeviationTimeBetweenTrans(receivedTransactions));
        features.put(FeatureName.DEVIATION_TIME_BETWEEN_SENT_TRANS.toString(), getDeviationTimeBetweenTrans(sentTransactions));
        return features;
    }

    public List<String> getReceivedTransactionsList(List<String> transactions) {
        List<String> receivedTransactions = new ArrayList<>();
        for (String transaction: transactions) {
            String direction = transaction.split(",")[1];
            if (direction.equals("IN")) {
                receivedTransactions.add(transaction);
            }
        }
        return receivedTransactions;
    }

    public List<String> getSentTransactionsList(List<String> transactions) {
        List<String> sentTransactions = new ArrayList<>();
        for (String transaction: transactions) {
            String direction = transaction.split(",")[1];
            if (direction.equals("OUT")) {
                sentTransactions.add(transaction);
            }
        }
        return sentTransactions;
    }

    /**
     * @return average time or null if there are less than two transactions
     * */
    private Double getAvgTimeBetweenTrans(List<String> transactions) throws ParseException {
        if (transactions.size() < 2) {
            return null;
        }
        double N = transactions.size() - 1;
        Date date2 = DATE_FORMAT.parse(transactions.get(0).split(",")[0]);
        Date date1 = DATE_FORMAT.parse(transactions.get(transactions.size() - 1).split(",")[0]);
        return (date2.getTime() - date1.getTime()) / N;
    }

    /**
     * @return deviation time or null if there are less than two transactions
     * */
    private Double getDeviationTimeBetweenTrans(List<String> transactions) throws ParseException {
        if (transactions.size() < 2) {
            return null;
        }
        double avgTime = getAvgTimeBetweenTrans(transactions);
        double sum = 0;
        Date date1 = DATE_FORMAT.parse(transactions.get(0).split(",")[0]);
        Date date2;
        for (int i = 1; i < transactions.size(); i++) {
            date2 = date1;
            date1 = DATE_FORMAT.parse(transactions.get(i).split(",")[0]);
            double time = date2.getTime() - date1.getTime();
            sum += (time - avgTime) * (time - avgTime);
        }
        double N = transactions.size() - 1;
        return Math.sqrt(sum / N);
    }
}