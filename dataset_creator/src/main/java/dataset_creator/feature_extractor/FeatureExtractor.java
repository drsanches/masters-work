package dataset_creator.feature_extractor;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
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
        features.put(FeatureName.AVG_TIME_BETWEEN_TRANS.toString(), getAvgTimeBetweenTrans(transactions));
        features.put(FeatureName.AVG_TIME_BETWEEN_RECEIVED_TRANS.toString(), getAvgTimeBetweenReceivedTrans(transactions));
        features.put(FeatureName.AVG_TIME_BETWEEN_SENT_TRANS.toString(), getAvgTimeBetweenSentTrans(transactions));
        features.put(FeatureName.DEVIATION_TIME_BETWEEN_TRANS.toString(), getDeviationTimeBetweenTrans(transactions));
        return features;
    }

    private Double getAvgTimeBetweenTrans(List<String> transactions) throws ParseException {
        double N = transactions.size() - 1;
        Date date2 = DATE_FORMAT.parse(transactions.get(0).split(",")[0]);
        Date date1 = DATE_FORMAT.parse(transactions.get(transactions.size() - 1).split(",")[0]);
        return (date2.getTime() - date1.getTime()) / N;
    }

    /**
     * @return transaction string or null if there are no received transactions
     * */
    private String getFirstReceivedTransaction(List<String> transactions) {
        for (int i = transactions.size() - 1; i >= 0; i--) {
            String direction = transactions.get(i).split(",")[1];
            if (direction.equals("IN")) {
                return transactions.get(i);
            }
        }
        return null;
    }

    /**
     * @return transaction string or null if there are no received transactions
     * */
    private String getLastReceivedTransaction(List<String> transactions) {
        for (String line: transactions) {
            String direction = line.split(",")[1];
            if (direction.equals("IN")) {
                return line;
            }
        }
        return null;
    }

    private int getReceivedTransactionsNumber(List<String> transactions) {
        int count = 0;
        for (String line: transactions) {
            String direction = line.split(",")[1];
            if (direction.equals("IN")) {
                count++;
            }
        }
        return count;
    }

    /**
     * @return average time or null if there are no received transactions
     * */
    private Double getAvgTimeBetweenReceivedTrans(List<String> transactions) throws ParseException {
        double N = getReceivedTransactionsNumber(transactions);
        String transaction1 = getFirstReceivedTransaction(transactions);
        String transaction2 = getLastReceivedTransaction(transactions);
        if (transaction1 != null && transaction2 != null) {
            Date date1 = DATE_FORMAT.parse(transaction1.split(",")[0]);
            Date date2 = DATE_FORMAT.parse(transaction2.split(",")[0]);
            return (date2.getTime() - date1.getTime()) / N;
        }
        return null;
    }

    /**
     * @return transaction string or null if there are no sent transactions
     * */
    private String getFirstSentTransaction(List<String> transactions) {
        for (int i = transactions.size() - 1; i >= 0; i--) {
            String direction = transactions.get(i).split(",")[1];
            if (direction.equals("OUT")) {
                return transactions.get(i);
            }
        }
        return null;
    }

    /**
     * @return transaction string or null if there are no sent transactions
     * */
    private String getLastSentTransaction(List<String> transactions) {
        for (String line: transactions) {
            String direction = line.split(",")[1];
            if (direction.equals("OUT")) {
                return line;
            }
        }
        return null;
    }

    private int getSentTransactionsNumber(List<String> transactions) {
        int count = 0;
        for (String line: transactions) {
            String direction = line.split(",")[1];
            if (direction.equals("OUT")) {
                count++;
            }
        }
        return count;
    }

    /**
     * @return average time or null if there are no sent transactions
     * */
    private Double getAvgTimeBetweenSentTrans(List<String> transactions) throws ParseException {
        double N = getSentTransactionsNumber(transactions);
        String transaction1 = getFirstSentTransaction(transactions);
        String transaction2 = getLastSentTransaction(transactions);
        if (transaction1 != null && transaction2 != null) {
            Date date1 = DATE_FORMAT.parse(transaction1.split(",")[0]);
            Date date2 = DATE_FORMAT.parse(transaction2.split(",")[0]);
            return (date2.getTime() - date1.getTime()) / N;
        }
        return null;
    }

    private Double getDeviationTimeBetweenTrans(List<String> transactions) throws ParseException {
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