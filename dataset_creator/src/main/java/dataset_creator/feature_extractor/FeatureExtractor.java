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
        features.put(FeatureName.AVG_TIME_BETWEEN_TRANS.toString(), getAvgTimeBetweenTrans(csvLines));
        features.put(FeatureName.AVG_TIME_BETWEEN_RECEIVED_TRANS.toString(), getAvgTimeBetweenReceivedTrans(csvLines));
        features.put(FeatureName.AVG_TIME_BETWEEN_SENT_TRANS.toString(), getAvgTimeBetweenSentTrans(csvLines));
        return features;
    }

    private Double getAvgTimeBetweenTrans(List<String> csvLines) throws ParseException {
        double N = csvLines.size() - 1;
        Date date2 = DATE_FORMAT.parse(csvLines.get(1).split(",")[0]);
        Date date1 = DATE_FORMAT.parse(csvLines.get(csvLines.size() - 1).split(",")[0]);
        return (date2.getTime() - date1.getTime()) / N;
    }

    /**
     * @return transaction string or null if there are no received transactions
     * */
    private String getFirstReceivedTransaction(List<String> csvLines) {
        for (int i = csvLines.size() - 1; i >= 0; i--) {
            String direction = csvLines.get(i).split(",")[1];
            if (direction.equals("IN")) {
                return csvLines.get(i);
            }
        }
        return null;
    }

    /**
     * @return transaction string or null if there are no received transactions
     * */
    private String getLastReceivedTransaction(List<String> csvLines) {
        for (String line: csvLines) {
            String direction = line.split(",")[1];
            if (direction.equals("IN")) {
                return line;
            }
        }
        return null;
    }

    private int getReceivedTransactionsNumber(List<String> csvLines) {
        int count = 0;
        for (String line: csvLines) {
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
    private Double getAvgTimeBetweenReceivedTrans(List<String> csvLines) throws ParseException {
        double N = getReceivedTransactionsNumber(csvLines);
        String transaction1 = getFirstReceivedTransaction(csvLines);
        String transaction2 = getLastReceivedTransaction(csvLines);
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
    private String getFirstSentTransaction(List<String> csvLines) {
        for (int i = csvLines.size() - 1; i >= 0; i--) {
            String direction = csvLines.get(i).split(",")[1];
            if (direction.equals("OUT")) {
                return csvLines.get(i);
            }
        }
        return null;
    }

    /**
     * @return transaction string or null if there are no sent transactions
     * */
    private String getLastSentTransaction(List<String> csvLines) {
        for (String line: csvLines) {
            String direction = line.split(",")[1];
            if (direction.equals("OUT")) {
                return line;
            }
        }
        return null;
    }

    private int getSentTransactionsNumber(List<String> csvLines) {
        int count = 0;
        for (String line: csvLines) {
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
    private Double getAvgTimeBetweenSentTrans(List<String> csvLines) throws ParseException {
        double N = getSentTransactionsNumber(csvLines);
        String transaction1 = getFirstSentTransaction(csvLines);
        String transaction2 = getLastSentTransaction(csvLines);
        if (transaction1 != null && transaction2 != null) {
            Date date1 = DATE_FORMAT.parse(transaction1.split(",")[0]);
            Date date2 = DATE_FORMAT.parse(transaction2.split(",")[0]);
            return (date2.getTime() - date1.getTime()) / N;
        }
        return null;
    }
}