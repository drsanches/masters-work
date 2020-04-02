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
        features.put(FeatureName.AVG_TRANS_ETH.toString(), getAvgTransEth(transactions));
        features.put(FeatureName.AVG_ETH_RECEIVED.toString(), getAvgTransEth(receivedTransactions));
        features.put(FeatureName.AVG_ETH_SENT.toString(), getAvgTransEth(sentTransactions));
        features.put(FeatureName.DEVIATION_TRANS_ETH.toString(), getDeviationTransEth(transactions));
        features.put(FeatureName.DEVIATION_ETH_RECEIVED.toString(), getDeviationTransEth(receivedTransactions));
        features.put(FeatureName.DEVIATION_ETH_SENT.toString(), getDeviationTransEth(sentTransactions));

        //TODO: think
        features.put(FeatureName.PERCENT_OF_SMART_CONTRACT_TRANS.toString(), getPercentOfSmartContractTrans(transactions));
        features.put(FeatureName.PERCENT_OF_TRANS_RECEIVED_FROM_SMART_CONTRACTS.toString(), getPercentOfSmartContractTrans(receivedTransactions));
        features.put(FeatureName.PERCENT_OF_TRANS_SENT_TO_SMART_CONTRACTS.toString(), getPercentOfSmartContractTrans(sentTransactions));
        features.put(FeatureName.PERCENT_OF_SMART_CONTRACT_ETH.toString(), getPercentOfSmartContractEth(transactions));
        features.put(FeatureName.PERCENT_OF_ETH_RECEIVED_FROM_SMART_CONTRACTS.toString(), getPercentOfSmartContractEth(receivedTransactions));
        features.put(FeatureName.PERCENT_OF_ETH_SENT_TO_SMART_CONTRACTS.toString(), getPercentOfSmartContractEth(sentTransactions));
        return features;
    }

    public List<String> getReceivedTransactionsList(List<String> transactions) {
        List<String> receivedTransactions = new ArrayList<>();
        for (String transaction: transactions) {
            if (getDirection(transaction).equals("IN")) {
                receivedTransactions.add(transaction);
            }
        }
        return receivedTransactions;
    }

    public List<String> getSentTransactionsList(List<String> transactions) {
        List<String> sentTransactions = new ArrayList<>();
        for (String transaction: transactions) {
            if (getDirection(transaction).equals("OUT")) {
                sentTransactions.add(transaction);
            }
        }
        return sentTransactions;
    }

    public List<String> getSmartContractTransactionsList(List<String> transactions) {
        List<String> smartContractTransactions = new ArrayList<>();
        for (String transaction: transactions) {
            if (isSmartContract(transaction)) {
                smartContractTransactions.add(transaction);
            }
        }
        return smartContractTransactions;
    }

    /**
     * @return average time or 0 if there are less than two transactions
     * */
    private Double getAvgTimeBetweenTrans(List<String> transactions) throws ParseException {
        if (transactions.size() < 2) {
            return 0.0;
        }
        double N = transactions.size() - 1;
        Date date2 = getDate(transactions.get(0));
        Date date1 = getDate(transactions.get(transactions.size() - 1));
        return (date2.getTime() - date1.getTime()) / N;
    }

    /**
     * @return deviation time or 0 if there are less than two transactions
     * */
    private Double getDeviationTimeBetweenTrans(List<String> transactions) throws ParseException {
        if (transactions.size() < 2) {
            return 0.0;
        }
        double avgTime = getAvgTimeBetweenTrans(transactions);
        double sum = 0;
        Date date1 = getDate(transactions.get(0));
        Date date2;
        for (int i = 1; i < transactions.size(); i++) {
            date2 = date1;
            date1 = getDate(transactions.get(i));
            double time = date2.getTime() - date1.getTime();
            sum += (time - avgTime) * (time - avgTime);
        }
        double N = transactions.size() - 1;
        return Math.sqrt(sum / N);
    }

    /**
     * @return average eth or 0 if there are no transactions
     * */
    private Double getAvgTransEth(List<String> transactions) {
        if (transactions.isEmpty()) {
            return 0.0;
        }
        double sum = 0;
        for (String transaction: transactions) {
            sum += getEth(transaction);
        }
        return sum / transactions.size();
    }

    /**
     * @return deviation eth or 0 if there are less than two transactions
     * */
    private Double getDeviationTransEth(List<String> transactions) {
        if (transactions.isEmpty()) {
            return 0.0;
        }
        double avgEth = getAvgTransEth(transactions);
        double sum = 0;
        for (String transaction: transactions) {
            double eth = getEth(transaction);
            sum += (eth - avgEth) * (eth - avgEth);
        }
        double N = transactions.size();
        return Math.sqrt(sum / N);
    }

    /**
     * @return percent or 0 if there are no transactions
     * */
    private Double getPercentOfSmartContractTrans(List<String> transactions) {
        double smartContractN = getSmartContractTransactionsList(transactions).size();
        double N = transactions.size();
        if (N > 0) {
            return smartContractN / N;
        } else {
            return 0.0;
        }
    }

    /**
     * @return percent or 0 if there are no eth in transactions
     * */
    private Double getPercentOfSmartContractEth(List<String> transactions) {
        double smartContractEth = getTotalEth(getSmartContractTransactionsList(transactions));
        double allEth = getTotalEth(transactions);
        if (allEth > 0) {
            return smartContractEth / allEth;
        } else {
            return 0.0;
        }
    }

    private Double getTotalEth(List<String> transactions) {
        double sum = 0;
        for (String transaction: transactions) {
            sum += getEth(transaction);
        }
        return sum;
    }

    private String getDirection(String transaction) {
        return transaction.split(",")[1];
    }

    private Boolean isSmartContract(String transaction) {
        return transaction.split(",")[3].equals("true");
    }

    private Date getDate(String transaction) throws ParseException {
        return DATE_FORMAT.parse(transaction.split(",")[0]);
    }

    private Double getEth(String transaction) {
        String eth = transaction.split(",")[4];
        return Double.parseDouble(eth.substring(0, eth.indexOf(" ")));
    }
}