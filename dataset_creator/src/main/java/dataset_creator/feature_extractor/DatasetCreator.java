package dataset_creator.feature_extractor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class DatasetCreator {

    static private final String DATA_FOLDER = "../data";
    static private final String DATASET_FOLDER = "../dataset";
    static private FeatureExtractor featureExtractor = new FeatureExtractor();
    static private int TRANS_THRESHOLD = 100;

    private static List<String> getGroupList() throws IOException {
        List<String> groups = new ArrayList<>();
        Path groupsPath = Paths.get(DATA_FOLDER + "/groups");
        try (Stream<Path> paths = Files.walk(groupsPath)) {
            paths.filter(Files::isDirectory)
                    .filter(x -> !x.toString().equals(groupsPath.toString()))
                    .forEach(s -> groups.add(s.getFileName().toString()));
        }
        return groups;
    }

    private static List<String> getAddressList(String group) throws IOException {
        List<String> addresses = new ArrayList<>();
        Path groupPath = Paths.get(DATA_FOLDER + "/groups/" + group);
        try (Stream<Path> paths = Files.walk(groupPath)) {
            paths.filter(Files::isRegularFile)
                    .forEach(s -> addresses.add(
                           s.getFileName()
                                   .toString()
                                   .substring(0, s.getFileName().toString().length() - ".csv".length())
                    ));
        }
        return addresses;
    }

    private static List<List<List<List<Double>>>> createDataset(String[] groups) throws IOException, ParseException {
        List<List<Double>> trainX = new ArrayList<>();
        List<List<Double>> trainY = new ArrayList<>();
        List<List<Double>> testX = new ArrayList<>();
        List<List<Double>> testY = new ArrayList<>();
        for (int groupIndex = 0; groupIndex < groups.length; groupIndex++) {
            List<List<Double>> groupX = new ArrayList<>();
            List<List<Double>> groupY = new ArrayList<>();
            List<String> addresses = getAddressList(groups[groupIndex]);
            for (String address: addresses) {
                List<String> csvLines = Files.readAllLines(Paths.get(DATA_FOLDER + "/groups/" + groups[groupIndex] + "/" + address + ".csv"));
                if (csvLines.size() > TRANS_THRESHOLD) {
                    Map<String, Double> features = featureExtractor.extractFeaturesFromCSV(csvLines);
                    groupX.add(getX(features));
                    groupY.add(getY(groupIndex, groups.length));
                }
            }
            addData(groupX, trainX, testX, 0.3);
            addData(groupY, trainY, testY, 0.3);
        }
        List<List<List<Double>>> trainDataset = new ArrayList<>();
        trainDataset.add(trainX);
        trainDataset.add(trainY);
        List<List<List<Double>>> testDataset = new ArrayList<>();
        testDataset.add(testX);
        testDataset.add(testY);
        List<List<List<List<Double>>>> dataset = new ArrayList<>();
        dataset.add(trainDataset);
        dataset.add(testDataset);
        return dataset;
    }

    private static void addData(List<List<Double>> groupX, List<List<Double>> trainX, List<List<Double>> testX, double testPart) {
        int N = new Double(groupX.size() * testPart).intValue();
        for (int i = 0; i < groupX.size(); i++) {
            if (i <= N) {
                testX.add(groupX.get(i));
            } else {
                trainX.add(groupX.get(i));
            }
        }
    }

    private static List<Double> getX(Map<String, Double> features) {
        List<Double> X = new ArrayList<>();
        for (FeatureName featureName: FeatureName.values()) {
            X.add(features.get(featureName.toString()));
        }
        return X;
    }

    private static List<Double> getY(int groupIndex, int groupsNumber) {
        List<Double> Y = new ArrayList<>();
        for (int i = 0; i < groupsNumber; i++) {
            if (i != groupIndex) {
                Y.add(0.0);
            } else {
                Y.add(1.0);
            }
        }
        return Y;
    }

    private static void saveDataset(List<List<List<Double>>> dataset, String filename) throws IOException {
        String path = DATASET_FOLDER + "/" + filename;
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        Files.write(Paths.get(path), dataset.toString().getBytes());
    }

    public static void main(String[] args) throws Exception {
        String[] groups = {
                "exchange",
                "ico-wallets",
                "mining",
                "token-contract"
        };
        List<List<List<List<Double>>>> dataset = createDataset(groups);
        List<List<List<Double>>> trainDataset = dataset.get(0);
        List<List<List<Double>>> testDataset = dataset.get(1);
        DataGenerator trainDataGenerator = new DataGenerator(trainDataset);
        System.out.println("Train: " + trainDataGenerator.getSize());
        System.out.println("Test: " + testDataset.get(0).size());
        trainDataGenerator.addShift(0.05);
        trainDataGenerator.addShift(0.1);
        trainDataGenerator.addShift(0.15);
        trainDataGenerator.addShift(0.2);
        trainDataGenerator.addShift(-0.05);
        trainDataGenerator.addShift(-0.1);
        trainDataGenerator.addShift(-0.15);
        trainDataGenerator.addShift(-0.2);
        System.out.println();
        System.out.println("Train: " + trainDataGenerator.getSize());
        System.out.println("Test: " + testDataset.get(0).size());
        saveDataset(trainDataGenerator.getDataset(), "train_threshold_100_shift_05_2.txt");
        saveDataset(testDataset, "test_threshold_100.txt");
    }
}