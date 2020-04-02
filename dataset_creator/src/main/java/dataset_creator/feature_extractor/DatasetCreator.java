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

    private static List<List<List<Double>>> createDataset() throws IOException, ParseException {
        List<String> groups = getGroupList();
        List<List<Double>> X = new ArrayList<>();
        List<List<Double>> Y = new ArrayList<>();
        for (int groupIndex = 0; groupIndex < groups.size(); groupIndex++) {
            List<String> addresses = getAddressList(groups.get(groupIndex));
            for (String address: addresses) {
                List<String> csvLines = Files.readAllLines(Paths.get(DATA_FOLDER + "/groups/" + groups.get(groupIndex) + "/" + address + ".csv"));
                if (csvLines.size() < TRANS_THRESHOLD) {
                    Map<String, Double> features = featureExtractor.extractFeaturesFromCSV(csvLines);
                    X.add(getX(features));
                    Y.add(getY(groupIndex, groups.size()));
                }
            }
        }
        List<List<List<Double>>> dataset = new ArrayList<>();
        dataset.add(X);
        dataset.add(Y);
        return dataset;
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

    private static void saveDataset(List<List<List<Double>>> dataset) throws IOException {
        String path = DATASET_FOLDER + "/dataset.txt";
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        Files.write(Paths.get(path), dataset.toString().getBytes());
    }

    public static void main(String[] args) throws Exception {
//        List<String> groups = getGroupList();

//        String group = "ico-wallets";
//        List<String> addresses = getAddressList(group);
//
//        for (String address: addresses) {
//            try {
//                List<String> csvLines = Files.readAllLines(Paths.get(DATA_FOLDER + "/groups/" + group + "/" + address + ".csv"));
//                featureExtractor.extractFeaturesFromCSV(csvLines).forEach((k, v) -> System.out.println(k + ": " + v));
//                System.out.println();
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//        }

        List<List<List<Double>>> dataset = createDataset();
        saveDataset(dataset);
    }
}