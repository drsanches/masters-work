package dataset_creator.feature_extractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class DatasetCreator {

    static private final String DATA_FOLDER = "../data";
    static private final String DATASET_FOLDER = "../dataset";
    static private FeatureExtractor featureExtractor = new FeatureExtractor();

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

    public static void main(String[] args) throws IOException {
//        List<String> groups = getGroupList();

        String group = "ico-wallets";
        List<String> addresses = getAddressList(group);

        for (String address: addresses) {
            try {
                List<String> csvLines = Files.readAllLines(Paths.get(DATA_FOLDER + "/groups/" + group + "/" + address + ".csv"));
                featureExtractor.extractFeaturesFromCSV(csvLines).forEach((k, v) -> System.out.println(k + ": " + v));
                System.out.println();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}