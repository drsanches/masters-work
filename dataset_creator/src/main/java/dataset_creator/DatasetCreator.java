package dataset_creator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class DatasetCreator {

    static private final String DATA_FOLDER = "../data";
    static private final String DATASET_FOLDER = "../dataset";

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

    private static Map<String, String> transformCSV(String group, String address) throws IOException, ParseException {
        Map<String, String> features = new HashMap<>();
        List<String> lines = Files.readAllLines(Paths.get(DATA_FOLDER + "/groups/" + group + "/" + address + ".csv"));
        int N = lines.size() - 2;
        DateFormat format = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");
        Date date2 = format.parse(lines.get(1).split(",")[0]);
        Date date1 = format.parse(lines.get(lines.size() - 1).split(",")[0]);
        features.put("avg_interval", String.valueOf((date2.getTime() - date1.getTime()) / N ));
        double eth_sent = 0L;
        double eth_got = 0L;

        for (int i = 1; i < lines.size(); i++) {
            String[] featureLine = lines.get(i).split(",");
            String direction = featureLine[1];
            double value = Double.parseDouble(featureLine[3].replace(" Ether", ""));
            if (direction.equals("IN")) {
                eth_got += value;
            } else {
                eth_sent += value;
            }
        }

        features.put("avg_eth_sent", String.valueOf(eth_sent / N));
        features.put("avg_eth_got", String.valueOf(eth_got / N));
        features.put("avg_eth", String.valueOf((eth_got - eth_sent) / N));
        return features;
    }

    public static void main(String[] args) throws Exception {
        List<String> groups = getGroupList();
        for (String group: groups) {
            List<String> addresses = getAddressList(group);
            for (String address: addresses) {
                transformCSV(group, address).forEach((k, v) -> System.out.println(k + ": " + v));
                System.out.println();
            }
        }
    }
}