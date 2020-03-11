package dataset_creator;

import javafx.util.Pair;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;

public class DataLoader {

    private static final String DATA_FOLDER = "../data";
    private static PageLoader pageLoader = new PageLoader();
    private static Parser parser = new Parser(pageLoader);

    private static void printGroups() {
        List<Pair<String, Integer>> groups = parser.getGroupsAndAccountsNumber();
        groups.sort(Comparator.comparingInt(Pair::getValue));
        groups.forEach(x -> System.out.println(x.getValue() + " " + x.getKey()));
    }

    /**
     * No longer works, authorization needed
     * */
    private static void downloadAddressesByRequest(String group) throws IOException {
        StringBuilder stringBuilder = new StringBuilder();
        parser.getAddressesByRequest(group).forEach(x -> stringBuilder.append(x).append("\n"));
        String path = DATA_FOLDER + "/groups/" + group + ".txt";
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        Files.write(Paths.get(path), stringBuilder.toString().getBytes());
    }

    private static void downloadAddressesFromHtml(String group) throws IOException {
        StringBuilder stringBuilder = new StringBuilder();
        int count = 0;
        for (int i = 1;; i++) {
            String sourcePath = DATA_FOLDER + "/html/" + group + "-" + i + ".html";
            if (!Files.exists(Paths.get(sourcePath))) {
                break;
            }
            List<String> addresses = parser.getAddressesFromHtml(new String(Files.readAllBytes(Paths.get(sourcePath)), StandardCharsets.UTF_8));
            addresses.forEach(x -> stringBuilder.append(x).append("\n"));
            count += addresses.size();
        }
        System.out.println(stringBuilder.toString());
        System.out.println("Count: " + count);
        String path = DATA_FOLDER + "/groups/" + group + ".txt";
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        Files.write(Paths.get(path), stringBuilder.toString().getBytes());
    }

    private static void downloadTransactions(String address, String group) throws IOException {
        String csv = parser.getTransactionsCSV(address);
        String path = DATA_FOLDER + "/groups/" + group + "/" + address + ".csv";
        File file = new File(path);
        file.getParentFile().mkdirs();
        file.createNewFile();
        Files.write(Paths.get(path), csv.getBytes());
    }

    public static void main(String[] args) {
//        printGroups();

//        try {
//            downloadAddressesFromHtml("token-contract");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        String group = "token-contract";
        String[] addresses = (
                "address1\n" +
                "address2\n" +
                "address3\n" +
                "address4\n" +
                "address5"
        ).split("\n");

        for (int i = 0; i < addresses.length; i++) {
            System.out.println("Address #" + i);
            try {
                downloadTransactions(addresses[i], group);
                pageLoader.close(); // To update a session (the resource has a limit of requests per session)
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}