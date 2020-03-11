package dataset_creator;

import javafx.util.Pair;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

class Parser {

    private PageLoader pageLoader;

    public Parser(PageLoader pageLoader) {
        this.pageLoader = pageLoader;
    }

    List<Pair<String, Integer>> getGroupsAndAccountsNumber() {
        List<Pair<String, Integer>> groups = new LinkedList<>();
        String source = pageLoader.downloadHtml("https://etherscan.io/labelcloud");
        Document document = Jsoup.parse(source);
        Elements elements = document.body().getElementsByClass("dropdown-menu list-unstyled py-2 mb-0 w-100 font-size-base");
        for (Element element: elements) {
            String tmp = element.toString();
            if (tmp.contains("Accounts (")) {
                tmp = tmp.substring(tmp.indexOf("Accounts (") + 10);
                tmp = tmp.substring(0, tmp.indexOf(')'));
                groups.add(new Pair<>(element.attributes().get("aria-labelledby"), Integer.parseInt(tmp)));
            }
        }
        return groups;
    }

    /**
     * No longer works, authorization needed
     * */
    List<String> getAddressesByRequest(String groupName) {
        List<String> addresses = new ArrayList<>();
        for (int page = 1;; page++) {
            System.out.println("Page: " + page);
            String source = pageLoader.downloadHtml("https://etherscan.io/accounts/label/" + groupName + "/" + page);
            Document document = Jsoup.parse(source);
            Elements lines = document.body()
                    .getElementsByClass("table table-hover").first()
                    .getElementsByTag("tbody").first()
                    .getElementsByTag("tr");
            if (lines.toString().contains("There are no matching entries")) {
                break;
            }
            for (Element line: lines) {
                addresses.add(line.getElementsByTag("td").first()
                        .getElementsByTag("a").first()
                        .text().replace(" ", ""));
            }
        }
        return addresses;
    }

    List<String> getAddressesFromHtml(String source) {
        List<String> addresses = new ArrayList<>();
        Document document = Jsoup.parse(source);
        Elements lines = document.body()
                .getElementsByClass("table table-hover").first()
                .getElementsByTag("tbody").first()
                .getElementsByTag("tr");
        for (Element line: lines) {
            addresses.add(line.getElementsByTag("td").first()
                    .getElementsByTag("a").first()
                    .text().replace(" ", ""));
        }
        return addresses;
    }

    String getTransactionsCSV(String address) {
        int pagesNumber = 10;
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("time,direction,address,contract,value,price\n");
        for (int page = 1; page <= pagesNumber; page++) {
            System.out.println("Page: " + page + "/" + pagesNumber);
            String source = pageLoader.downloadHtml("https://etherscan.io/txs?a=" + address + "&p=" + page);
            Document document = Jsoup.parse(source);
            Elements lines = document.body()
                    .getElementsByClass("table table-hover").first()
                    .getElementsByTag("tbody").first()
                    .getElementsByTag("tr");
            if (lines.toString().contains("There are no matching entries")) {
                break;
            }
            for (Element line: lines) {
                String time = line.getElementsByTag("td").get(3).getElementsByTag("span").attr("title");
                String direction = line.getElementsByTag("td").get(5).text();
                String address2;
                boolean smartContract;
                if (direction.equals("IN")) {
                    smartContract = !line.getElementsByTag("td").get(4).getElementsByTag("i").isEmpty();
                    address2 = line.getElementsByTag("td").get(4).text();
                } else {
                    smartContract = !line.getElementsByTag("td").get(6).getElementsByTag("i").isEmpty();
                    address2 = line.getElementsByTag("td").get(6).text();
                }
                String value = line.getElementsByTag("td").get(7).text();
                String price = line.getElementsByTag("td").get(8).text();
                stringBuilder.append(time).append(",")
                        .append(direction).append(",")
                        .append(address2).append(",")
                        .append(smartContract).append(",")
                        .append(value).append(",")
                        .append(price).append("\n");
            }
        }
        return stringBuilder.toString();
    }
}