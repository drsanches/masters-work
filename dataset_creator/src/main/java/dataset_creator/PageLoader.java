package dataset_creator;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

class PageLoader {

    private WebDriver webDriver;

    public PageLoader() {
        System.setProperty("webdriver.chrome.driver", "./driver/chromedriver.exe");
        webDriver = new ChromeDriver();
    }

    String downloadHtml(String url) {
        webDriver.get(url);
        return webDriver.getPageSource();
    }

    void close() {
        webDriver.quit();
    }
}