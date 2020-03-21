package dataset_creator.data_loader;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

class PageLoader {

    private WebDriver webDriver = null;

    private void initDriver() {
        System.setProperty("webdriver.chrome.driver", "./driver/chromedriver.exe");
        webDriver = new ChromeDriver();
    }

    String downloadHtml(String url) {
        if (webDriver == null) {
            initDriver();
        }
        webDriver.get(url);
        return webDriver.getPageSource();
    }

    void close() {
        webDriver.quit();
        webDriver = null;
    }
}