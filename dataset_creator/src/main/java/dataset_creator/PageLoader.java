package dataset_creator;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

class PageLoader {

    String downloadHtml(String url) {
        System.setProperty("webdriver.chrome.driver", "./driver/chromedriver.exe");
        WebDriver webDriver = new ChromeDriver();
        webDriver.get(url);
        String source = webDriver.getPageSource();
        webDriver.quit();
        return source;
    }
}