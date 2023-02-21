import java.io.File;

import net.sourceforge.tess4j.*;

public class Main {
    public static void main(String[] args) {
        Tesseract tesseract = new Tesseract();
        try {
            tesseract.setDatapath("C:\\Users\\appsm\\Downloads\\Tess4J-3.4.8-src\\Tess4J\\tessdata");

            String text = tesseract.doOCR(new File("C:\\Users\\appsm\\Downloads\\airi2.png"));

            System.out.print("airi high res\n" + text);

            text = tesseract.doOCR(new File("C:\\Users\\appsm\\Downloads\\fuuka level.png"));

            System.out.print("\n akane \n");
            System.out.print(text);
        } catch ( TesseractException e) {
            e.printStackTrace();
        }
    }
}