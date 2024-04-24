import data.CsvToImageReader;
import data.ImageReader;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

public class ImageDisplay {

    public static void displayImage(double[][] imagePixels) {
        int width = imagePixels.length;
        int height = imagePixels[0].length;

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        // Convert grayscale values to RGB
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int grayscaleValue = (int) imagePixels[x][y];
                int rgb = (grayscaleValue << 16) | (grayscaleValue << 8) | grayscaleValue;
                image.setRGB(x, y, rgb);
            }
        }

        // Display the image
        ImageIcon icon = new ImageIcon(image);
        JFrame frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(width, height);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static void main(String[] args) {
        // Example usage:
        // Create a 28x28 array filled with grayscale values (e.g., from some image processing algorithm)
        List<ImageReader> testImages = new CsvToImageReader().readImage("data/test data.csv");

        System.out.println(testImages.get(6).getLetterLabel());
        double[][] imagePixels = testImages.get(6).getData();
        // Fill imagePixels with grayscale values...
        // Call the displayImage function to visualize the image
        displayImage(imagePixels);
    }
}
