package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class CsvToImageReader {

    //mnist dataset is capped at 28 x 28 images
    private final int rows = 28;
    private final int columns = 28;

    public List<ImageReader> readImage(String filepath) {
        //create a list which contains all our images
        List<ImageReader> imageList= new ArrayList<>();

        try(BufferedReader r = new BufferedReader(new FileReader(filepath))) {

            String line;
            while((line = r.readLine()) != null) {

                String[] pixelValues = line.split(",");
                double[][] data = new double[rows][columns];
                char letterLabel = (char) (Integer.parseInt(pixelValues[0]) + 97);

                int i = 1;
                for(int j = 0; j < rows; j++) {

                    for(int k = 0; k < columns; k++) {

                        data[j][k] = Double.parseDouble(pixelValues[i]);
                        i++;

                    }

                }

                imageList.add(new ImageReader(data, letterLabel));

            }

        }

        catch(Exception e) {
            System.out.println("Exception Executed");
        }

        return imageList;

    }

}
