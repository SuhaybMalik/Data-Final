package data;

public class ImageReader {

    private double[][] data;//grayscale values as a 2D double array
    private char letterLabel;//label of which letter the image is supposed to be

    public ImageReader(double[][] data, char letterLabel) {

        this.data = data;
        this.letterLabel = letterLabel;

    }


    public double[][] getData() {
        return data;
    }

    public char getLetterLabel() {
        return letterLabel;
    }

    @Override
    public String toString() {

        String imageRepresentation = letterLabel + "\n";

        for(int i = 0; i < data.length; i++) {

            for(int j = 0; j < data[0].length; j++) {
                imageRepresentation += data[i][j] + ", ";
            }

            imageRepresentation += "\n";

        }

        return imageRepresentation;

    }
}
