package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    protected Layer nextLayer;
    protected Layer previousLayer;

    public abstract int outputLength();
    public abstract int outputRows();
    public abstract int outputColumns();
    public abstract int outputSize();

    public abstract double[] output(List<double[][]> input);
    public abstract double[] output(double[] input);

    public abstract void backProp(List<double[][]> dL_dO);
    public abstract void backProp(double[] dL_dO);

    public List<double[][]> vectorToListMatrix(double[] input, int length, int rows, int columns) {

        List<double[][]> matrix = new ArrayList<>();
        int currentIndex = 0;

        for(int i = 0; i < length; i++) {

            double[][] tempDoubleMatrix = new double[rows][columns];

            for(int j = 0; j < rows; j++) {

                for(int k = 0; k < columns; k++) {

                    tempDoubleMatrix[i][j] = input[currentIndex];
                    currentIndex++;

                }

            }

            matrix.add(tempDoubleMatrix);

        }

        return matrix;

    }

    public double[] listMatrixToVector(List<double[][]> input) {

        int length = input.size();
        int rows = input.get(0).length;
        int columns = input.get(0)[0].length;
        int vectorSize = length * rows * columns;

        double[] vector = new double[vectorSize];
        int listIndex = 0;

        for(int i = 0; i < length; i++) {

            for(int j = 0; j < rows; j++) {

                for(int k = 0; k < columns; k++) {

                    vector[listIndex] = input.get(i)[j][k];
                    listIndex++;

                }

            }

        }

        return vector;

    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

}
