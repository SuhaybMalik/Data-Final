package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolingLayer extends Layer {

    private int stepSize;
    private int windowSize;
    private int inputLength;
    private int inputColumns;
    private int inputRows;
    List<int[][]> lastMaxRow;
    List<int[][]> lastMaxColumn;

    public MaxPoolingLayer(int stepSize, int windowSize, int inputLength, int inputColumns, int inputRows) {

        this.stepSize = stepSize;
        this.windowSize = windowSize;
        this.inputLength = inputLength;
        this.inputColumns = inputColumns;
        this.inputRows = inputRows;

    }

    @Override
    public int outputLength() {
        return inputLength;
    }

    @Override
    public int outputRows() {
        return (inputRows - windowSize) / stepSize + 1;
    }

    @Override
    public int outputColumns() {
        return (inputColumns - windowSize) / stepSize + 1;
    }

    @Override
    public int outputSize() {
        return inputLength * outputColumns() * outputRows();
    }

    public List<double[][]> poolForwardProp(List<double[][]> input) {

        List<double[][]> output = new ArrayList<>();
        lastMaxRow = new ArrayList<>();
        lastMaxColumn = new ArrayList<>();

        for(int i = 0; i < input.size(); i++) {

            output.add(pool(input.get(i)));

        }

        return output;

    }

    public double[][] pool(double[][] input) {

        double[][] output = new double[outputRows()][outputColumns()];
        int[][] maxRows = new int[outputRows()][outputColumns()];
        int[][] maxColumns = new int[outputRows()][outputColumns()];

        for(int i = 0; i < outputRows(); i += stepSize) {

            for(int j = 0; j < outputColumns(); j += stepSize) {

                double temp = 0;
                maxRows[i][j] = -1;
                maxColumns[i][j] = -1;

                for(int k = 0; k < windowSize; k++) {

                    for(int l = 0; l < windowSize; l++) {

                        if(temp < input[i + k][j + l]) {

                            temp = input[i + k][j + l];
                            maxRows[i][j] = i + k;
                            maxColumns[i][j] = j + l;

                        }

                    }

                }

                output[i][j] = temp;

            }

        }

        lastMaxRow.add(maxRows);
        lastMaxColumn.add(maxColumns);

        return output;

    }

    @Override
    public double[] output(List<double[][]> input) {

        List<double[][]> forwardPassResult = poolForwardProp(input);
        return nextLayer.output(forwardPassResult);

    }

    @Override
    public double[] output(double[] input) {

        List<double[][]> listMatrix = vectorToListMatrix(input, outputLength(), outputRows(), outputColumns());
        return output(listMatrix);

    }

    @Override
    public void backProp(List<double[][]> dL_dO) {

        List<double[][]> dX_dL = new ArrayList<>();
        int x = 0;

        for(double[][] arr: dL_dO) {

            double[][] error = new double[inputRows][inputColumns];

            for(int i = 0; i < outputRows(); i++) {

                for(int j = 0; j < outputColumns(); j++){

                    int maxX = lastMaxRow.get(x)[i][j];
                    int maxY = lastMaxColumn.get(x)[i][j];

                    if(maxX != -1) {
                        error[maxX][maxY] += arr[i][j];
                    }

                }

            }

            dX_dL.add(error);
            x++;

        }

        if(previousLayer != null) {
            previousLayer.backProp(dX_dL);
        }

    }

    @Override
    public void backProp(double[] dL_dO) {

        List<double[][]> listMatrix = vectorToListMatrix(dL_dO, outputLength(), outputRows(), outputColumns());
        backProp(listMatrix);

    }
}
