package layers;

import operations.VectorAndMatrixOperations;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionalLayer extends Layer {

    private List<double[][]> filterList;
    private int filterSize;
    private int stepSize;
    private int inputLength;
    private int inputRows;
    private int inputColumns;
    private long randomSeed;
    private List<double[][]> previousInput;
    private double learningRate;

    public ConvolutionalLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputColumns, long randomSeed, int numberOfFilters, double learningRate) {

        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputColumns = inputColumns;
        this.randomSeed = randomSeed;
        this.learningRate = learningRate;

        generateRandomFilters(numberOfFilters);

    }

    public void generateRandomFilters(int numberOfFilters) {

        List<double[][]> output = new ArrayList<>();
        Random r = new Random(randomSeed);

        for(int i = 0; i < numberOfFilters; i++) {

            double[][] temp = new double[filterSize][filterSize];

            for(int j = 0; j < filterSize; j++) {

                for(int k = 0; k < filterSize; k++) {

                    temp[j][k] = r.nextGaussian();

                }

            }

            output.add(temp);

        }

        filterList = output;

    }

    public double[][] convolution(double[][] input, double[][] filter, int stepSize) {

        int outputRows = (input.length - filter.length)/stepSize + 1;
        int outputColumns = (input[0].length - filter[0].length)/stepSize + 1;
        int inputRows = input.length;
        int inputColumns = input[0].length;
        int filterRows = filter.length;
        int filterColumns = filter[0].length;

        double[][] output = new double[outputRows][outputColumns];

        int row = 0;
        int column;

        for(int i = 0; i <= inputRows - filterRows; i += stepSize) {

            column = 0;

            for(int j = 0; j <= inputColumns - filterColumns; j += stepSize) {

                double sum = 0;

                for(int k = 0; k < filterRows; k++) {

                    for(int l = 0; l < filterColumns; l++) {

                        int inputRowIndex = i + k;
                        int inputColumnIndex = j + l;

                        double val = filter[k][l] * input[inputRowIndex][inputColumnIndex];
                        sum+= val;

                    }

                }

                output[row][column] = sum;
                column++;

            }

            row++;

        }

        return output;

    }

    public List<double[][]> convolutionalForwardProp(List<double[][]> input) {

        previousInput = input;
        List<double[][]> output = new ArrayList<>();

        for(int i = 0; i < input.size(); i++) {

            for(double[][] filter: filterList) {

                output.add(convolution(input.get(i), filter, stepSize));

            }

        }

        return output;

    }

    public double[][] matrixSpacer(double[][] input) {

        if(stepSize == 1) {
            return input;
        }

        int outputRows = (input.length - 1) * stepSize + 1;
        int outputColumns = (input[0].length - 1) * stepSize + 1;

        double[][] output = new double[outputRows][outputColumns];

        for(int i = 0; i < input.length; i++) {

            for(int j = 0; j < input[0].length; j++) {

                output[i * stepSize][j * stepSize] = input[i][j];

            }

        }

        return output;

    }

    public double[][] matrixHorizontalFlip(double[][] input) {

        int outputRows = input.length;
        int outputColumns = input[0].length;
        double[][] output = new double[outputRows][outputColumns];

        for(int i = 0; i < outputRows; i++) {

            for(int j = 0; j < outputColumns; j++) {

                output[outputRows - i - 1][j] = input[i][j];

            }

        }

        return output;

    }

    public double[][] matrixVerticalFlip(double[][] input) {

        int outputRows = input.length;
        int outputColumns = input[0].length;
        double[][] output = new double[outputRows][outputColumns];

        for(int i = 0; i < outputRows; i++) {

            for(int j = 0; j < outputColumns; j++) {

                output[i][outputColumns - j - 1] = input[i][j];

            }

        }

        return output;

    }

    public double[][] completeConvolution(double[][] input, double[][] filter) {

        int outputRows = (input.length + filter.length) + 1;
        int outputColumns = (input[0].length + filter[0].length)+ 1;
        int inputRows = input.length;
        int inputColumns = input[0].length;
        int filterRows = filter.length;
        int filterColumns = filter[0].length;

        double[][] output = new double[outputRows][outputColumns];

        int row = 0;
        int column;

        for(int i = - filterRows + 1; i < inputRows; i++) {

            column = 0;

            for(int j = - filterColumns + 1; j < inputColumns; j++) {

                double sum = 0;

                for(int k = 0; k < filterRows; k++) {

                    for(int l = 0; l < filterColumns; l++) {

                        int inputRowIndex = i + k;
                        int inputColumnIndex = j + l;

                        if(inputRowIndex >= 0 && inputColumnIndex >= 0 && inputRowIndex < inputRows && inputColumnIndex < inputColumns) {

                            double val = filter[k][l] * input[inputRowIndex][inputColumnIndex];
                            sum += val;

                        }

                    }

                }

                output[row][column] = sum;
                column++;

            }

            row++;

        }

        return output;

    }


    @Override
    public int outputLength() {
        return (filterList.size() * inputLength);
    }

    @Override
    public int outputRows() {
        return (inputRows - filterSize) / stepSize + 1;
    }

    @Override
    public int outputColumns() {
        return (inputColumns - filterSize) / stepSize + 1;
    }

    @Override
    public int outputSize() {
        return outputLength() * outputRows() * outputColumns();
    }

    @Override
    public double[] output(List<double[][]> input) {

        List<double[][]> output = convolutionalForwardProp(input);
        return nextLayer.output(output);

    }

    @Override
    public double[] output(double[] input) {

        List<double[][]> matrixInput = vectorToListMatrix(input, inputLength, inputRows, inputColumns);
        return output(matrixInput);

    }

    @Override
    public void backProp(List<double[][]> dL_dO) {

        List<double[][]> dF = new ArrayList<>();
        List<double[][]> previousLayerError = new ArrayList<>();

        for(int i = 0; i < filterList.size(); i++) {

            dF.add(new double[filterSize][filterSize]);

        }

        for(int i = 0; i < previousInput.size(); i++) {

            double[][] errorMatrix = new double[inputRows][inputColumns];

            for(int j = 0; j < filterList.size(); j++) {

                double[][] tempFilter = filterList.get(j);
                double[][] error = dL_dO.get(i * filterList.size() + j);
                double[][] matrixSpacedError = matrixSpacer(error);
                double[][] dL_dF = convolution(previousInput.get(i), matrixSpacedError, 1);

                double[][] d = VectorAndMatrixOperations.scalarProduct(dL_dF, learningRate * -1);
                double[][] sumD = VectorAndMatrixOperations.sum(dF.get(j), d);
                dF.set(j, sumD);

                double[][] newError = matrixHorizontalFlip(matrixVerticalFlip(matrixSpacedError));
                errorMatrix = VectorAndMatrixOperations.sum(errorMatrix, completeConvolution(tempFilter, newError));

            }

            previousLayerError.add(errorMatrix);

        }

        for(int i = 0; i < filterList.size(); i++) {

            double[][] newFilter = VectorAndMatrixOperations.sum(dF.get(i), filterList.get(i));
            filterList.set(i, newFilter);

        }

        if(previousLayer != null) {
            previousLayer.backProp(previousLayerError);
        }

    }

    @Override
    public void backProp(double[] dL_dO) {

        List<double[][]> matrixError = vectorToListMatrix(dL_dO, inputLength, inputRows, inputColumns);
        backProp(matrixError);

    }
}
