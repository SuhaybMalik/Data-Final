package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer {

    private double[][] weights;
    private int inputLength;
    private int outputLength;
    private long randomSeed;
    double learningRate;
    private double[] lastOutput;
    private double[] lastInput;

    public FullyConnectedLayer(int inputLength, int outputLength, long randomSeed, double learningRate) {

        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.randomSeed = randomSeed;
        this.learningRate = learningRate;
        weights = new double[inputLength][outputLength];
        generateRandomWeights();

    }

    @Override
    public int outputLength() {
        return 0;
    }

    @Override
    public int outputRows() {
        return 0;
    }

    @Override
    public int outputColumns() {
        return 0;
    }

    @Override
    public int outputSize() {
        return outputLength;
    }

    public double[] fullForwardProp(double[] input) {

        lastInput = input;
        double[] z = new double[outputLength];

        for(int i = 0; i < inputLength; i++) {

            for(int j = 0; j < outputLength; j++) {

                z[j] += weights[i][j] * input[i];

            }

        }

        lastOutput = z;

        double[] zRelu = new double[outputLength];

        for(int i = 0; i < outputLength; i++) {

            zRelu[i] = RELU(z[i]);

        }

        return zRelu;

    }

    @Override
    public double[] output(List<double[][]> input) {

        double[] vector = listMatrixToVector(input);
        return output(vector);

    }

    @Override
    public double[] output(double[] input) {

        double[] output = fullForwardProp(input);

        if(nextLayer != null) {
            return nextLayer.output(output);
        }

        else {
            return output;
        }

    }

    @Override
    public void backProp(List<double[][]> dL_dO) {

        double[] vector = listMatrixToVector(dL_dO);
        backProp(vector);

    }

    @Override
    public void backProp(double[] dL_dO) {

        double dO_dZ;
        double dZ_dW;
        double dL_dW;
        double dZ_dX;
        double[] dL_dX = new double[inputLength];

        for(int i = 0; i < inputLength; i++) {

            double dL_dX_SUM = 0;

            for(int j = 0; j < outputLength; j++) {

                dO_dZ = leakyRELU(lastOutput[j]);
                dZ_dW = lastInput[i];
                dZ_dX = weights[i][j];

                dL_dW = dO_dZ * dZ_dW * dL_dO[j];
                weights[i][j] -= dL_dW * learningRate;

                dL_dX_SUM += dL_dO[j] * dO_dZ * dZ_dX;

            }

            dL_dX[i] = dL_dX_SUM;

        }

        if(previousLayer != null) {
            previousLayer.backProp(dL_dX);
        }

    }

    public void generateRandomWeights() {

        Random r = new Random(randomSeed);

        for(int i = 0; i < inputLength; i++) {

            for(int j = 0; j < outputLength; j++) {

                weights[i][j] = r.nextGaussian();

            }

        }

    }

    public double RELU(double input) {

        if(input <= 0) {
            return 0;
        }

        else {
            return input;
        }

    }

    public double leakyRELU(double input) {

        if(input <= 0) {
            return 0.01;
        }

        else {
            return input;
        }

    }

}

