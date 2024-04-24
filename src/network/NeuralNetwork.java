package network;

import data.ImageReader;
import operations.VectorAndMatrixOperations;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    List<Layer> layerList;
    double scaleDown;

    public NeuralNetwork(List<Layer> layerList, double scaleDown) {

        this.scaleDown = scaleDown;
        this.layerList = layerList;
        linkLayers();

    }

    public void linkLayers() {

        if(layerList.size() <= 1) {
            return;
        }

        for(int i = 0; i < layerList.size(); i++) {

            if(i == layerList.size() - 1) {
                layerList.get(i).setPreviousLayer(layerList.get(i - 1));
            }

            else if(i == 0) {
                layerList.get(i).setNextLayer(layerList.get(i + 1));
            }

            else {

                layerList.get(i).setPreviousLayer(layerList.get(i - 1));
                layerList.get(i).setNextLayer(layerList.get(i + 1));

            }

        }

    }

    public double[] getError(double[] guessOutput, char expectedAnswer) {

        int totalGuesses = guessOutput.length;
        double[] expectedOutput = new double[totalGuesses];
        expectedOutput[(int) expectedAnswer - 97] = 1;

        return VectorAndMatrixOperations.sum(guessOutput, VectorAndMatrixOperations.scalarProduct(expectedOutput, -1));

    }

    public char guess(double[] guessOutput) {

        int tempIndex = 0;
        double tempMax = 0;
        char guess;

        for(int i = 0; i < guessOutput.length; i++) {

            if(guessOutput[i] > tempMax) {

                tempMax = guessOutput[i];
                tempIndex = i;

            }

        }

        guess = (char) (tempIndex + 97);
        return guess;

    }

    public char makeGuess(ImageReader image) {

        List<double[][]> imageList = new ArrayList<>();
        imageList.add(VectorAndMatrixOperations.scalarProduct(image.getData(), 1 / scaleDown));

        return guess(layerList.get(0).output(imageList));

    }

    public void trainNetwork(List<ImageReader> imageList) {

        for(ImageReader image: imageList) {

            List<double[][]> input = new ArrayList<>();
            input.add(VectorAndMatrixOperations.scalarProduct(image.getData(), 1 / scaleDown));
            double[] output = layerList.get(0).output(input);
            double[] rectifiedOutput = squishTo255(output);
            //System.out.println(output.length);
            //System.out.println(output[0]);
            double[] dL_dO = getError(output, image.getLetterLabel());
            layerList.getLast().backProp(dL_dO);

        }

    }

    public float testNetworkPercentage(List<ImageReader> imageList) {

        int count = 0;

        for(ImageReader image: imageList) {

            int guess = makeGuess(image);

            if(guess == image.getLetterLabel()) {
                count++;
            }

        }

        return ((float) count / imageList.size());

    }

    public static double[] squishTo255(double[] values) {
        double[] squishedValues = new double[values.length];
        double minValue = Double.MAX_VALUE;
        double maxValue = Double.MIN_VALUE;

        // Find the minimum and maximum values
        for (double value : values) {
            if (value < minValue) {
                minValue = value;
            }
            if (value > maxValue) {
                maxValue = value;
            }
        }

        // Scale the values to be between 0 and 255
        double range = maxValue - minValue;
        for (int i = 0; i < values.length; i++) {
            squishedValues[i] = ((values[i] - minValue) / range) * 255.0;
        }

        return squishedValues;
    }


}
