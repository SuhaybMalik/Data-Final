package network;

import layers.ConvolutionalLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolingLayer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkGenerator {

    private NeuralNetwork network;
    private int inputRows;
    private int inputColumns;
    private double scaleDown;
    private List<Layer> layerList = new ArrayList<>();

    public NeuralNetworkGenerator(int inputRows, int inputColumns, double scaleDown) {

        this.inputRows = inputRows;
        this.inputColumns = inputColumns;
        this.scaleDown = scaleDown;

    }

    public void addFullyConnectedLayer(int outputLength, long randomSeed, double learningRate) {

        if(layerList.isEmpty()) {
            layerList.add(new FullyConnectedLayer(inputRows * inputColumns, outputLength, randomSeed, learningRate));
        }

        else{

            Layer lastLayer = layerList.getLast();
            layerList.add(new FullyConnectedLayer(lastLayer.outputSize(), outputLength, randomSeed, learningRate));

        }

    }

    public void addMaxPoolLayer(int windowSize, int stepSize) {

        if(layerList.isEmpty()) {
            layerList.add(new MaxPoolingLayer(stepSize, windowSize, 1, inputRows, inputColumns));
        }

        else {

            Layer lastLayer = layerList.getLast();
            layerList.add(new MaxPoolingLayer(stepSize, windowSize, lastLayer.outputLength(), lastLayer.outputRows(), lastLayer.outputColumns()));

        }

    }

    public void addConvolutionalLayer(int numberOfFilters, int filterSize, int stepSize, double learningRate, long randomSeed) {

        if(layerList.isEmpty()) {
            layerList.add(new ConvolutionalLayer(filterSize, stepSize, 1, inputRows, inputColumns, randomSeed, numberOfFilters, learningRate));
        }

        else {

            Layer lastLayer = layerList.getLast();
            layerList.add(new ConvolutionalLayer(filterSize, stepSize, lastLayer.outputLength(), lastLayer.outputRows(), lastLayer.outputColumns(), randomSeed, numberOfFilters, learningRate));

        }

    }

    public NeuralNetwork buildNetwork() {

        network = new NeuralNetwork(layerList, scaleDown);

        return network;

    }

    public List<Layer> getLayerList() {
        return layerList;
    }

}
