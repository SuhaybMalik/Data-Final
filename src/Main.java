import data.CsvToImageReader;
import data.ImageReader;
import network.NeuralNetwork;
import network.NeuralNetworkGenerator;

import java.util.ArrayList;
import java.util.List;

import static java.util.Collections.shuffle;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    public static void main(String[] args) {

        System.out.println("Start loading images: ");

        //List<ImageReader> trainImages = new CsvToImageReader().readImage("data/mnist_train.csv");
        List<ImageReader> trainImages = new CsvToImageReader().readImage("data/A_Z Handwritten Data.csv");
        List<ImageReader> testImages = new CsvToImageReader().readImage("data/test data.csv");
        //List<ImageReader> testImages = new CsvToImageReader().readImage("data/mnist_test.csv");

        //System.out.println("Train Loaded: " + trainImages.size());
        //System.out.println("Test Loaded: " + testImages.size());


        //ImageReader img = new ImageReader(data, 'a');
        //System.out.println(testImages.get(0).toString());

        NeuralNetworkGenerator builder = new NeuralNetworkGenerator(28, 28, 25600);
        //builder.addFullyConnectedLayer(50, 123, 0.1);
        builder.addConvolutionalLayer(8, 5, 1, 0.1, 123);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(26, 123, 0.1);

        NeuralNetwork network = builder.buildNetwork();

        float rate = network.testNetworkPercentage(testImages);
        //
        //
        //
        //
        //
        //
        //
        System.out.println("Pre-Train: " + rate);

        for(int i = 0; i < 100; i++) {

            shuffle(trainImages);
            network.trainNetwork(trainImages);
            rate = network.testNetworkPercentage(testImages);
            System.out.println("Test " + i  + " - " + rate);
            //System.out.println("Train " + i + " - " + network.testNetworkPercentage(trainImages));

        }

    }

}