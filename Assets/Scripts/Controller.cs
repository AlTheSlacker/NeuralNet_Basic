using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Controller : MonoBehaviour
{
    [SerializeField] private int inputs = 1;
    [SerializeField] int[] hiddenLayers = new int[] { 3, 3 };
    [SerializeField] bool[] addBias = new bool[] { false, true, true, false };
    [SerializeField] AFType[] afFunctions = new AFType[] { AFType.Base, AFType.LeakyReLu, AFType.Sigmoid, AFType.Base };
    [SerializeField] int outputs = 1;
    [SerializeField] double learningRate = 0.05;

    void Start()
    {

        // create the network
        Network net = new Network(inputs, hiddenLayers, afFunctions, outputs, addBias, learningRate);

        // train network
        double[][] sampledata = new double[10][] { new double[] { 0.0 }, new double[] { 10.0 }, new double[] { 20.0 }, new double[] { 30.0 }, new double[] { 40.0 }, new double[] { 50.0 }, new double[] { 60.0 }, new double[] { 70.0 }, new double[] { 80.0 }, new double[] { 90.0 } };
        double[][] targetdata = new double[10][] { new double[] { 0.0 }, new double[] { 0.173648 }, new double[] { 0.34202 }, new double[] { 0.5 }, new double[] { 0.642788 }, new double[] { 0.766044 }, new double[] { 0.866025 }, new double[] { 0.93969 }, new double[] { 0.9848 }, new double[] { 1.0 } };
        int epochs = 20000;
        net.BackPropagate(sampledata, targetdata, epochs);

        // define specific input and run trained network
        double[] inputValues = new double[] { 45 };
        net.InputsAdd(inputValues);
        net.FeedForward();

        // get network output
        double[] Out = new double[outputs];
        Out = net.OutputsGet();
        print("Final output 0: " + Out[0]);
        print("Finished");

    }

}
