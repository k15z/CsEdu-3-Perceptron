using CsEdu_3_Perceptron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/**
 * This class contains the main entry point for this project. It creates a new Network and attempts to 
 * teach the network to recognize XOR. After training the Network, it prints the result of feeding the 
 * Network all four possible binary inputs and terminates. This class was written primarily to benchmark 
 * the speed of regular steepest gradient descent and memoizized SGD (backpropogation).
 * 
 * BENCHMARK RESULTS:
 * Surface Pro 3, Intel Core i5
 *    w/o backpropogation: 482ms
 *    w/  backpropogation: 256ms
 * Lenovo Yoga Pro 2, Intel Core i7
 *    w/o backpropogation: 371ms
 *    w/  backpropogation: 147ms
 * 
 * author: Kevin Zhang
 * version: 201503.26
 */
class Program
{
   static void Main(string[] args)
   {
      Network net = new Network(2, 5, 5, 1);
      net.feed(new float[] { 0.0f, 0.0f });

      int N = 4;
      float[][] input = new float[N][];
      float[][] output = new float[N][];

      input[0] = new float[] { 0.0f, 0.00f };
      output[0] = new float[] { 0.0f };

      input[1] = new float[] { 1.0f, 0.00f };
      output[1] = new float[] { 1.0f };

      input[2] = new float[] { 0.0f, 1.00f };
      output[2] = new float[] { 1.0f };

      input[3] = new float[] { 1.0f, 1.00f };
      output[3] = new float[] { 0.0f };

      net.train(N, input, output);
      for (int n = 0; n < N; n++)
         Console.WriteLine(string.Join(" ", input[n]) + " -> " + string.Join(" ", net.feed(input[n])));
   }
}
