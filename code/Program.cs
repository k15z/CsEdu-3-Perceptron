using CsEdu_3_Perceptron;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Program
{
   static void Main(string[] args)
   {
      Network net = new Network(2, 5, 4, 1);
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

      while (true)
      {
         net.train(N, input, output);
         for (int n = 0; n < N; n++)
         {
            Console.WriteLine(string.Join(" ", input[n]) + " -> " + string.Join(" ", net.feed(input[n])));
         }
         Console.WriteLine("Press \"Enter\" to continue training...");
         Console.ReadLine();
      }
   }
}
