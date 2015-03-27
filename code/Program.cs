using System;
using CsEdu_3_Perceptron;
using System.Drawing;

/**
 * BENCHMARK RESULTS:
 * Surface Pro 3, Core i5:       xorProblem() | boxCrossProblem()
 *    w/o backpropogation:       482ms        | 64ms
 *    w/  backpropogation:       256ms        | 47ms
 * Lenovo Yoga Pro 2, Core i7,   xorProblem() | boxCrossProblem()
 *    w/o backpropogation:       371ms        | 51ms
 *    w/  backpropogation:       147ms        | 42ms
 * 
 * author: Kevin Zhang
 * version: 201503.26
 */
class Program
{
   static void Main(string[] args)
   {
      ocrProblem();
   }

   static void xorProblem() {
      Network net = new Network(2, 5, 5, 1);

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

   static void boxCrossProblem()
   {
      Network net = new Network(2, 5, 5, 2);

      int N = 2;
      float[][] input = new float[N][];
      float[][] output = new float[N][];

      input[0] = new float[] { // cross (X)
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
                    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                };
      output[0] = new float[] { 1.0f, 0.0f };

      input[1] = new float[] { // box (O)
                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 
                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
                };
      output[1] = new float[] { 0.0f, 1.0f };

      net.train(N, input, output);
      for (int n = 0; n < N; n++)
         Console.WriteLine(string.Join(" ", input[n]) + " -> " + string.Join(" ", net.feed(input[n])));
   }

   static void ocrProblem()
   {
      int N = 26;
      float[][] input = new float[N][];
      float[][] output = new float[N][];
      for (char c = 'A'; c < 'Z'; c++)
      {
         int n = (int)(c - 'A');
         Bitmap img = new Bitmap(Image.FromFile("img/" + c + ".bmp"));

         int i = 0;
         input[n] = new float[img.Height * img.Width];
         for (int row = 0; row < img.Height; row++)
            for (int col = 0; col < img.Width; col++)
               input[n][i++] = img.GetPixel(col, row).GetBrightness();

         output[n] = new float[N];
         output[n][n] = 1.0f;
      }

      int S, R, Q, P;
      S = input[0].Length;
      P = output[0].Length;
      R = (new Random()).Next(P, S/2);
      Q = (new Random()).Next(P, S/3);
      Console.WriteLine(S + " -> " + R + " -> " + Q + " -> " + P);
      Network net = new Network(S, R, Q, P);
      net.train(N, input, output);
   }
}
