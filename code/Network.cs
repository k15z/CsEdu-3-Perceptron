using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/**
 * This is a naive implementation of the network described in the "3-perceptron.pdf" that 
 * does not use backpropagation. The reason for writing this undocumented, messy  code is 
 * just to make sure my math is correct.
 * 
 * author: Kevin Zhang
 * version: 201503.26
 */
namespace CsEdu_3_Perceptron
{
   class Utils
   {
      private static Random rng = new Random(0);
      public static float random() { return (float)(rng.NextDouble() * 2.0f - 1.0f); }
      public static float sigmoid(float x) { return (float)(1 / (1 + Math.Exp(-x))); }
      public static float d_sigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
   }

   public class Network
   {
      private int S, R, Q, P;
      public float[] n1, n2, n3, n4;
      public float[] a1, a2, a3, a4;
      public float[,] w12, w23, w34;

      public Network(int S, int R, int Q, int P)
      {
         this.S = S; this.R = R; this.Q = Q; this.P = P;

         n1 = new float[S]; a1 = new float[S];
         w12 = new float[S, R];
         n2 = new float[R]; a2 = new float[R];
         w23 = new float[R, Q];
         n3 = new float[Q]; a3 = new float[Q];
         w34 = new float[Q, P];
         n4 = new float[P]; a4 = new float[P];

         for (int s = 0; s < S; s++)
            for (int r = 0; r < R; r++)
               w12[s, r] = Utils.random();
         for (int r = 0; r < R; r++)
            for (int q = 0; q < Q; q++)
               w23[r, q] = Utils.random();
         for (int q = 0; q < Q; q++)
            for (int p = 0; p < P; p++)
               w34[q, p] = Utils.random();
      }

      public float[] feed(float[] input)
      {
         a1 = input;

         for (int r = 0; r < R; r++)
         {
            n2[r] = 0.0f;
            for (int s = 0; s < S; s++)
               n2[r] += a1[s] * w12[s, r];
            a2[r] = Utils.sigmoid(n2[r]);
         }

         for (int q = 0; q < Q; q++)
         {
            n3[q] = 0.0f;
            for (int r = 0; r < R; r++)
               n3[q] += a2[r] * w23[r, q];
            a3[q] = Utils.sigmoid(n3[q]);
         }

         for (int p = 0; p < P; p++)
         {
            n4[p] = 0.0f;
            for (int q = 0; q < Q; q++)
               n4[p] += a3[q] * w34[q, p];
            a4[p] = Utils.sigmoid(n4[p]);
         }

         return a4;
      }

      public bool train(int N, float[][] input, float[][] output)
      {
         for (int n = 0; n < N; n++)
         {
            float[] F = this.feed(input[n]);
            float[] T = output[n];

            for (int q = 0; q < Q; q++)
               for (int p = 0; p < P; p++)
                  w34[q, p] += -1.0f * (F[p] - T[p]) * Utils.d_sigmoid(n4[p]) * a3[q];

            for (int r = 0; r < R; r++)
               for (int q = 0; q < Q; q++)
                  for (int p = 0; p < P; p++)
                     w23[r, q] += -1.0f * (F[p] - T[p]) * Utils.d_sigmoid(n4[p]) * w34[q, p] * Utils.d_sigmoid(n3[q]) * a2[r];

            for (int s = 0; s < S; s++)
               for (int r = 0; r < R; r++)
                  for (int p = 0; p < P; p++)
                     for (int q = 0; q < Q; q++)
                        w12[s, r] += -1.0f * (F[p] - T[p]) * Utils.d_sigmoid(n4[p]) * w34[q, p] * Utils.d_sigmoid(n3[q]) * (w23[r, q] * Utils.d_sigmoid(n2[r]) * a1[s]);

         }
         return false;
      }
   }
}
