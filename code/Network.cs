using System;

/**
 * This file contains a C# implementation of the multilayer perceptron with three layers of weights that 
 * is described in the document "3-perceptron.pdf". The variable names correspond with those used in the 
 * document. Superscript (in the document) becomes the suffix of variable name, while subscript (in the 
 * document) becomes the index/indices into the array. The names of the arrays used for backpropagation 
 * memoization are prefixed with "bp_", while the array used to memoize the adaptive learning rate is 
 * prefixed with "lr_".
 * 
 * author: Kevin Zhang
 * version: 201503.26
 */
namespace CsEdu_3_Perceptron
{
   /**
    * The `Utils` class holds one-line methods which are frequently used by the Network. These three 
    * little methods generate random numbers, calculate the sigmoid of x, and calculate the derivative of 
    * the sigmoid at x. Since these methods are so ridiculously short and self-explanatory, they are not 
    * properly documented.
    */
   class Utils
   {
      private static Random rng = new Random();
      public static float random() { return (float)(rng.NextDouble() * 2.0f - 1.0f); }
      public static float sigmoid(float x) { return (float)(1 / (1 + Math.Exp(-x))); }
      public static float dSigmoid(float x) { x = sigmoid(x); return x * (1.0f - x); }
   }

   /**
    * The public `Network` class represents a multilayer perceptron with three layers of weights. This 
    * network is fully connected between adjacent layers in the forwards direction, and uses the sigmoid 
    * function (provided by the `Utils` class) as the activation function. It stores node values (both 
    * pre-activation `n` and post-activation `a`) in float arrays, and it stores weight values (`w`) in 
    * two-dimension float arrays. The weights are initially set to random numbers (provided by the 
    * `Utils` class), and then the `train` and `feed` functions can be used, respectively, to teach and 
    * test the network.
    */
   public class Network
   {
      private int S, R, Q, P;
      public float[] n1, n2, n3, n4;
      public float[] a1, a2, a3, a4;
      public float[,] w12, w23, w34;

      /**
       * This creates a new Network object where parameters `S`, `R`, `Q`, and `P` are the number of 
       * nodes in each of the four layers of nodes. The three layers of weights sandwiched in-between 
       * the node layers are two-dimensional arrays which are initialized to random values provided 
       * by the `Utils` class.
       */
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

      /**
       * This feeds the input data through the network, using the equations listed on page one of the 
       * document "3-perceptron.pdf". First, it loads the input data into `a1`, which is the first 
       * array of nodes. Note that the activation function is NOT applied to the input data, and that 
       * the `n1` array is never actually used. It is simply there for consistancy, and in case this 
       * design changes in the near future.
       * 
       * The function then moves through the network from left to right (input nodes to output nodes), 
       * adding up the signals to create the pre-activation node values `n#` and then applying the 
       * sigmoid function (provided by the `Utils` class) to create the post-activation node values 
       * which are stored in `a#`.
       * 
       * Finally, the `a4` array containing the values in the output layer is returned. Note that the 
       * returned value is a direct pointer to the Network's internal node representation, so DO NOT 
       * modify the array.
       */
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

      /**
       * This trains the network until either the maximum number of attempts has been exceeded or the 
       * maximum of points has been reached. Points are awarded for each correct output, where a correct 
       * output is defined as the actual value being no more than the `MARGIN_OF_ERROR` away from the 
       * expected value. The score is calculated after every `RESCORE_INTERVAL` training cycles, and if 
       * a perfect score is achieved, the while loop ends.
       * 
       * The while loop is divided into two parts, "adjust weights" and "calculate score". In the first 
       * phase, every weight value in the network is updated, using the equations shown on page two of 
       * the document "3-perceptron.pdf". Backpropogation is implemented using the memoization tables 
       * `bp_p` and `bp_q_p` which store the intermediate values used in calculating the derivative, 
       * underlined on page two of the document in blue and orange, respectively. Also, the `lr_p` array 
       * holds the adaptive learning rates for each of the `p` outputs. The learning rate is simply the 
       * absolute value of the difference between actual and expected, so a larger difference (error) 
       * creates a larger learning rate.
       * 
       * In the "calculate score" phase, the number of points is reset to zero, and if the current attempt 
       * is a "rescore attempt" (determined using the RESCORE_INTERVAL), then all the inputs are fed into 
       * the networks and the results are compared to the outputs using the `MARGIN_OF_ERROR`.
       * 
       * Finally, no matter how the while loop terminates (either MAX_ATTEMPTS or MAX_POINTS), the method 
       * returns a boolean value indicating whether `points` equals `MAX_POINTS`.
       */
      public bool train(int N, float[][] input, float[][] output)
      {
         int points = 0;
         int MAX_POINTS = N * P;
         float MARGIN_OF_ERROR = 0.25f;

         int attempts = 0;
         int MAX_ATTEMPTS = 1000;
         int RESCORE_INTERVAL = 1;

         while (points < MAX_POINTS && attempts++ < MAX_ATTEMPTS)
         {
            float deriv;
            for (int n = 0; n < N; n++)
            {
               float[] F = this.feed(input[n]);
               float[] T = output[n];

               float[] bp_p = new float[P];
               float[,] bp_q_p = new float[Q, P];

               float[] lr_p = new float[P];
               for (int p = 0; p < P; p++)
                  lr_p[p] = Math.Abs(F[p] - T[p]);

               for (int q = 0; q < Q; q++)
                  for (int p = 0; p < P; p++)
                  {
                     bp_p[p] = (F[p] - T[p]) * Utils.dSigmoid(n4[p]);
                     deriv = bp_p[p] * a3[q];
                     w34[q, p] += -lr_p[p] * deriv;
                  }

               for (int r = 0; r < R; r++)
                  for (int q = 0; q < Q; q++)
                     for (int p = 0; p < P; p++)
                     {
                        bp_q_p[q, p] = w34[q, p] * Utils.dSigmoid(n3[q]);
                        deriv = bp_p[p] * bp_q_p[q, p] * a2[r];
                        w23[r, q] += -lr_p[p] * deriv;
                     }

               for (int s = 0; s < S; s++)
                  for (int r = 0; r < R; r++)
                     for (int p = 0; p < P; p++)
                        for (int q = 0; q < Q; q++)
                        {
                           deriv = bp_p[p] * bp_q_p[q, p] * (w23[r, q] * Utils.dSigmoid(n2[r]) * a1[s]);
                           w12[s, r] += -lr_p[p] * deriv;
                        }

            }

            points = 0;
            if (attempts % RESCORE_INTERVAL == 0)
            {
               for (int n = 0; n < N; n++)
               {
                  float[] F = this.feed(input[n]);
                  float[] T = output[n];
                  for (int p = 0; p < P; p++)
                     if (Math.Abs(F[p] - T[p]) < MARGIN_OF_ERROR)
                        points++;
               }
               Console.WriteLine("Attempt #" + attempts + ": " + points + "/" + MAX_POINTS);
            }
         }

         return (points == MAX_POINTS);
      }
   }
}
