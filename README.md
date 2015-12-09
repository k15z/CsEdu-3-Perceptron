# CsEdu: 3-Perceptron
** THIS PROJECT HAS BEEN ABANDONED. I WROTE THE CODE BACK IN HIGH SCHOOL AND IT'S HILARIOUSLY BAD IN SOME WAYS. THE MATH STUFF IS OKAY THOUGH...*

This is a neural network based on the multilayer perceptron, with three layers of weights connecting the four layers of nodes. The number of nodes in each of the four layers is configurable, and the network is fully-connected between adjacent layers in the forwards direction. It uses the log-sigmoid function as the activation function, and trains using steepest gradient descent with backpropagation.

_Note: While everything shown here is my own work, the theoretical aspects and general design of the network are heavily influenced by the neural networks course taught by Dr. Eric Nelson at The Harker School._

## OCR Demo
To see this network in action, run the "ocr-demo" application, which creates a new Network and trains it to recognize the 26 images included in the images subdirectory. The number of nodes in the hidden layers are generated randomly every time, so results may vary.
