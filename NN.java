public class NN {

    int[] structure;
    int size;
    double lr;
    Matrix[] weights, biases;
    Matrix result;

    NN(int[] structure){

        this.structure = structure;
        this.size = this.structure.length;

        this.weights = new Matrix[this.size - 1];
        this.biases = new Matrix[this.size - 1];

        for(int i = 0; i < this.size - 1; i++){
            this.weights[i] = new Matrix(this.structure[i + 1], this.structure[i]);
            this.weights[i].randomise();
            this.biases[i] = new Matrix(this.structure[i + 1], 1);
            this.biases[i].randomise();
        }

        this.lr = Math.pow(10, -2);

    }

    public void train(Matrix input, Matrix target, boolean training){
        Matrix[] inputs = new Matrix[this.size];
        Matrix curInp = input;
        inputs[0] = curInp;
        for(int i = 1; i < inputs.length; i++){
            curInp = Matrix.dot(this.weights[i - 1], curInp);
            curInp = Matrix.add(this.biases[i - 1], curInp);
            curInp = NN.activate(curInp, false);
            inputs[i] = curInp;
        }

        this.result = curInp;

        if(training){
            Matrix[] errors = new Matrix[this.size - 1];
            Matrix curErr = Matrix.sub(target, this.result);
            for(int j = errors.length - 1; j >= 0; j--) {
                errors[j] = curErr;
                curErr = Matrix.dot(Matrix.transpose(this.weights[j]), curErr);
            }
            for(int k = 0; k < errors.length; k++){
//                System.out.println("Inputs");
//                System.out.println(inputs[k + 1]);
//
//                System.out.println("Errors");
//                System.out.println(errors[k]);

                Matrix gradient = NN.activate(inputs[k + 1], true);

//                System.out.println("Input Derivative");
//                System.out.println(gradient);

                gradient = Matrix.schur(gradient, errors[k]);

//                System.out.println("Schur product of input derivative and error");
//                System.out.println(gradient);

                gradient = Matrix.scMul(gradient, this.lr);

//                System.out.println("gradients x lr");
//                System.out.println(gradient);
//
//                System.out.println("Transpose of previous weights layer");
//                System.out.println(Matrix.transpose(inputs[k]));

                Matrix delta = Matrix.dot(gradient, Matrix.transpose(inputs[k]));
                this.weights[k] = Matrix.add(this.weights[k], delta);

//                System.out.println("Affected weights");
//                System.out.println(this.weights[k]);

//                System.out.println("Bias");
//                System.out.println(this.biases[k]);

                this.biases[k] = Matrix.add(this.biases[k], gradient);

//                System.out.println("Delta for bias");
//                System.out.println(gradient);
//
//                System.out.println("Affected bias");
//                System.out.println(this.biases[k]);

            }

        }
    }


    public static Matrix activate(Matrix m, boolean deriv){
        return Matrix.tanh(m, deriv);
    }


    public Matrix predict(Matrix input){
        this.train(input, new Matrix(1,1), false);
        return this.result;
    }



}
