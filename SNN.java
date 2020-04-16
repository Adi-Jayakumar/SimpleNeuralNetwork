public class SNN {

    public double lr;
    public int hidden, output;
    public Matrix weightsIH, weightsHO, result, biasH, biasO;

    SNN(int input, int hidden, int output){
        this.hidden = hidden;
        this.output = output;

        this.weightsIH = new Matrix(this.hidden, input);
        this.weightsIH.randomise();

        this.weightsHO = new Matrix(this.output, this.hidden);
        this.weightsHO.randomise();

        this.biasH = new Matrix(this.hidden, 1);
        this.biasH.randomise();

        this.biasO = new Matrix(this.output, 1);
        this.biasO.randomise();

        this.lr = Math.pow(10, -1);
    }

    public void train(Matrix input, Matrix target, boolean training){
        Matrix hiddenInput = Matrix.dot(this.weightsIH, input);
        hiddenInput = Matrix.add(hiddenInput, this.biasH);
        hiddenInput = SNN.activate(hiddenInput, false);
        Matrix out = Matrix.dot(this.weightsHO, hiddenInput);
        out = Matrix.add(out, this.biasO);
        out = SNN.activate(out, false);
        this.result = out;

        if(training){

            Matrix outputError = Matrix.sub(target, out);
            Matrix hiddenError = Matrix.dot(Matrix.transpose(this.weightsHO), outputError);

            Matrix outputGradient = SNN.activate(out, true);
            outputGradient = Matrix.schur(outputGradient, outputError);
            outputGradient = Matrix.scMul(outputGradient, this.lr);
            Matrix HOdeltas = Matrix.dot(outputGradient, Matrix.transpose(hiddenInput));
            this.weightsHO = Matrix.add(this.weightsHO, HOdeltas);
            this.biasO = Matrix.add(this.biasO, outputGradient);

            Matrix hiddenGradient = SNN.activate(hiddenInput, true);
            hiddenGradient = Matrix.schur(hiddenGradient, hiddenError);
            hiddenGradient = Matrix.scMul(hiddenGradient, this.lr);
            Matrix IHdeltas = Matrix.dot(hiddenGradient, Matrix.transpose(input));
            this.weightsIH = Matrix.add(this.weightsIH, IHdeltas);
            this.biasH = Matrix.add(this.biasH, hiddenGradient);

        }

    }


    public static Matrix activate(Matrix m, boolean deriv){
        return Matrix.RelU(m, deriv);
    }


    public Matrix predict(Matrix input){
        this.train(input, new Matrix(0,0), false);
        return this.result;
    }


}
