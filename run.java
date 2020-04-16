import java.util.Random;

public class run {

    public static void main(String[] args){

//        SNN neu = new SNN(2, 2, 1);
        NN neu = new NN(new int[]{2, 3, 1});

        Matrix[] inp = new Matrix[4];
        Matrix[] out = new Matrix[4];

        inp[0] = new Matrix(2,1);
        inp[0].m = new double[][]{{0}, {0}};
        inp[0].update();

        inp[1] = new Matrix(2,1);
        inp[1].m = new double[][]{{1}, {0}};
        inp[1].update();

        inp[2] = new Matrix(2,1);
        inp[2].m = new double[][]{{0}, {1}};
        inp[2].update();

        inp[3] = new Matrix(2,1);
        inp[3].m = new double[][]{{1}, {1}};
        inp[3].update();

        out[0] = new Matrix(1,1);
        out[0].m = new double[][]{{0}};
        out[0].update();

        out[1] = new Matrix(1,1);
        out[1].m = new double[][]{{1}};
        out[1].update();

        out[2] = new Matrix(1,1);
        out[2].m = new double[][]{{1}};
        out[2].update();

        out[3] = new Matrix(1,1);
        out[3].m = new double[][]{{0}};
        out[3].update();

        Random rn = new Random();
        int iterations = 100 * 1000;
        for(int i = 0; i < iterations; i++){
            int index = rn.nextInt(inp.length);
//            int index = i % 4;
            neu.train(inp[index], out[index], true);
        }

        for(int j = 0; j < inp.length; j++){
            neu.predict(inp[j]);
            System.out.println(neu.result);
        }

    }

}
