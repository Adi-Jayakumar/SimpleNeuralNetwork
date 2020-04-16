import java.util.Random;
import java.lang.Math;

public class Matrix {

        int rows, cols;
        double[][] m;
        String size;

        Matrix(int rows, int cols){
            this.rows = rows;
            this.cols = cols;
            this.m = new double[rows][cols];
            for(int r = 0; r < this.rows; r++){
                for(int c = 0; c < this.cols; c++){
                    this.m[r][c] = 0;
                }
            }
            this.size = "[" + this.rows + ", " + this.cols + "]";
        }


        @Override
        public String toString(){
            String res = "";
            for(int r = 0; r < this.rows; r++){
                res += "[";
                for(int c = 0; c < this.cols; c++){
                    if(c != this.cols - 1){
                        res += Double.toString(this.m[r][c]) + "\t";
                    }
                    else{
                        res += Double.toString(this.m[r][c]);
                    }
                }
                res += "]\n";
            }
            return res;
        }


        public void update(){
            this.rows = this.m.length;
            this.cols = this.m[0].length;
        }


        public static Matrix transpose(Matrix m){
            Matrix res = new Matrix(m.cols, m.rows);
            for(int i = 0; i < m.rows; i++){
                for(int j = 0; j < m.cols; j++){
                    res.m[j][i] = m.m[i][j];
                }
            }
            return res;
        }


        public void randomise(){
            Random rn = new Random();
//             rn.setSeed(0);
            for(int r = 0; r < this.rows; r++){
                for(int c = 0; c < this.cols; c++){
                    double res = rn.nextDouble();
                    this.m[r][c] = res;
                }
            }
        }


        public static Matrix add(Matrix a, Matrix b){
            Matrix res = new Matrix(a.rows, a.cols);
            for(int i = 0; i < a.rows; i++){
                for(int j = 0; j < a.cols; j++){
                    res.m[i][j] = a.m[i][j] + b.m[i][j];
                }
            }
            return res;
        }


        public static Matrix sub(Matrix a, Matrix b){
            Matrix res = new Matrix(a.rows, a.cols);
            for(int i = 0; i < a.rows; i++){
                for(int j = 0; j < a.cols; j++){
                    res.m[i][j] = a.m[i][j] - b.m[i][j];
                }
            }
            return res;
        }


        public static Matrix scMul(Matrix a, double n){
            Matrix res = new Matrix(a.rows, a.cols);
            for(int i = 0; i < a.rows; i++){
                for(int j = 0; j < a.cols; j++){
                    res.m[i][j] = a.m[i][j] * n;
                }
            }
            return res;
        }


        public static Matrix dot(Matrix a, Matrix b){
            if(a.cols != b.rows){
                System.out.println("Cannot multiply [" + Integer.toString(a.rows) + ", " + Integer.toString(a.cols) + "] with "  + "[" + Integer.toString(b.rows) + ", " + Integer.toString(b.cols) + "]");
                return new Matrix(1,1);
            }
            else{
                Matrix res = new Matrix(a.rows, b.cols);
                for(int i = 0; i < res.rows; i++){
                    for(int j = 0; j < res.cols; j++){
                        double sum = 0;
                        for(int k = 0; k < a.cols; k++){
                            sum += a.m[i][k]  * b.m[k][j];
                        }
                        res.m[i][j] = sum;
                    }
                }
                return res;
            }

        }


        public static Matrix sigmoid(Matrix m, boolean deriv){
            Matrix res = new Matrix(m.rows, m.cols);
            for(int i = 0; i < m.rows; i++){
                for(int j = 0; j < m.cols; j++){
                    if(!deriv){
                        res.m[i][j] = 1/(1 + Math.pow(Math.E, - 1 * m.m[i][j]));
                    }
                    else{
                        res.m[i][j] = (1 - m.m[i][j]) * m.m[i][j];
                    }

                }
            }
            return res;
        }


        public static Matrix tanh(Matrix m, boolean deriv){
            Matrix res = new Matrix(m.rows, m.cols);
            for(int i = 0; i < m.rows; i++){
                for(int j = 0; j < m.cols; j++){
                    if(!deriv){
                        res.m[i][j] = Math.tanh(m.m[i][j]);
                    }
                    else{
                        res.m[i][j] = (1 - m.m[i][j] * m.m[i][j]);
                    }

                }
            }
            return res;
        }


        public static Matrix RelU(Matrix m, boolean deriv){
            Matrix res = new Matrix(m.rows, m.cols);
            for(int i = 0; i < m.rows; i++){
                for(int j = 0; j < m.cols; j++){
                    if(!deriv){
                        res.m[i][j] = Math.max(0, m.m[i][j]);
                    }
                    else{
                        if(m.m[i][j] >= 0){
                            res.m[i][j] = 1;
                        }
                        else{
                            res.m[i][j] = 0;
                        }
                    }

                }
            }
            return res;
        }


        public static Matrix schur(Matrix a, Matrix b){
            Matrix res = new Matrix(a.rows, a.cols);
            for(int i = 0; i < a.rows; i++){
                for(int j = 0; j < a.cols; j++){
                    res.m[i][j] = a.m[i][j] * b.m[i][j];
                }
            }
            return res;
        }




}

