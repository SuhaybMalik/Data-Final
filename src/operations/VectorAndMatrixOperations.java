package operations;

public class VectorAndMatrixOperations {

    public static double[][] sum(double[][] a, double[][] b) {

        double[][] c = new double[a.length][a[0].length];

        for(int i = 0; i < a.length; i++) {

            for(int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }

        }

        return c;

    }

    public static double[] sum(double[] a, double[] b) {

        double[] c = new double[a.length];

        for(int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }

        return c;

    }

    public static double[][] scalarProduct(double[][] a, double b) {

        double[][] c = new double[a.length][a[0].length];

        for(int i = 0; i < a.length; i++) {

            for(int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] * b;
            }

        }

        return c;

    }

    public static double[] scalarProduct(double[] a, double b) {

        double[] c = new double[a.length];

        for(int i = 0; i < a.length; i++) {
            c[i] = a[i] * b;
        }

        return c;

    }

}
