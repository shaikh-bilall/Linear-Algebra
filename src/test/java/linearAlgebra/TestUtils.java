package linearAlgebra;

import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Stateless helpers shared by more than one test class.
 */
public final class TestUtils {

    private static final double DEFAULT_TOLERANCE = 1e-12;
    private static final double[] SCALARS = {-3, -2, -1, -0.5, 0.5, 1, 2, 3};

    // ---------- random data ----------

    // A nonzero scalar, so that every elementary operation built from it is valid.
    public static double randomScalar(
        final Random random
    ) {
        return SCALARS[random.nextInt(SCALARS.length)];
    }

    // Integer entries in [-10, 10].
    public static double[] randomEntries(
        final Random random,
        final int size
    ) {
        double[] entries = new double[size];
        for (int i = 0; i < size; i++) {
            entries[i] = random.nextInt(21) - 10;
        }
        return entries;
    }

    // Integer entries in [-bound, bound].
    public static double[][] randomData(
        final Random random,
        final int rows,
        final int cols,
        final int bound
    ) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = random.nextInt(2 * bound + 1) - bound;
            }
        }
        return data;
    }

    public static List<Integer> shuffledIndices(
        final Random random,
        final int size
    ) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);
        return indices;
    }

    // Permuting by p and then by the inverse of p must restore the original order.
    public static List<Integer> inverseOf(
        final List<Integer> permutation
    ) {
        Integer[] inverse = new Integer[permutation.size()];
        for (int i = 0; i < permutation.size(); i++) {
            inverse[permutation.get(i)] = i;
        }
        return Arrays.asList(inverse);
    }

    // ---------- fixtures and independent oracles ----------

    // A fresh 3x2 array each call, so tests can change one copy and compare against another.
    public static double[][] sampleData() {
        return new double[][]{{1, 2}, {3, 4}, {5, 6}};
    }

    public static Matrix identity(
        final int n
    ) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            data[i][i] = 1;
        }
        return new Matrix(data);
    }

    public static Vector vector(
        final double... values
    ) {
        return new Vector(values);
    }

    // Plain triple loop, deliberately independent of MatrixLogic.multiply.
    public static double[][] product(
        final double[][] left,
        final double[][] right
    ) {
        double[][] result = new double[left.length][right[0].length];
        for (int i = 0; i < left.length; i++) {
            for (int j = 0; j < right[0].length; j++) {
                double sum = 0;
                for (int k = 0; k < right.length; k++) {
                    sum += left[i][k] * right[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    public static double[][] scaled(
        final double[][] data,
        final double factor
    ) {
        double[][] result = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                result[i][j] = factor * data[i][j];
            }
        }
        return result;
    }

    // ---------- assertions ----------

    public static void assertVectorEquals(
        final double[] expected,
        final Vector actual
    ) {
        assertEquals(expected.length, actual.length(), "vector length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual.get(i), DEFAULT_TOLERANCE, "mismatch at index " + i);
        }
    }

    public static void assertMatrixEquals(
        final double[][] expected,
        final Matrix actual
    ) {
        assertMatrixEquals(new Matrix(expected), actual, DEFAULT_TOLERANCE);
    }

    public static void assertMatrixEquals(
        final double[][] expected,
        final Matrix actual,
        final double tolerance
    ) {
        assertMatrixEquals(new Matrix(expected), actual, tolerance);
    }

    public static void assertMatrixEquals(
        final Matrix expected,
        final Matrix actual
    ) {
        assertMatrixEquals(expected, actual, DEFAULT_TOLERANCE);
    }

    public static void assertMatrixEquals(
        final Matrix expected,
        final Matrix actual,
        final double tolerance
    ) {
        assertEquals(expected.rows(), actual.rows(), "row count");
        assertEquals(expected.cols(), actual.cols(), "column count");
        for (int i = 0; i < expected.rows(); i++) {
            for (int j = 0; j < expected.cols(); j++) {
                assertEquals(
                    expected.get(i, j),
                    actual.get(i, j),
                    tolerance,
                    "mismatch at (" + i + ", " + j + ")"
                );
            }
        }
    }
}
