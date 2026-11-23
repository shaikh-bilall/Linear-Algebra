package linearAlgebra.logic;

import com.google.common.base.Preconditions;

import linearAlgebra.domain.Diagonalization;
import linearAlgebra.logic.exceptions.ComplexEigenvaluesException;
import linearAlgebra.logic.exceptions.DefectiveMatrixException;
import linearAlgebra.logic.exceptions.DependentEigenvectorsException;
import linearAlgebra.logic.exceptions.InverseNotAvailableException;
import linearAlgebra.domain.exceptions.MatrixMultiplyDimMismatchException;
import linearAlgebra.domain.Matrix;
import linearAlgebra.domain.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Defines the diagonalization operation for a real square {@link Matrix} as {@code A = P D P^-1},
 * using real numbers only.
 */
public final class DiagonalizationLogic {

    // Machine epsilon: the spacing of doubles just above 1.
    // Drives the QR iteration's deflation test.
    private static final double EPSILON = Math.ulp(1.0);
    // QR sweeps allowed on the active window before giving up;
    // the count restarts whenever an eigenvalue splits off.
    // Ordinary matrices need a handful,
    // but defective ones converge slowly (a 4x4 Jordan block took about 50).
    private static final int MAX_ITERATIONS = 100;
    // Relative to the largest absolute entry of the input;
    // eigenvalues closer than this are one repeated eigenvalue,
    // and an imaginary part this small counts as zero.
    private static final double CLUSTER_TOLERANCE = 1e-6;
    // Relative to the largest absolute entry of the input;
    // entries of T - lambda*I this small are rounding noise.
    private static final double ZERO_TOLERANCE = 1e-10;
    // In an RREF every pivot is 1 and everything else in a pivot's row to its left is rounding noise.
    private static final double PIVOT_THRESHOLD = 0.5;
    // Entries within this relative distance of each other count as equally large
    // when fixing an eigenvector's sign.
    private static final double SIGN_TIE_TOLERANCE = 1e-9;

    // Prevents instantiation.
    private DiagonalizationLogic() {
    }

    /**
     * Computes the {@link Diagonalization} of the {@link Matrix}.
     *
     * @return the factorization,
     *         with the eigenvalues ordered by decreasing magnitude and
     *         the columns of {@code P} the matching unit eigenvectors
     * @throws ComplexEigenvaluesException if the {@link Matrix} has complex eigenvalues
     * @throws DefectiveMatrixException if the {@link Matrix} is defective
     * @throws DependentEigenvectorsException if the eigenvectors of the {@link Matrix} are linearly dependent
     */
    public static Diagonalization diagonalize(
        final Matrix A
    )
        throws ComplexEigenvaluesException,
        DefectiveMatrixException,
        DependentEigenvectorsException
    {
        Preconditions.checkNotNull(
            A,
            "DiagonalizationLogic.diagonalize: matrix must not be null"
        );

        final int n = A.rows();
        final double scale = getScale(A);
        final double tolerance = CLUSTER_TOLERANCE * scale;

        // A = Q T Q^T with Q orthogonal and T upper triangular; T and Q are private working arrays.
        final double[][] T = A.data();
        final double[][] Q = Matrix.identity(n).data();
        reduceToHessenberg(T, Q);
        reduceToSchur(T, Q, tolerance);

        final Matrix orthogonal = new Matrix(Q);
        final double[][] eigenvectorData = new double[n][n];
        final double[][] diagonalData = new double[n][n];
        int column = 0;
        for (var cluster : getClusters(T, tolerance)) {
            for (var eigenvector : getEigenvectors(T, orthogonal, cluster, scale)) {
                for (int i = 0; i < n; i++) {
                    eigenvectorData[i][column] = eigenvector.get(i);
                }
                diagonalData[column][column] = cluster.value();
                column++;
            }
        }
        Preconditions.checkState(
            column == n,
            "DiagonalizationLogic.diagonalize: every eigenvalue must contribute exactly one eigenvector"
        );

        final Matrix P = new Matrix(eigenvectorData);
        final Matrix inverseP = getInverse(P);
        return new Diagonalization(P, new Matrix(diagonalData), inverseP);
    }

    // Reduces H to Hessenberg form with Householder reflections, accumulating them in Q.
    private static void reduceToHessenberg(
        final double[][] H,
        final double[][] Q
    ) {
        final int n = H.length;
        for (int k = 0; k < n - 2; k++) {
            final double[] v = new double[n - k - 1];
            double below = 0;
            for (int i = 0; i < v.length; i++) {
                v[i] = H[k + 1 + i][k];
                if (i > 0) {
                    below += v[i] * v[i];
                }
            }
            if (below == 0) {
                continue; // nothing to clear below the subdiagonal in this column
            }

            toReflector(v);
            reflectRows(H, v, k + 1, k, n);
            reflectColumns(H, v, k + 1, n);
            reflectColumns(Q, v, k + 1, n);
            for (int i = k + 2; i < n; i++) {
                H[i][k] = 0;
            }
        }
    }

    // Reduces the Hessenberg matrix H to real Schur form by chasing subdiagonal entries to zero.
    private static void reduceToSchur(
        final double[][] H,
        final double[][] Q,
        final double tolerance
    )
        throws ComplexEigenvaluesException
    {
        final double norm = getFrobeniusNorm(H);
        int high = H.length - 1;
        int iterations = 0;
        while (high >= 0) {
            int low = high;
            while (low > 0) {
                double size = Math.abs(H[low - 1][low - 1]) + Math.abs(H[low][low]);
                if (size <= EPSILON * norm) {
                    size = norm;
                    // Both neighbors are about zero, so compare against the whole matrix instead.
                }
                if (Math.abs(H[low][low - 1]) <= EPSILON * size) {
                    H[low][low - 1] = 0;
                    break;
                }
                low--;
            }

            if (low == high) {
                high--; // one eigenvalue split off
                iterations = 0;
            } else if (low == high - 1) {
                standardizeBlock(H, Q, low, tolerance);
                // Two eigenvalues split off together as a 2x2 block.
                high -= 2;
                iterations = 0;
            } else {
                iterations++;
                Preconditions.checkState(
                    iterations <= MAX_ITERATIONS,
                    "DiagonalizationLogic.reduceToSchur: QR iteration did not converge"
                );
                francisStep(H, Q, low, high, iterations);
            }
        }
    }

    // Performs one implicit double-shift QR sweep on the active window.
    private static void francisStep(
        final double[][] H,
        final double[][] Q,
        final int low,
        final int high,
        final int iteration
    ) {
        final int n = H.length;

        // With s1, s2 the shifts:
        // mean = (s1 + s2) / 2 and
        // discriminant = mean^2 - s1 s2 = ((a - d) / 2)^2 + bc.
        final double a = H[high - 1][high - 1];
        final double b = H[high - 1][high];
        final double c = H[high][high - 1];
        final double d = H[high][high];
        double mean = (a + d) / 2;
        double discriminant = ((a - d) / 2) * ((a - d) / 2) + b * c;
        if (iteration == 10 || iteration == 20) { //fixme: exceptional shift bounds
            // Exceptional shift: the ordinary shifts have stalled, so take a deliberately different
            // pair.
            final double size = Math.abs(c) + Math.abs(H[high - 1][high - 2]);
            mean = 0.75 * size; //fixme: Wilkinson shift constants
            discriminant = -0.4375 * size * size; //fixme: Wilkinson shift constants
        }

        for (int k = low; k < high; k++) {
            final int size = Math.min(3, high - k + 1);
            final double[] v = new double[size];
            if (k == low) {
                // First column of (H - s1 I)(H - s2 I); differences from the mean avoid cancellation.
                final double topShifted = H[low][low] - mean;
                v[0] = topShifted * topShifted - discriminant + H[low][low + 1] * H[low + 1][low];
                v[1] = H[low + 1][low] * (topShifted + (H[low + 1][low + 1] - mean));
                v[2] = H[low + 1][low] * H[low + 2][low + 1];
            } else {
                // The bulge sits in column k - 1, just below the subdiagonal.
                for (int i = 0; i < size; i++) {
                    v[i] = H[k + i][k - 1];
                }
            }
            if (!toReflector(v)) {
                continue;
            }

            reflectRows(H, v, k, Math.max(k - 1, low), n);
            reflectColumns(H, v, k, Math.min(k + 3, high) + 1);
            reflectColumns(Q, v, k, n);
            if (k > low) {
                for (int i = 1; i < size; i++) {
                    H[k + i][k - 1] = 0;
                }
            }
        }
    }

    // Splits the 2x2 block at rows p and p + 1, reporting a complex eigenvalue pair.
    private static void standardizeBlock(
        final double[][] H,
        final double[][] Q,
        final int p,
        final double tolerance
    )
        throws ComplexEigenvaluesException
    {
        final int n = H.length;
        final double a = H[p][p];
        final double b = H[p][p + 1];
        final double c = H[p + 1][p];
        final double d = H[p + 1][p + 1];

        // eigenvalues are (a + d) / 2 +/- sqrt(half^2 + bc)
        final double half = (a - d) / 2;
        final double discriminant = half * half + b * c;
        if (discriminant < -tolerance * tolerance) {
            throw new ComplexEigenvaluesException();
        }
        final double root = Math.sqrt(Math.max(discriminant, 0));

        // The eigenvalue is d + z, with z chosen to avoid cancellation and the longer eigenvector used.
        final double z = (half >= 0) ? half + root : half - root;
        double x = z;
        double y = c;
        if (Math.abs(b) + Math.abs(z - 2 * half) > Math.abs(z) + Math.abs(c)) {
            x = b;
            y = z - 2 * half;
        }
        final double length = Math.hypot(x, y);
        if (length == 0) {
            return;
        }
        final double cos = x / length;
        final double sin = y / length;

        // H := G^T H G and Q := Q G, with G = [[cos, -sin], [sin, cos]] acting on indices p and p + 1
        for (int j = p; j < n; j++) {
            final double top = H[p][j];
            final double bottom = H[p + 1][j];
            H[p][j] = cos * top + sin * bottom;
            H[p + 1][j] = cos * bottom - sin * top;
        }
        rotateColumns(H, p, p + 2, cos, sin);
        rotateColumns(Q, p, n, cos, sin);
        H[p + 1][p] = 0;
    }

    // One distinct eigenvalue and how many diagonal entries it covers.
    private record Cluster(double value, int size) {}

    // Groups the diagonal of T into repeated eigenvalues, ordered by decreasing magnitude.
    private static List<Cluster> getClusters(
        final double[][] T,
        final double tolerance
    ) {
        final List<Integer> order = new ArrayList<>();
        for (int i = 0; i < T.length; i++) {
            order.add(i);
        }
        order.sort(Comparator.comparingDouble(i -> T[i][i]));

        final List<Cluster> clusters = createClusters(T, tolerance, order);

        // Round magnitudes to tolerance steps so +1 and -1 tie, then put the larger value first.
        clusters.sort(
            Comparator
                .comparingLong(
                    (Cluster cluster) -> Math.round(Math.abs(cluster.value()) / tolerance)
                )
                .thenComparingDouble(Cluster::value)
                .reversed()
        );
        return clusters;
    }

    private static List<Cluster> createClusters(
        final double[][] T,
        final double tolerance,
        final List<Integer> order
    ) {
        final List<Cluster> clusters = new ArrayList<>();
        double sum = 0;
        int size = 0;
        double previous = 0;
        for (int index : order) {
            final double value = T[index][index];
            if (size > 0 && value - previous > tolerance) {
                clusters.add(new Cluster(sum / size, size));
                sum = 0;
                size = 0;
            }
            sum += value;
            size++;
            previous = value;
        }
        clusters.add(new Cluster(sum / size, size));
        return clusters;
    }

    // Returns a basis of eigenvectors for one eigenvalue, or reports a defective matrix.
    private static List<Vector> getEigenvectors(
        final double[][] T,
        final Matrix Q,
        final Cluster cluster,
        final double scale
    )
        throws DefectiveMatrixException
    {
        final int n = T.length;

        // Scale A down and zero rounding-level entries, which dominate for repeated eigenvalues.
        final double[][] shifted = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                final double entry = ((i == j) ? T[i][j] - cluster.value() : T[i][j]) / scale;
                shifted[i][j] = (Math.abs(entry) <= ZERO_TOLERANCE) ? 0 : entry;
            }
        }
        final Matrix reduced = RowOperationLogic.rref(new Matrix(shifted));

        final int[] pivotColumn = new int[n];
        final boolean[] isPivot = new boolean[n];
        Arrays.fill(pivotColumn, -1);
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (Math.abs(reduced.get(row, col)) > PIVOT_THRESHOLD) {
                    pivotColumn[row] = col;
                    isPivot[col] = true;
                    break;
                }
            }
        }

        final List<Vector> eigenvectors = new ArrayList<>();
        for (int free = 0; free < n && eigenvectors.size() < cluster.size(); free++) {
            if (isPivot[free]) {
                continue;
            }
            final double[] y = new double[n];
            y[free] = 1;
            for (int row = 0; row < n; row++) {
                if (pivotColumn[row] != -1) {
                    y[pivotColumn[row]] = -reduced.get(row, free);
                }
            }
            Preconditions.checkState(
                Q.cols() == y.length,
                "DiagonalizationLogic.getEigenvectors: matrix columns must match vector length"
            );
            try {
                eigenvectors.add(normalize(Q.multiply(new Vector(y))));
            } catch (MatrixMultiplyDimMismatchException ignored) {
                // Unreachable: the dimensions are asserted above.
            }
        }

        if (eigenvectors.size() < cluster.size()) {
            throw new DefectiveMatrixException();
        }
        return eigenvectors;
    }

    // Unit length, with the sign fixed so that the entry of largest magnitude is positive (the first one
    // on a tie).
    private static Vector normalize(
        final Vector v
    ) {
        final Vector unit = v.scale(1 / v.norm());
        int largest = 0;
        for (int i = 1; i < unit.dimension(); i++) {
            if (Math.abs(unit.get(i)) > Math.abs(unit.get(largest)) * (1 + SIGN_TIE_TOLERANCE)) {
                largest = i;
            }
        }
        return (unit.get(largest) < 0) ? unit.scale(-1) : unit;
    }

    // Inverts P, translating a singular P into a dependent-eigenvectors failure.
    private static Matrix getInverse(
        final Matrix P
    )
        throws DependentEigenvectorsException
    {
        final Matrix inverse;
        try {
            inverse = RowOperationLogic.inverse(P);
        } catch (InverseNotAvailableException e) {
            throw new DependentEigenvectorsException();
        }
        return inverse;
    }

    // Turns x into a Householder reflector, returning false when x is zero.
    private static boolean toReflector(
        final double[] x
    ) {
        double norm = 0;
        for (double entry : x) {
            norm = Math.hypot(norm, entry);
        }
        if (norm == 0) {
            return false;
        }
        x[0] += (x[0] >= 0) ? norm : -norm;
        double length = 0;
        for (double entry : x) {
            length = Math.hypot(length, entry);
        }
        for (int i = 0; i < x.length; i++) {
            x[i] /= length;
        }
        return true;
    }

    // M := (I - 2vv^T) M on the rows row0 ... row0 + v.length - 1, restricted to the columns [col0,
    // col1].
    private static void reflectRows(
        final double[][] M,
        final double[] v,
        final int row0,
        final int col0,
        final int col1
    ) {
        for (int j = col0; j < col1; j++) {
            double dot = 0;
            for (int i = 0; i < v.length; i++) {
                dot += v[i] * M[row0 + i][j];
            }
            for (int i = 0; i < v.length; i++) {
                M[row0 + i][j] -= 2 * dot * v[i];
            }
        }
    }

    // M := M (I - 2vv^T) on the columns col0 ... col0 + v.length - 1, restricted to the rows [row0,
    // row1].
    private static void reflectColumns(
        final double[][] M,
        final double[] v,
        final int col0,
        final int row1
    ) {
        for (int r = 0; r < row1; r++) {
            double dot = 0;
            for (int i = 0; i < v.length; i++) {
                dot += v[i] * M[r][col0 + i];
            }
            for (int i = 0; i < v.length; i++) {
                M[r][col0 + i] -= 2 * dot * v[i];
            }
        }
    }

    // M := M G on the columns p and p + 1, restricted to the rows [row0, row1), with G = [[cos, -sin],
    // [sin, cos]].
    private static void rotateColumns(
        final double[][] M,
        final int p,
        final int row1,
        final double cos,
        final double sin
    ) {
        for (int r = 0; r < row1; r++) {
            final double left = M[r][p];
            final double right = M[r][p + 1];
            M[r][p] = cos * left + sin * right;
            M[r][p + 1] = cos * right - sin * left;
        }
    }

    // The Frobenius norm of M, which is the square root of the sum of the squares of its entries.
    private static double getFrobeniusNorm(
        final double[][] M
    ) {
        double norm = 0;
        for (double[] row : M) {
            for (double entry : row) {
                norm = Math.hypot(norm, entry);
            }
        }
        return norm;
    }

    // The largest absolute entry of A, or 1 for the zero matrix so that dividing by it is always safe.
    private static double getScale(
        final Matrix A
    ) {
        final double maxAbs = A.maxAbs();
        return (maxAbs == 0) ? 1 : maxAbs;
    }
}
