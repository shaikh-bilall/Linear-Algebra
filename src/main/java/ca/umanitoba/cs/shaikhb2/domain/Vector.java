package ca.umanitoba.cs.shaikhb2.domain;

import com.google.common.base.Preconditions;
import org.jspecify.annotations.NonNull;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public record Vector(double[] data) implements Iterable<Double> {

    public Vector(
            final double[] data
    ) {
        Preconditions.checkNotNull(data, "Vector data must not be null");
        this.data = data.clone();
        checkVector();
    }
    public Vector(int size) {
        this(zeros(size));
    }
    public Vector add(
            final Vector other
    ) {
        checkOtherVector(other);
        double[] result = data.clone();
        for (int i = 0; i < data.length; i++) {
            result[i] += other.data[i];
        }
        return new Vector(result); // post: same length, finite entries (the constructor rejects an overflowed sum)
    }
    public Vector scale(
            final double scalar
    ) {
        Preconditions.checkState(Double.isFinite(scalar), "Scalar must be finite");
        double[] result = data.clone();
        for (int i = 0; i < data.length; i++) {
            result[i] *= scalar;
        }
        return new Vector(result); // post: same length, finite entries (the constructor rejects an overflowed product)
    }
    public double dot(
            final Vector other
    ) {
        checkOtherVector(other);
        double result = 0;
        for (int i = 0; i < data.length; i++) {
            result += this.data[i] * other.data[i];
        }
        Preconditions.checkState(Double.isFinite(result), "Dot product must be finite");
        return result;
    }
    public double norm() {
        double norm = 0;
        for (double v : data) {
            norm = Math.hypot(norm, v); // hypot avoids overflowing on the intermediate squares
        }
        Preconditions.checkState(Double.isFinite(norm), "Norm must be finite");
        return norm;
    }
    public Vector permute(
            final List<Integer> shuffledIndices
    ) {
        checkVector();
        Preconditions.checkNotNull(shuffledIndices, "Shuffled indices list must not be null");
        Preconditions.checkState(shuffledIndices.size() == length(), "Shuffled indices list size must match the vector length");
        checkPermutation(shuffledIndices);

        double[] shuffledData = new double[length()];
        for (int i = 0; i < length(); i++) {
            shuffledData[i] = data[shuffledIndices.get(i)];
        }

        return new Vector(shuffledData); // post: same length, same entries in a new order
    }
    public double get(
            final int index
    ) {
        Preconditions.checkState(index >= 0, "Index must be non-negative");
        Preconditions.checkState(index < data.length, "Index out of bounds");

        double value = data[index];
        Preconditions.checkState(Double.isFinite(value), "Entry must be finite");
        return value;
    }
    public int length() {
        Preconditions.checkState(data.length > 0, "Vector must have at least one element");
        return data.length;
    }

    @Override@NonNull
    public Iterator<Double> iterator() {
        checkVector();
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < data.length;
            }

            @Override
            public Double next() {
                if (!hasNext()) {
                    throw new NoSuchElementException("No more elements in the vector");
                }
                return data[currentIndex++];
            }
        };
    }

    private void checkOtherVector(
            final Vector other
    ) {
        Preconditions.checkNotNull(other, "Other vector must not be null");
        Preconditions.checkState(other.length() > 0, "Other vector must have at least one element");
        Preconditions.checkState(other.length() == this.length(),
                "Vectors must be of the same length: %s != %s", other.length(), this.length());
    }
    private void checkVector() {
        Preconditions.checkNotNull(data, "Vector data must not be null");
        Preconditions.checkState(data.length > 0, "Vector must have at least one element");
        Preconditions.checkState(Arrays.stream(data).allMatch(Double::isFinite), "All entries must be finite numbers");
    }
    // The list must hold every index from 0 to length - 1 exactly once (its size is checked before this runs).
    private void checkPermutation(
            final List<Integer> indices
    ) {
        boolean[] seen = new boolean[length()];
        for (Integer index : indices) {
            Preconditions.checkNotNull(index, "Shuffled indices must not contain null");
            Preconditions.checkState(index >= 0 && index < seen.length, "Shuffled index out of bounds: %s", index);
            Preconditions.checkState(!seen[index], "Shuffled indices must not repeat: %s", index);
            seen[index] = true;
        }
    }
    private static double[] zeros(
            final int size
    ) {
        Preconditions.checkArgument(size > 0, "Vector size must be positive");
        return new double[size];
    }
}