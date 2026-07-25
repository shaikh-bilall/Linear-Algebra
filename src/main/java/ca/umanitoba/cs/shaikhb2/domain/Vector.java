package ca.umanitoba.cs.shaikhb2.domain;

import com.google.common.base.Preconditions;
import org.jspecify.annotations.NonNull;

import java.util.Iterator;
import java.util.List;

public record Vector(double[] data) implements Iterable<Double> {

    public Vector(final double[] data) {
        this.data = data.clone();
        checkVector();
    }
    public Vector(int size) {
        this(new double[size]);
    }

    public Vector add(final Vector other) {
        checkOtherVector(other);
        double[] result = data.clone();
        for (int i = 0; i < data.length; i++) {
            result[i] += other.data[i];
        }
        return new Vector(result);

    }
    public Vector scale(final double scalar) {
        double[] result = data.clone();
        for (int i = 0; i < data.length; i++) {
            result[i] *= scalar;
        }
        return new Vector(result);
    }
    public double dot(final Vector other) {
        checkOtherVector(other);
        double result = 0;
        for (int i = 0; i < data.length; i++) {
            result += this.data[i] * other.data[i];
        }
        return result;
    }
    public double norm() {
        double sum = 0;
        for (double v : data) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    public Vector permute(final List<Integer> shuffledIndices) {
        Preconditions.checkNotNull(shuffledIndices, "Shuffled indices list must not be null");
        Preconditions.checkArgument(shuffledIndices.size() == length(), "Shuffled indices list size must match number of rows in the matrix");
        checkVector();

        double[] shuffledData = new double[length()];
        for (int i = 0; i < length(); i++) {
            shuffledData[i] = data[shuffledIndices.get(i)];
        }

        return new Vector(shuffledData);
    }

    public double get(final int index) {
        Preconditions.checkState(index >= 0, "Index must be non-negative");
        Preconditions.checkState(index < data.length, "Index out of bounds");

        return data[index];
    }

    public int length() {
        return data.length;
    }

    @Override@NonNull
    public Iterator<Double> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < data.length;
            }

            @Override
            public Double next() {
                return data[currentIndex++];
            }
        };
    }

    private void checkOtherVector(final Vector other) {
        Preconditions.checkNotNull(other, "Other vector must not be null");
        Preconditions.checkState(other.length() > 0, "Other vector must have at least one element");
        Preconditions.checkState(other.length() == this.length(),
                "Vectors must be of the same length: %s != %s", other.length(), this.length());
    }
    private void checkVector() {
        Preconditions.checkNotNull(data, "Vector data must not be null");
        Preconditions.checkState(data.length > 0, "Vector must have at least one element");
    }
}
