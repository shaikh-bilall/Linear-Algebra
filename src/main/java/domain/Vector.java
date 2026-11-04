package domain;

import com.google.common.base.Preconditions;
import org.jspecify.annotations.NonNull;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * A vector of finite real numbers, with basic arithmetic: addition, scaling, dot product and length. It can be
 * iterated over entry by entry.
 *
 * @param data the entries in order; {@code data()} returns this array itself, not a copy
 */
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

    /**
     * Returns the entrywise sum of this vector and another of the same length.
     *
     * @param other the vector to add
     * @return a new vector holding the sum
     * @throws NullPointerException if {@code other} is {@code null}
     * @throws IllegalStateException if the lengths differ, the sum overflows, or either vector is invalid
     */
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
    /**
     * Returns this vector multiplied by a scalar.
     *
     * @param scalar the finite factor to multiply every entry by
     * @return a new vector holding the product
     * @throws IllegalStateException if {@code scalar} is not finite, the product overflows, or this vector is invalid
     */
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
    /**
     * Returns the dot product of this vector and another of the same length.
     *
     * @param other the vector to take the dot product with
     * @return the dot product
     * @throws NullPointerException if {@code other} is {@code null}
     * @throws IllegalStateException if the lengths differ, the result overflows, or either vector is invalid
     */
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
    /**
     * Returns the Euclidean length of this vector.
     *
     * @return the length, never negative
     * @throws IllegalStateException if the result overflows, or this vector is invalid
     */
    public double norm() {
        double norm = 0;
        for (double v : data) {
            norm = Math.hypot(norm, v); // hypot avoids overflowing on the intermediate squares
        }
        Preconditions.checkState(Double.isFinite(norm), "Norm must be finite");
        return norm;
    }
    /**
     * Reorders the entries: entry {@code i} of the result is entry {@code shuffledIndices.get(i)} of this vector.
     *
     * @param shuffledIndices each index from {@code 0} to {@code length() - 1}, exactly once
     * @return a new vector with the entries reordered
     * @throws NullPointerException if {@code shuffledIndices} or any of its elements is {@code null}
     * @throws IllegalStateException if the list is not a permutation of the indices, or this vector is invalid
     */
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
    /**
     * Returns a single entry.
     *
     * @param index the zero-based position
     * @return the entry at that position
     * @throws IllegalStateException if {@code index} is out of range, or this vector is invalid
     */
    public double get(
            final int index
    ) {
        Preconditions.checkState(index >= 0, "Index must be non-negative");
        Preconditions.checkState(index < data.length, "Index out of bounds");

        double value = data[index];
        Preconditions.checkState(Double.isFinite(value), "Entry must be finite");
        return value;
    }
    /**
     * Returns the number of entries.
     *
     * @return the length
     * @throws IllegalStateException if this vector is invalid
     */
    public int length() {
        Preconditions.checkState(data.length > 0, "Vector must have at least one element");
        return data.length;
    }

    /**
     * Iterates over the entries in order.
     *
     * @return an iterator over the entries
     * @throws IllegalStateException if this vector is invalid
     */
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

    /**
     * Compares this vector to another object for equality, entry by entry. Two vectors are equal when they have the
     * same length and every pair of matching entries is numerically equal, so {@code 0.0} and {@code -0.0} count as
     * the same entry.
     *
     * @param otherObject the object to compare with
     * @return {@code true} if {@code otherObject} is a vector with the same length and entries
     * @throws IllegalStateException if this vector, or the other one, is invalid
     */
    @Override
    public boolean equals(
            final Object otherObject
    ) {
        checkVector();
        if (this == otherObject) {
            return true;
        }
        if (!(otherObject instanceof Vector otherVector)) {
            return false;
        }
        otherVector.checkVector();
        return entriesEqual(data, otherVector.data);
    }
    /**
     * Returns a hash code that agrees with {@link #equals}: vectors that are equal have the same hash code.
     *
     * @return the hash code of the entries
     * @throws IllegalStateException if this vector is invalid
     */
    @Override
    public int hashCode() {
        checkVector();
        return entriesHashCode(data);
    }

    // Checks that the other vector is valid and of the same length as this one.
    private void checkOtherVector(
            final Vector other
    ) {
        Preconditions.checkNotNull(other, "Other vector must not be null");
        Preconditions.checkState(other.length() > 0, "Other vector must have at least one element");
        Preconditions.checkState(other.length() == this.length(),
                "Vectors must be of the same length: %s != %s", other.length(), this.length());
    }
    // Checks that this vector is valid: non-null, non-empty, and all entries finite.
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
    // Returns a new array of the given size, filled with zeros.
    private static double[] zeros(
            final int size
    ) {
        Preconditions.checkArgument(size > 0, "Vector size must be positive");
        return new double[size];
    }

    /**
     * Tells whether two arrays have the same length and numerically equal entries, so {@code 0.0} equals {@code -0.0}.
     *
     * @param first  the first array
     * @param second the second array
     * @return {@code true} if the arrays have the same length and every pair of matching entries is equal
     * @throws NullPointerException if either array is {@code null}
     */
    static boolean entriesEqual(
            final double[] first,
            final double[] second
    ) {
        Preconditions.checkNotNull(first, "First array must not be null");
        Preconditions.checkNotNull(second, "Second array must not be null");
        if (first.length != second.length) {
            return false;
        }
        for (int i = 0; i < first.length; i++) {
            if (first[i] != second[i]) {
                return false;
            }
        }
        return true;
    }
    /**
     * Returns a hash code of the entries of an array that agrees with {@link #entriesEqual}, so {@code 0.0} and
     * {@code -0.0} give the same hash code.
     *
     * @param entries the array to hash
     * @return the hash code
     * @throws NullPointerException if {@code entries} is {@code null}
     */
    static int entriesHashCode(
            final double[] entries
    ) {
        Preconditions.checkNotNull(entries, "Entries must not be null");
        int result = 1;
        for (double entry : entries) {
            result = 31 * result + Double.hashCode(entry + 0.0); // adding 0.0 turns -0.0 into 0.0
        }
        return result;
    }

}