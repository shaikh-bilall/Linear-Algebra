package linearAlgebra.domain.rref;

/**
 * An elementary row operation.
 */
public sealed interface RowOperation permits RowSwap, RowScale, RowCombine {

    /**
     * Gives the updated value of the determinant after the application of this operation.
     *
     * @param determinant The determinant before applying this operation.
     * @return the determinant after applying this operation.
     */
    double updatedDeterminant(
        final double determinant
    );
}
