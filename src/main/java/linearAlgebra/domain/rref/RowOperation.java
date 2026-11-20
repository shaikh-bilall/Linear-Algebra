package linearAlgebra.domain.rref;

/**
 * An elementary row operation.
 */
public sealed interface RowOperation permits RowSwap, RowScale, RowCombine {
}
