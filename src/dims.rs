pub trait DimValid {}

pub struct Dim<const DIM: usize>;

impl DimValid for Dim<1> {}
impl DimValid for Dim<2> {}
impl DimValid for Dim<3> {}
impl DimValid for Dim<4> {}

pub struct DimPair<const LHS: usize, const RHS: usize>;

pub trait DimEq {}

impl<const DIM: usize> DimEq for DimPair<DIM, DIM> {}

pub trait DimLt {}

impl DimLt for DimPair<1, 2> {}
impl DimLt for DimPair<1, 3> {}
impl DimLt for DimPair<1, 4> {}
impl DimLt for DimPair<2, 3> {}
impl DimLt for DimPair<2, 4> {}
impl DimLt for DimPair<3, 4> {}

pub trait DimGtE {}

impl DimGtE for DimPair<1, 1> {}
impl DimGtE for DimPair<2, 2> {}
impl DimGtE for DimPair<3, 3> {}
impl DimGtE for DimPair<4, 4> {}
impl<const LHS: usize, const RHS: usize> DimGtE for DimPair<LHS, RHS>
where
    Dim<LHS>: DimValid,
    Dim<RHS>: DimValid,
    DimPair<RHS, LHS>: DimLt,
{
}
