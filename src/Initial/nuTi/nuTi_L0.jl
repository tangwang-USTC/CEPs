# using KahanSummation

nModL0 = 3


naiL0 = rand(nModL0)
naiL0 /= sum_kbn(naiL0)
# naiL00 = naiL0 / sum_kbn(naiL0)
# sum(naiL00)-1, sum(naiL0)-1


vthiL0 = rand(nModL0)

uaiL0 = randn(nModL0)
uaiL0 /= (maximum(abs.(uaiL0)) * 5)

njL0 = 3 * nModL0
njML0 = njL0 + 3
jvecL0 = 0:2:2(njL0-1) |> Vector{Int}

mathtype = :Exact       # [:Exact, :Taylor0, :Taylor1, :TaylorInf]
is_renorm = true
# is_renorm = false
Mhj0 = zeros(njML0)
MhsKMM0!(Mhj0,jvecL0,naiL0,uaiL0,vthiL0,nModL0;is_renorm=is_renorm,mathtype=mathtype)
